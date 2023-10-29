"""Chemical reaction networks
"""
import csv
import re
from copy import deepcopy
from typing import cast, Callable, Dict, Iterable, List, Optional, Tuple, Union
import xarray as xr
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp # type: ignore
import lmfit # type: ignore

Reactants = Tuple[Tuple[str, int], ...]

class CRN:
    """Chemical reaction network

    CRN represents a chemical reaction network, i.e. reactions among
    multisets of species. Kinetics are assumed to follow mass action
    kinetics. CRN supports only irreversible reactions. To model
    reversible reactions, express them as two separate forward and
    backward reactions.

    There are several ways to create a new reaction network:

    >>> from lmfit import Parameter
    >>> crn = CRN([
    ...     ((("A", 1), ("B", 1)), (("C", 1),), Parameter('k_forward', 0.1)),
    ...     ((("C", 1),), (("A", 1), ("B", 1)), Parameter('k_backward', 0.1)),
    ... ])

    This is equivalent to the more convenient function:

    >>> crn = from_string(\"\"\"
    ...    A + B -> C; k_forward = 0.1
    ...    C -> A + B; k_backward = 1.0
    ... \"\"\")

    Instance attributes
    -------------------
    species: pandas.Index
        species names in state vector
    complexes: list of (species, stoichiomentry) pairs
    reactions: mapping of complex pairs to rate constant names
    params: lmfit.Parameters instance of rate constants
    """
    # TODO: support open networks and buffered species

    species: pd.Index
    complexes: List[Reactants]
    reactions: Dict[Tuple[Reactants, Reactants], str]
    params: lmfit.Parameters

    def __init__(self,
                 reactions: Optional[List[Tuple[Reactants, Reactants, lmfit.Parameter]]] = None,
                 species: Optional[Iterable[str]] = None):
        """Create an chemical reaction network.

        Parameters
        ----------
        reactions: list of tuples of educts, products and a rate constant
            See class documentation for details.
        species: list of strings
            If species names are provided, they determine the order of
            the species in the state vector used in crn.integrate and
            crn.rate_law.
        """
        self.species = pd.Index(species or [])
        self.complexes = []
        self.reactions = {}
        self.params = lmfit.Parameters()
        for reaction in reactions or []:
            self.add_reaction(*reaction)

    def __str__(self) -> str:
        def render(complexes, name):
            educts = self._render_reactants(complexes[0])
            products = self._render_reactants(complexes[1])
            return f"{educts} -> {products}; {name}={self.params[name].value}"
        return '\n'.join(render(reaction, name) for reaction, name in self.reactions.items())

    def _repr_html_(self) -> str:
        return (
            '<table>'
            + '\n'.join(
                f'''<tr>
                    <td style="text-align: right">{self._render_reactants(reaction[0])}</td>
                    <td style="text-align: center">&LongRightArrow;</td>
                    <td style="text-align: left">{self._render_reactants(reaction[1])}</td>
                    <td style="text-align: left">{name} = {self.params[name].value:.2g}</td>
                </tr>'''
                for reaction, name in self.reactions.items()
            )
            + '</table>'
        )

    @property
    def complex_graph(self) -> np.ndarray:
        """Complex graph of the reaction network.

        See van der Schaft et al. (2011) SIAM J Appl Math 73(2):953-973
        for details.

        Returns
        -------
        A 2D numpy.array where each row vector gives the stoichiometries
        of each involved species.
        """
        return np.array([
            [
                sum(stoich for species, stoich in compl if species == name)
                for compl in self.complexes
            ]
            for name in self.species
        ])

    @property
    def complex_adjacency(self) -> np.ndarray:
        """Augmented complex graph adjacency matrix.

        See van der Schaft et al. (2011) SIAM J Appl Math 73(2):953-973
        for details.

        Returns
        -------
        A 2D numpy array denoting reaction rate constants among reaction
        complexes.
        """
        return np.array([
            [
                self.params[name] if (name := self.reactions.get((educts, products), '')) else 0.
                for educts in self.complexes
            ]
            for products in self.complexes
        ])

    def scale_concentration_unit(self, scale_factor: float):
        """Scale reaction rate constants to new concentration unit.

        For example, if current rate constants are given in M^-1s^-1, the
        call crn.scale_concentration_unit(1e-9) will rescale those to
        nM^-1s^-1.
        """
        for reaction, name in self.reactions.items():
            self.params[name].value *= scale_factor**(len(reaction[0])-1)

    def add_reaction(self, educts: Reactants, products: Reactants, rate: lmfit.Parameter):
        """Add a reaction to the network.

        Any novel species that occur among the reactants are automatically
        added to the set of species of the network.

        Parameters
        ----------
        educts: tuple of species name, stoichiometry tuples
        products: tuple of species name, stoichiometry tuples
        rate: lmfit.Parameter of the rate constant
        """
        if rate.name in self.params:
            raise ValueError(f"Parameter '{rate.name}' is already used.")

        # collect species and complexes
        self.species = self.species.append(pd.Index([
            name for name, _ in educts+products
            if name not in self.species
        ]))

        for compl in [educts, products]:
            if compl not in self.complexes:
                self.complexes.append(compl)

        self.reactions[educts, products] = rate.name
        self.params.add(rate)

    def __getitem__(self, name: str) -> lmfit.Parameter:
        return self.params[name]

    def __setitem__(self, name: str, value: Union[float, lmfit.Parameter]):
        self.params[name] = value

    def state(self, conc: xr.DataArray) -> xr.DataArray:
        """Generate a state vector with given species concentrations.

        Parameters
        ----------
        conc: an xarray.DataArray with coordinate "species"
            giving species concentrations.

        Returns
        -------
        A DataArray with the same contents as conc, but padded
        with 0's for any unspecified species.
        """
        # TODO: accept different inputs, e.g. iterables, dicts, DataArrays
        return conc.reindex({'species': self.species}, fill_value=0.)

    def rate_law(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Derive mass action kinetic rate function.

        Internally, this method uses the method of van der Schaft et al.
        (2011) SIAM J Appl Math 73(2):953-973.

        Returns
        -------
        A function rate(time: float, state: xarray.DataArray) that
        gives the mass action rate vector for the given state.
        """
        # pylint: disable=invalid-name

        # calculate graph Laplacian
        Z = self.complex_graph
        D = np.diag(np.sum(self.complex_adjacency, axis=0))
        L = D - self.complex_adjacency

        def kinetics(_, state):
            # complex_graph.T @ log(state) with convention 0*inf = 0
            with np.errstate(invalid='ignore'):
                tmp = np.log(state, out=-np.inf*np.ones_like(state), where=state != 0)
                tmp = np.nansum(Z*tmp, axis=0)
            return -Z @ L @ np.exp(tmp)

        return kinetics

    def integrate(self, initial_condition: xr.DataArray,                                         # pylint: disable=invalid-name
                  t_eval: Optional[pd.Index] = None, t0: float = 0., **options) -> xr.DataArray: # pylint: disable=invalid-name
        """Generate trajectory for given initial condition(s).

        If the initial condition is a 1D vector, this returns a
        2D DataArray of states over the requested interval t_eval.

        If the initial condition is a 2D DataArray, the return value is
        a 3D DataArray with trajectories for each initial condition.

        Internally, the method uses scipy.integrate.solve_ivp.
        Optional keyword arguments (method, atol, rtol, etc.) are
        passed to solve_ivp.

        Parameters
        ----------
        initial_condition: 1D or 2D xarray.DataArray
            the last coord must denote species concentrations
        t_eval: pd.Index
            Array of time points at which system states should be reported.
            (Does not influence the numerical step width of integration).
        t0: float
            time point at which integration starts.
        options
            any remaining keyword arguments are passed to
            scipy.optimize.solve_ivp

        Returns
        -------
            2D or 3D DataArray of trajectories. See above.
        """
        initial_condition = self.state(initial_condition)
        t_eval = t_eval if t_eval is not None else pd.Index(np.linspace(0, 100, 101), name="time")
        kinetics = self.rate_law()

        result = xr.DataArray(
            np.zeros(initial_condition.shape+t_eval.shape),
            [(dim, initial_condition.indexes[dim]) for dim in initial_condition.dims]+[t_eval]
        )
        if len(initial_condition.dims) == 1:
            result[0:] = solve_ivp(kinetics, (t0, t_eval[-1]), initial_condition,
                                   t_eval=t_eval, vectorized=True, **options).y
        else:
            for idx, initial in enumerate(initial_condition):
                # TODO: parallelize using multiprocessing.Pool's
                result[idx, 0:] = solve_ivp(kinetics, (t0, t_eval[-1]), initial,
                                            t_eval=t_eval, vectorized=True, **options).y
        return result

    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray,
            conversion: Optional[Callable[[xr.DataArray], xr.DataArray]]=None,
            error: Union[float, xr.DataArray]=1.,
            t0: Optional[lmfit.Parameter]=None,  # pylint: disable=invalid-name
            **options) -> lmfit.minimizer.MinimizerResult:
        """Fit model parameters to experimental data

        Parameters
        ----------
        data: xarray.DataArray with rfu over time
        initial: xarray.DataArray with concentrations of species
        conversion: a function that converts concentrations to RFU values
            The conversion must accept DataArrays of concentrations
            over time and must return a DataArray of RFU values over
            time. Can be obtained from mars.Assay.calibrate.
        options:
            Any remaining keyword arguments are pass to
            lmfit.minimize

        Result
        ------
            An lmfit MinimizerResult that contains (among others) the
            attribute params, which are the optimized parameters.
        """
        conversion = conversion or (lambda conc: conc)
        original = deepcopy(self.params)
        params = self.params
        t0 = t0 if t0 is not None else lmfit.Parameter('t0', value=0., max=0.)
        params.add(t0)
        def objective(params):
            self.params = params
            model = conversion(self.integrate(initial, t_eval=data.time,
                                              t0=params['t0'].value))
            return (data-model)/error
        fit = lmfit.minimize(objective, params, **options)
        self.params = original
        return fit

    @staticmethod
    def _render_reactants(multiset):
        return ' + '.join(
            species if stoich == 1 else f'{stoich} {species}'
            for species, stoich in multiset
        )


class ImpureCRN(CRN):
    """Chemical reaction networks with instantaneous side reactions

    This class models chemical reaction networks with impure side
    reactions among an impure fraction of components. Any species in
    the network can have one side reaction. A fraction of this species
    will engage into the side reaction instantaneously at the beginning
    of a simulation, consuming the impure fraction as much as possible.
    """
    side_reactions: Dict[str, Tuple[Reactants, Reactants, str]]

    def __init__(
        self,
        reactions: Optional[List[Tuple[Reactants, Reactants, lmfit.Parameter]]] = None,
        impurities: Optional[Dict[str, Tuple[Reactants, Reactants, lmfit.Parameter]]] = None,
        species: Optional[Iterable[str]] = None
    ):
        """Create impure reaction network

        Parameters
        ----------
        impurities: mapping from strings to educts, products, and impurity fraction
        """
        super().__init__(reactions, species)
        self.side_reactions = {}

        for impurity, side_reaction in impurities.items() if impurities else []:
            self.add_impurity(impurity, *side_reaction)

    def __str__(self) -> str:
        def render(impurity, side_reaction):
            educt_set, product_set, fraction = side_reaction
            educts = ' + '.join(
                (species if stoich == 1 else f"{-stoich} {species}")
                + (' [impure]' if species == impurity else '')
                for species, stoich in educt_set
            )
            products = ' + '.join(
                species if stoich == 1 else f"{stoich} {species}"
                for species, stoich in product_set
            )
            return f"{educts} -> {products}; {fraction.name}={fraction.value}"
        return super().__str__() + '\n' + '\n'.join(
            render(*side_reaction) for side_reaction in self.side_reactions.items()
        )

    def _repr_html_(self) -> str:
        def render(impurity, side_reaction):
            educt_set, product_set, fraction = side_reaction
            educts = ' + '.join(
                (species if stoich == 1 else f"{-stoich} {species}")
                + (' [impure]' if species == impurity else '')
                for species, stoich in educt_set
            )
            products = ' + '.join(
                species if stoich == 1 else f"{stoich} {species}"
                for species, stoich in product_set
            )
            return educts, products, fraction
        return (
            '<table>'
            + '\n'.join(
                f'''<tr>
                    <td style="text-align: right">{self._render_reactants(reaction[0])}</td>
                    <td style="text-align: center">&LongRightArrow;</td>
                    <td style="text-align: left">{self._render_reactants(reaction[1])}</td>
                    <td style="text-align: left">{name} = {self.params[name].value:.2g}</td>
                </tr>'''
                for reaction, name in self.reactions.items()
            )
            + '\n'.join(
                f'''<tr>
                    <td style="text-align: right">{(res:=render(impurity, side_reaction))[0]}</td>
                    <td style="text-align: center">&LongRightArrow;</td>
                    <td style="text-align: left">{res[1]}</td>
                    <td style="text-align: left">{res[2].name} = {res[2].value:.2g}</td>
                </tr>'''
                for impurity, side_reaction in self.side_reactions.items()
            )
            + '</table>'
        )

    def add_impurity(self, impurity: str, educts: Reactants, products: Reactants,
                     fraction: lmfit.Parameter):
        """Add side reaction for impure species

        Parameters
        ----------
        impurity: string
        educts: tuple of species name, stoichiometry tuples
        products: tuple of species name, stoichiometry tuples
        fraction: float between 0 and 1
        """
        if impurity in self.side_reactions:
            raise ValueError(f"Species {impurity} can only have one declared side reaction.")

        self.side_reactions[impurity] = educts, products, fraction.name
        self.params.add(fraction)

        # collect species
        self.species = self.species.append(pd.Index([
            name for name, _ in educts+products
            if name not in self.species
        ]))

    def integrate(self, initial_condition: xr.DataArray,                                         # pylint: disable=invalid-name
                  t_eval: Optional[pd.Index] = None, t0: float = 0., **options) -> xr.DataArray: # pylint: disable=invalid-name
        fluxes = []
        initial_condition = self.state(initial_condition)
        for impurity, side_reaction in self.side_reactions.items():
            conc = initial_condition.sel(species=impurity)
            educts, products, name = side_reaction

            # compute stoichiometry vector
            stoichiometry = np.zeros_like(self.species)
            for species, stoich in educts:
                stoichiometry[self.species.get_loc(species)] -= stoich
            for species, stoich in products:
                stoichiometry[self.species.get_loc(species)] += stoich

            if not stoichiometry[self.species.get_loc(impurity)] < 0:
                raise ValueError("Side reactions cannot be catalytic")

            flux = np.min([
                -(self.params[name] if conc.species==impurity else 1.)/stoich*conc
                for conc, stoich in zip(initial_condition.T, stoichiometry)
                if stoich < 0
            ], axis=0)
            fluxes.append((flux, stoichiometry))

        for flux, stoichiometry in fluxes:
            if len(initial_condition.dims) == 1:
                initial_condition = initial_condition + flux*stoichiometry
            else:
                initial_condition = initial_condition + flux.reshape(-1,1)*stoichiometry
        return super().integrate(initial_condition, t_eval, t0, **options)


def _parse_reaction(string: str) -> Tuple[Reactants, Reactants, Optional[str]]:
    def parse_complex(string: str) -> Tuple[Reactants, Optional[str]]:
        reactants: Dict[str, int] = {}
        pattern = re.compile(r' *([0-9]*) *\*? *([^\s+0-9][^\s+]*) *(\[impure\])? *')
        impurity: Optional[str] = None
        for expr in string.split(' + '):
            match = pattern.fullmatch(expr)
            if not match:
                raise ValueError(f"Syntax error in reactant: '{expr.strip()}'.")
            name = match.group(2)
            stoich = int(match.group(1)) if match.group(1) else 1
            reactants[name] = reactants.get(name, 0) + stoich
            if match.group(3):
                impurity = name
        return tuple(sorted(reactants.items())), impurity

    educts_string, _, products_string = string.partition('->')
    educts, impurity = parse_complex(educts_string)
    products, prod_impurity = parse_complex(products_string)
    if prod_impurity:
        raise ValueError(f"Impurities not allowed in reaction product: {prod_impurity} [impure]")
    return educts, products, impurity

def from_string(string: str, species: Optional[List[str]] = None) -> Union[CRN, ImpureCRN]:
    """Construct a chemical reaction network from a string representation.

    The format of the string definition is as follows: each reaction is
    specified on a single line. The left hand side and the right hand
    side (reaction complexes) of the reaction are separated by the
    character sequence '->'. Reaction complexes use '+' to separate
    individual chemical species names. Reactions can be followed by a
    semicolon (;) after which the reaction rate constant is specified in
    the format name=value. Both name and value are optional. If no value
    is given, 1 is assumed. If no name is given, a generic name that is
    not used in other reactions is provided. A hash character (#)
    anywhere in the input marks the beginning of a comment that extends
    until the end of the line. See the class docstring for an example.

    If any educt species name is followed by [impure], the function
    returns an ImpureCRN instance and any respectively marked reaction
    is taken to be an instantaneous side reaction.

    Parameters
    ----------
    string: network definition.
        See above.
    species: list of species names
        Sort order of species used in state vectors.

    Returns
    -------
    A CRN instance with the given reactions.
    """
    # TODO: support reversible reactions

    # parse reaction
    reactions: List[Tuple[Reactants, Reactants, Optional[str], float, Optional[str]]] = []

    for raw_line in string.split('\n'):
        line, _, __ = raw_line.partition('#') # remove comments
        line = line.strip()
        if not line:
            continue

        reaction, sep, rate_constant = line.partition(';')

        if '->' in reaction:
            educts, products, impurity = _parse_reaction(reaction)
            # format of rate_constant string: [identifier][=][float-literal]
            if not sep:
                const_name = None
                const_val = 0.01 if impurity else 1.
            else:
                param, equals, value = rate_constant.partition('=')
                if equals:
                    const_name = param.strip()
                    const_val = float(value)
                else:
                    try:
                        const_name = None
                        const_val = float(param)
                    except ValueError:
                        const_name = param.strip()
                        const_val = 1.
            reactions.append((educts, products, const_name, const_val, impurity))

        else:
            raise ValueError(f"Invalid input: {raw_line.strip()}")

    # name unnamed constants
    bound_names = [name for educts, products, name, val, impurity in reactions if name]
    rate_names = [free_name for idx, _ in enumerate(reactions, 1)
                  if (free_name := f'k{idx}') not in bound_names]
    frac_names = [free_name for idx, _ in enumerate(reactions, 1)
                  if (free_name := f'p{idx}') not in bound_names]

    cls = ImpureCRN if any(impurity for *_, impurity in reactions) else CRN
    network = cls(species=species)

    # add reactions
    for educts, products, name, val, impurity in reactions:
        if not name:
            name = rate_names.pop(0) if not impurity else frac_names.pop(0)
        constant = lmfit.Parameter(name, value=val, vary=True, min=0)
        if impurity:
            constant.max=1.
            cast(ImpureCRN, network).add_impurity(impurity, educts, products, constant)
        else:
            network.add_reaction(educts, products, constant)

    return network

def from_kinDA(path: str, species: Optional[List[str]] = None): # pylint: disable=invalid-name
    """Construct a CRN from a kinDA csv file.

    See https://github.com/DNA-and-Natural-Algorithms-Group/KinDA.

    Parameters
    ----------
    A kinDA csv file.

    Returns
    -------
    A CRN instance of the kinDA generated network.
    """
    network = CRN(species=species)
    with open(path, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        # skip to reaction rate data table
        while (row := next(reader)) != ['# REACTION RATE DATA']:
            pass
        # and table header
        assert next(reader)[0] == 'reaction'

        # parse each reaction in the table
        idx = 0 # reaction index
        while len(row := next(reader)) == 5:
            reaction, k_forward, _, k_backward = row[:4]
            educts, products, _ = _parse_reaction(reaction)

            # skip reactions that do not convert species
            if educts == products:
                continue

            pattern = re.compile(r"\[Complex\(([^\)]*)\)\]")
            educts = tuple(
                (cast(re.Match, pattern.match(name)).group(1), stoich)
                for name, stoich in educts
            )
            products = tuple(
                (cast(re.Match, pattern.match(name)).group(1), stoich)
                for name, stoich in products
            )

            k_plus = float(k_forward)
            k_minus = float(k_backward)
            k_effective = k_plus*k_minus/(k_plus+k_minus)

            constant = lmfit.Parameter(f"k{idx}", value=k_effective, vary=True, min=0)
            idx += 1

            network.add_reaction(educts, products, constant)
        return network
