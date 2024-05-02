"""Chemical reaction networks
"""
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import xarray as xr
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp # type: ignore
import lmfit # type: ignore
from . import crn_parser

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
    ...    A + B <=> C; k_forward = 0.1, k_backward = 1.0
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
    def burst_reactions(self) -> Dict[Tuple[Reactants, Reactants], str]:
        """Return subset of reactions with infinite rate constant"""
        return {
            reaction: rate
            for reaction, rate in self.reactions.items()
            if self.params[rate].value == float('inf')
        }

    def get_complex_adjacency(self, burst: bool=False) -> np.ndarray:
        """Augmented complex graph adjacency matrix.

        See van der Schaft et al. (2011) SIAM J Appl Math 73(2):953-973
        for details.

        Returns
        -------
        A 2D numpy array denoting reaction rate constants among reaction
        complexes.
        """
        def get_rate_constant(educts, products):
            value = (self.params[name]
                     if (name := self.reactions.get((educts, products), ''))
                     else 0.)
            if burst:
                return 1 if value == float('inf') else 0.
            return 0 if value == float('inf') else value
        return np.array([
            [
                get_rate_constant(educts, products)
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
        # collect species and complexes
        self.species = self.species.append(pd.Index([
            name for name, _ in educts+products
            if name not in self.species
        ]))

        for compl in [educts, products]:
            if compl not in self.complexes:
                self.complexes.append(compl)

        self.reactions[educts, products] = rate.name

        if rate.name not in self.params:
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
        A = self.get_complex_adjacency()
        L = np.diag(np.sum(A, axis=0)) - A

        def kinetics(_, state):
            # Z.T @ log(state) with convention 0*inf = 0
            with np.errstate(divide='ignore', invalid='ignore'):
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
        if any(param.value==float('inf') for param in self.params.values()):
            initial_condition = self.perform_burst_reactions(self.state(initial_condition))
        t_eval = t_eval if t_eval is not None else pd.Index(np.linspace(0, 100, 101), name="time")
        kinetics = self.rate_law()

        result = xr.DataArray(np.zeros(initial_condition.shape+t_eval.shape),
                              [(dim, initial_condition.indexes[dim])
                               for dim in initial_condition.dims]+[t_eval])
        if len(initial_condition.dims) == 1:
            result[0:] = solve_ivp(kinetics, (t0, t_eval[-1]), initial_condition,
                                   t_eval=t_eval, vectorized=True, **options).y
        else:
            for idx, initial in enumerate(initial_condition):
                # TODO: parallelize using multiprocessing.Pool's
                result[idx, 0:] = solve_ivp(kinetics, (t0, t_eval[-1]), initial,
                                            t_eval=t_eval, vectorized=True, **options).y
        return result

    def perform_burst_reactions(self, state: xr.DataArray) -> xr.DataArray:
        """Perform burst reactions

        The given state state is exposed to burst_reactions and species
        are redistributed according to mass action kinetic proportions until
        an equilibrium is reached. Burst reactions must not be reversible or
        circular.

        Parameters
        ----------
        state: xr.DataArray
            species distribution before burst reactions

        Returns
        -------
            xr.DataArray containing the redistributed species vector
        """
        # pylint: disable=invalid-name
        Z = self.complex_graph
        A = np.where(self.get_complex_adjacency(True), 1., 0)
        L = np.diag(np.sum(A, axis=0)) - A

        iterations = 10*len(self.burst_reactions)
        for _ in range(iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                rates = Z @ L @ np.exp(np.nansum(Z.T*np.log(state.values), axis=1))
            fraction = min(x/y for x, y in zip(state, rates) if y>0).values if rates.any() else 0
            state -= fraction*rates
            if fraction < 1e-10:
                break
        else:
            raise ValueError(f"Burst reactions did not converge within {iterations} steps.")
        return state

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
        conversion: optional function that converts concentrations to RFU values
            The conversion must accept DataArrays of concentrations
            over time and must return a DataArray of RFU values over
            time. Can be obtained from mars.Assay.calibrate.
        error: optional xr.DataArray with rfu over time or float (default 1.)
            Standard deviations of measured data
        t0:  optional lmfit.Parameter
            Time at which the reaction started.
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
        def subspecies_name(species):
            return species.name if not species.suffix else f'{species.name} [{species.suffix}]'
        return ' + '.join(
            subspecies_name(species) if stoich == 1 else f'{stoich} {subspecies_name(species)}'
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
            educt_set, product_set, name = side_reaction
            educts = ' + '.join(
                (species if stoich == 1 else f"{-stoich} {species}")
                + (' [impure]' if species == impurity else '')
                for species, stoich in educt_set
            )
            products = ' + '.join(
                species if stoich == 1 else f"{stoich} {species}"
                for species, stoich in product_set
            )
            return f"{educts} -> {products}; {name}={self.params[name].value}"
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
                    <td style="text-align: left">{res[2]} = {self.params[res[2]].value:.2g}</td>
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


def from_string(string: str, species: Optional[List[str]] = None) -> Union[CRN, ImpureCRN]:
    """Construct a chemical reaction network from a string representation.

    See nanosuite.crn_parser for a definition of the CRN specification language

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
    crn_def = crn_parser.parse(string)

    # create CRN from definition
    crn = CRN(species=species)
    for reaction in crn_def.reactions:
        if isinstance(reaction, crn_parser.Reaction):
            crn.add_reaction(*reaction)
        else:
            crn.add_reaction(reaction.educts, reaction.products, reaction.forward)
            crn.add_reaction(reaction.products, reaction.educts, reaction.backward)

    return crn
