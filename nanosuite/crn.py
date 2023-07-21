"""Chemical reaction networks
"""
import csv
import re
from typing import cast, List, Tuple, Dict, Callable, Optional, Union
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from lmfit import Parameter, Parameters

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
    >>> crn = CRN()
    >>> crn.add_reaction((("A", 1), ("B", 1)), (("C", 1),),
                         Parameter('k_forward', 0.1))
    >>> crn.add_reaction((("C", 1),), (("A", 1) , ("B", 1)),
                         Parameter('k_backward', 1.0))

    This is equivalent to the more convenient function:

    >>> crn = CRN.from_string(\"\"\"
    ...    A + B -> C; k_forward = 0.1
    ...    C -> A + B; k_backward = 1.0
    ... \"\"\")
    
    Instance attributes
    -------------------
        species: list of strings
        complexes: list of species, stoichiomentry pair tuples
        reactions: mapping of complex pairs to lmfit.Parameter instances
    """
    # TODO: support open networks and buffered species
    # TODO: make lmfit.Parameter's optional and if possible transparent.

    complexes: List[Reactants]
    reactions: Dict[Tuple[Reactants, Reactants], Parameter]

    def __init__(self, species: Optional[List[str]]=None):
        """Create an empty reaction network.

        To populate a new CRN with reactions, use crn.add_reaction.
    
        Parameters
        ----------
        species: list of strings
            If species names are provided, they determine the order of
            the species in the state vector used in crn.integrate and
            crn.rate_law
        """
        self.species: List[str] = species or []
        self.complexes: List[Reactants] = []
        self.reactions: Dict[Tuple[Reactants, Reactants], Parameter] = {}

    def _repr_html_(self) -> str:
        def reactants(multiset):
            return ' + '.join(
                species if stoich == 1 else f'{stoich} {species}'
                for species, stoich in multiset
            )
        return (
            '<table>'
            + '\n'.join(
                f'''<tr>
                    <td style="text-align: right">{reactants(reaction[0])}</td>
                    <td style="text-align: center">&LongRightArrow;</td>
                    <td style="text-align: left">{reactants(reaction[1])}</td>
                    <td style="text-align: left">{rate.name} = {rate.value:.2g}</td>
                </tr>'''
                for reaction, rate in self.reactions.items()
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
                self.reactions.get((educts, products), 0.)
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
        for reaction, rate in self.reactions.items():
            rate.value *= scale_factor**(len(reaction[0])-1)

    def add_reaction(self, educts: Reactants, products: Reactants, rate: Parameter):
        """Add a reaction to the network.

        Any novel species that occur among the reactants are automatically
        added to the set of species of the network.

        Parameters
        ----------
        educts: tuple of species names
        products: tuple of species names
        rate: lmfit.Parameter of the rate constant
        """
        if any(current_rate.name == rate.name for current_rate in self.reactions.values()):
            raise ValueError(f"Parameter '{rate.name}' is already used.")

        # collect species and complexes
        for name, _ in educts+products:
            if name not in self.species:
                self.species.append(name)

        for compl in [educts, products]:
            if compl not in self.complexes:
                self.complexes.append(compl)

        self.reactions[educts, products] = rate

    def parameterize(self, params: Parameters=None, **kwds: Dict[str, Union[float, Parameter]]):
        """Parameterize rate constants

        Parameters
        ----------
        params: an lmfit.Parameters objects
        kwds: additional parameters names with either float or Parameter instances
        """
        kwds = {
            kwd_name: kwd_val if isinstance(kwd_val, Parameter) else Parameter(kwd_name, kwd_val)
            for kwd_name, kwd_val in kwds.items()
        }
        if params is None:
            params = Parameters(**kwds)
        else:
            params.update(Parameters(**kwds))

        for reaction, rate_const in self.reactions.items():
            if (name := rate_const.name) in params:
                self.reactions[reaction] = params[name]

    def __getitem__(self, name: str) -> Parameter:
        for _, rate_const in self.reactions.items():
            if rate_const.name == name:
                return rate_const
        raise KeyError(f"No parameter '{name}'")

    def __setitem__(self, name: str, value: Union[float, Parameter]):
        for reaction, rate_const in self.reactions.items():
            if rate_const.name == name:
                if isinstance(value, Parameter):
                    self.reactions[reaction] = value
                else:
                    self.reactions[reaction] = Parameter(name, value)
                break
        raise KeyError(f"No parameter '{name}'")

    def state(self, conc: np.ndarray) -> np.ndarray:
        """Generate a state vector with given species concentrations.

        Parameters
        ----------
        conc: a 1D or 2D numpy array with given species concentrations.

        Returns
        -------
        A numpy array with the same contents as conc, but padded
        with 0's for any unspecified species.
        """
        if conc.shape[-1] > len(self.species):
            raise ValueError(
                f"conc last dimension must be smaller or equal to {len(self.species)}."
            )
        if conc.ndim > 2:
            raise ValueError("conc must have one or two dimensions.")

        exp = len(self.species) - conc.shape[-1]
        return np.pad(conc, (conc.ndim-1)*[(0, 0)] + [(0, exp)])

    def rate_law(self, repeats: int=1) -> Callable[[float, npt.ArrayLike], np.ndarray]:
        """Derive mass action kinetic rate function.

        Internally, this method uses the method of van der Schaft et al.
        (2011) SIAM J Appl Math 73(2):953-973.

        Parameters
        ----------
        repeats: positive integer
            Number of states for which rates should be calculated
            simultaneously.

        Returns
        -------
        A function rate(time: float, state: numpy.array) that gives the
        mass action rate vector for the given state. If repeat is given
        and not equal to 1, the rate function will accept a matrix of states
        and return rates for each provided state.
        """
        # pylint: disable=invalid-name

        # calculate graph Laplacian
        sum_diag = np.diag(np.sum(self.complex_adjacency, axis=0))
        laplacian = sum_diag - self.complex_adjacency

        Z = block_diag(*repeats*[self.complex_graph])
        L = block_diag(*repeats*[laplacian])

        def kinetics(_, state):
            # complex_graph.T @ log(state) with convention 0*inf = 0
            with np.errstate(invalid='ignore'):
                tmp = np.log(state, out=-np.inf*np.ones_like(state), where=state!=0)
                tmp = np.nansum(Z*tmp, axis=0)
            return -Z @ L @ np.exp(tmp)

        return kinetics

    def integrate(self, initial_condition: np.ndarray,
                  t_eval: Optional[np.ndarray]=None, t0: float=0.) -> np.ndarray: # pylint: disable=invalid-name
        """Generate trajectory for given initial condition(s).

        If the initial condition is a 1D vector, this returns a
        2D numpy.array of states over the requested interval t_eval.

        If the initial condition is a 2D matrix, the return value is
        a 3D numpye.array, one trajectory for each initial condition.

        Internally, the method uses scipy.integrate.solve_ivp with
        default parameters.

        Parameters
        ----------
        initial_condition: numpy.array
            1D or 2D initial condition(s).
        t_eval: numpy.array
            Array of time points at which system states should be reported.
            (Does not influence the numerical step width of integration).
        t0: float
            time point at which integration starts.

        Returns
        -------
            2D or 2D numpy.array of trajectories. See above.
        """
        repeats = 1 if len(initial_condition.shape) == 1 else initial_condition.shape[0]
        kinetics = self.rate_law(repeats)
        t_eval = t_eval if t_eval is not None else np.linspace(0, 100, 101)
        res = solve_ivp(kinetics, (t0, t_eval[-1]), initial_condition.flatten(),
                        t_eval=t_eval, vectorized=True)
        return res.y.reshape(initial_condition.shape+t_eval.shape)

    @staticmethod
    def _parse_reaction(string: str) -> Tuple[Reactants, Reactants]:
        def parse_complex(string):
            reactants = {}
            pattern = re.compile(r' *([0-9]*) *\*? *([^\s+0-9][^\s+]*) *')
            for expr in string.split(' + '):
                match = pattern.fullmatch(expr)
                if not match:
                    raise ValueError(f"Syntax error in reaction: '{expr.strip()}'.")
                name = match.group(2)
                stoich = int(match.group(1)) if match.group(1) else 1
                reactants[name] = reactants.get(name, 0) + stoich
            return tuple(sorted(reactants.items()))

        educts_string, _, products_string = string.partition('->')
        educts = parse_complex(educts_string)
        products = parse_complex(products_string)
        return educts, products

    @classmethod
    def from_string(cls, string: str, species: Optional[List[str]]=None):
        """Construct a CRN from a string representation.

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

        Parameters
        ----------
        string: network deifnition.
            See above.
        species: list of species names
            Sort order of species used in state vectors.
        
        Returns
        -------
        A CRN instance with the given reactions.
        """
        # TODO: support reversible reactions
        crn = cls(species=species)

        # parse reaction
        reactions: List[Tuple[Reactants, Reactants, Optional[str], float]] = []
        for raw_line in string.split('\n'):
            line, _, __ = raw_line.partition('#') # remove comments
            line = line.strip()
            if not line:
                continue

            reaction, sep, rate_constant = line.partition(';')

            if '->' in reaction:
                educts, products = cls._parse_reaction(reaction)
                # format of rate_constant string: [identifier][=][float-literal]
                if not sep:
                    reactions.append((educts, products, None, 1))
                else:
                    param, equals, value = rate_constant.partition('=')
                    if equals:
                        reactions.append((educts, products, param.strip(), float(value)))
                    else:
                        try:
                            reactions.append((educts, products, None, float(param)))
                        except ValueError:
                            reactions.append((educts, products, param.strip(), 1.))

            else:
                raise ValueError(f"Invalid input: {raw_line.strip()}")

        # name unnamed constants
        bound_names = [name for educts, products, name, val in reactions]
        free_names = [free_name for idx, _ in enumerate(reactions)
                      if (free_name := f'k{idx}') not in bound_names]

        # add reactions
        for educts, products, name, val in reactions:
            if not name:
                name = free_names.pop(0)
            constant = Parameter(name, value=val, vary=True, min=0)
            crn.add_reaction(educts, products, constant)

        return crn

    @classmethod
    # pylint: disable-next=invalid-name
    def from_kinDA(cls, path: str, species: Optional[List[str]]=None):
        """Construct a CRN from a kinDA csv file.

        See https://github.com/DNA-and-Natural-Algorithms-Group/KinDA.

        Parameters
        ----------
        A kinDA csv file.

        Returns
        -------
        A CRN instance of the kinDA generated network.
        """
        crn = cls(species=species)
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
                educts, products = cls._parse_reaction(reaction)

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

                constant = Parameter(f"k{idx}", value=k_effective, vary=True, min=0)
                idx += 1

                crn.add_reaction(educts, products, constant)

            return crn
