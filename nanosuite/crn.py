"""Chemical reaction networks
"""
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from itertools import chain
import xarray as xr
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp # type: ignore
import lmfit # type: ignore
from . import crn_parser

Reactants = Tuple[Tuple[str, int], ...] # TODO: support generic Tuple[Tuple[T, int], ...]

DEFAULT_INTEGRATION_START = 0
DEFAULT_INTEGRATION_END = 100
DEFAULT_INTEGRATION_POINTS = 501
DEFAULT_MIN_T0 = -np.inf

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
        self.params.add('t0', value=DEFAULT_INTEGRATION_START, min=DEFAULT_MIN_T0, vary=False)
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
    def burst_reactions(self) -> Dict[Tuple[Reactants, Reactants], str]:
        """Return subset of reactions with infinite rate constant"""
        return {
            reaction: rate
            for reaction, rate in self.reactions.items()
            if self.params[rate].value == float('inf')
        }

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
        for reactants in [educts, products]:
            self.species = self.species.append(pd.Index([
                name for name, _ in reactants
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

    def state(self, conc: Optional[Union[xr.DataArray, Dict[str, float]]] = None, /,
              **extra_conc: float) -> xr.DataArray:
        """Generate a state vector with given species concentrations.

        Create a state vector with the given species concentrations.
        Species of the CRN that are ommitted in the input are set to 0.

        >>> initial = crn.state(A=100, B=100)

        Parameters
        ----------
        conc: an xarray.DataArray with a coordinate "species"
              or a dictionary from species labels to float
            giving species concentrations.
        extra_conc:
            named keyword arguemtns of additional concentrations.

        Returns
        -------
        A DataArray with the same contents as conc, but padded
        with 0's for any unspecified species.
        """
        if conc is None:
            return xr.DataArray([extra_conc.get(species, 0.) for species in self.species],
                                {'species': self.species})
        if isinstance(conc, dict):
            conc.update(**extra_conc)
            return xr.DataArray([conc.get(species, 0.) for species in self.species],
                                {'species': self.species})
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

    def integrate(self, initial_condition: xr.DataArray,       # pylint: disable=invalid-name
                  t_eval: Union[Iterable, float, None] = None,
                  **options) -> xr.DataArray:
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
        t_eval: float, tuple, Iterable or None
            Time points at which system states should be reported.
            If t_eval is scalar, the reported range starts at self.params['t0']
            and stops at t_eval. If t_eval is a tuple, the values are taken
            as start and end points. If t_eval is an iterable, those are the
            returned integration points. If t_eval is not provided,
            results are reported between crn.DEFAULT_INTEGRATION_START and
            crn.DEFAULT_INTEGRATON_END with crn.DEFAULT_INTEGRATION_POINTS
            points.
            (t_eval does not influence the numerical step width of
            the integrator).
        options
            any remaining keyword arguments are passed to
            scipy.optimize.solve_ivp

        Returns
        -------
            2D or 3D DataArray of trajectories. See above.
        """
        if isinstance(t_eval, tuple):
            t_eval = (t_eval + (DEFAULT_INTEGRATION_POINTS,))[:3]
            t_eval = pd.Index(np.linspace(*t_eval, dtype=float), name="time")  # type: ignore
        elif isinstance(t_eval, pd.Index):
            pass
        elif isinstance(t_eval, xr.DataArray):
            if t_eval.ndim == 0:
                t_eval = pd.Index(np.linspace(
                    self.params['t0'].value, float(t_eval),
                    DEFAULT_INTEGRATION_POINTS, dtype=float
                ), name="time")
            elif t_eval.ndim == 1:
                pass
            else:
                raise ValueError("t_eval must have either zero or one dimension.")
        elif isinstance(t_eval, Iterable):
            t_eval = pd.Index(t_eval, name="time")
        elif t_eval is None:
            t_eval = pd.Index(np.linspace(self.params['t0'].value,
                                          DEFAULT_INTEGRATION_END,
                                          DEFAULT_INTEGRATION_POINTS, dtype=float),
                              name="time")
        else:
            t_eval = pd.Index(np.linspace(self.params['t0'].value, t_eval,
                                          DEFAULT_INTEGRATION_POINTS, dtype=float),
                              name="time")

        initial_condition = self.state(initial_condition)
        if any(param.value==float('inf') for param in self.params.values()):
            initial_condition = self.perform_burst_reactions(initial_condition)

        kinetics = self.rate_law()

        result = xr.DataArray(np.zeros(initial_condition.shape+t_eval.shape),
                              [(dim, initial_condition.indexes[dim])
                               for dim in initial_condition.dims]+[t_eval])
        if len(initial_condition.dims) == 1:
            result[0:] = solve_ivp(kinetics,
                                   (self.params['t0'], t_eval[-1]),
                                   initial_condition,
                                   t_eval=t_eval, vectorized=True, **options).y
        else:
            for idx, initial in enumerate(initial_condition):
                # TODO: parallelize using multiprocessing.Pool's
                result[idx, 0:] = solve_ivp(kinetics,
                                            (self.params['t0'], t_eval[-1]),
                                            initial,
                                            t_eval=t_eval, vectorized=True, **options).y
        result.name = "concentration"
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
                if len(state.dims) == 1:
                    rates = Z @ L @ np.exp(np.nansum(Z.T*np.log(state.values), axis=1))
                    fraction = min(x/y for x, y in zip(state, rates)
                                   if y > 0).values if rates.any() else 0
                else:
                    tmp = np.zeros((L.shape[0], state.shape[0]))
                    for idx, row in enumerate(state.values):
                        tmp[:, idx] = np.nansum(Z.T*np.log(row), axis=1)
                    rates = Z @ L @ np.exp(tmp)
                    fraction = np.array([min(x/y for x,y in zip(s, r) if y>0) if r.any() else 0
                                        for s, r in zip(state.values, rates.T)])
            state -= (fraction*rates).T
            if np.all(fraction < 1e-10):
                break
        else:
            raise ValueError(f"Burst reactions did not converge within {iterations} steps.")
        return state

    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray,
            conversion: Optional[Callable[[xr.DataArray], xr.DataArray]]=None,
            error: Union[float, xr.DataArray]=1.,
            vary_t0: bool=True,
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
        vary_t0:  optional bool (default True)
            If set to True (default) the fitting is allowed to vary the start
            time of the reaction to lie before the start data.time
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
        params['t0'].vary = vary_t0
        params['t0'].max = float(data.time[0])
        def objective(params):
            self.params = params
            model = conversion(self.integrate(initial, t_eval=data.time))
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


class PartitionedCRN(CRN):
    """CRN with subspecies partitioning

    This subclass allows for modelling of CRNs where certain species are a
    mixture of subspecies. Consider for example a biomarker where 1% is a mutant,
    the rest being wildtype. PartitionedCRN allows one to provide initial
    states in biomarker concentrations, whicu are converted to subspecies
    concentratrations before dynamics are simulated. Before reporting results
    back to the user, overall species concentrations are updated from the
    subspecies concentrations.

    To declare subspecies compositions, PartitionedCRN offers the method
    define_subspecies:

    >>> reactions = from_string(
    ...     "biomarker_mutant + probe -> biomarker_mutant + signal").reactions
    >>> crn = PartitionedCRN(reactions)
    >>> crn.define_subspecies("biomarker",
    ...     subspecies={"mutant": 0.01}, rest="wildtype")

    >>> initial = crn.state(biomarker=100)
    >>> trajectory = crn.integrate(initial)
    >>> total = trajecory.sel(species="biomarker")
    >>> mutant = trajecory.sel(species="biomarker_mutant")
    >>> wildtype = trajecory.sel(species="biomarker_wildtype")


    Internally, the mapping from species space to subspecies space is
    represented by matrix multiplications over the joint species-subspecies
    space, refered to as split (S) and merge (M). If x denotes a species state
    vector. y = x @ S is a vector in species-subspecies space where subspecies
    concentrations are set from x according to the CRN's subspecies definitions.
    Similarly, the merge matrix M sets the concentrations of species by adding
    up all their subspecies concentrations.
    """

    # The matrices S and M are chosen to be idempotent:
    #     x @ S @ S = x @ S
    #     M @ M @ y = M @ y
    # It holds that
    #     x @ S @ M @ S == x @ S
    # but generally not
    #     x @ S @ M == x

    subspecies: Dict[str, Dict[str, str]]
    subspecies_rests: Dict[str, str]


    def __init__(self,
                 reactions: Optional[List[Tuple[Reactants, Reactants, lmfit.Parameter]]] = None,
                 species_defs : Optional[List[Tuple[str, Dict[str, lmfit.Parameter], str]]] = None,
                 species: Optional[Iterable[str]] = None):
        super().__init__(reactions, species)
        self.subspecies = {}
        self.subspecies_rests = {}

        for species_def in species_defs or []:
            self.define_subspecies(*species_def)

    @property
    def split_species(self):
        """Split matrix distributing species into subspecies concentrations
        """
        all_subspecies = set(chain(*self.subspecies.values()))
        result = np.array([[(self.params[self.subspecies[species][subspecies]]
                             if subspecies in self.subspecies[species] else 0.)
                            if species in self.subspecies
                            else int(species==subspecies and subspecies not in all_subspecies)
                            for subspecies in self.species]
                           for species in self.species])
        for subspecies, species in self.subspecies_rests.items():
            i = self.species.tolist().index(species)
            j = self.species.tolist().index(subspecies)
            result[...,j] = 0.
            result[i, j] = (
                1 - result[i].sum())
            result[i, i] = 1.
        return result

    @property
    def merge_subspecies(self):
        """Merge matrix adding up subspecies concentrations into species
        """
        result = np.array([[int(subspecies in self.subspecies[species]
                                if species in self.subspecies
                                else species==subspecies)
                            for subspecies in self.species] for species in self.species])
        for subspecies, species in self.subspecies_rests.items():
            i = self.species.tolist().index(species)
            j = self.species.tolist().index(subspecies)
            result[i, j] = 1.
        return result

    def define_subspecies(self, species: str, subspecies: Dict[str, lmfit.Parameter],
                          rest: Optional[str]='pure'):
        """Define subspecies of a given species

        Parameters
        ----------
            species: str
                The species that should be partitioned into subspecies
            subspecies: Dict[str, lmfit.Parameter]
                Fractions (between 0 and 1) of named subspecies
            rest: str
                suffix for the remainder part of the species (default pure)
        """
        if species not in self.species:
            self.species = self.species.append(pd.Index([species]))
        if (rest) not in self.species:
            self.species = self.species.append(pd.Index([rest]))
        self.species = self.species.append(pd.Index([
            suffix for suffix in subspecies
            if suffix not in self.species
        ]))
        if subspecies and species not in self.subspecies:
            self.subspecies[species] = {}
            self.subspecies_rests[rest or 'pure'] = species
        for sub, par in subspecies.items():
            self.subspecies[species][sub] = par.name
            if par.name not in self.params:
                self.params.add(par)

    def state(self, conc: Optional[Union[xr.DataArray, Dict[str, float]]] = None, /,
              **extra_conc: float) -> xr.DataArray:
        state = super().state(conc, **extra_conc)
        return xr.DataArray(state.values @ self.split_species, state.coords)

    def integrate(self, initial_condition: xr.DataArray,                   # pylint: disable=invalid-name
                  t_eval: Union[Iterable, float, None] = None,
                  **options) -> xr.DataArray:
        """Generate trajectory for given initial condition(s).

        This converts the given initial condition to subspecies concentrations
        which are then integrated using CRN.integrate. Trajectories are merged
        back into total species concentrations.
        """
        initial = self.state(initial_condition)
        initial_subspecies = xr.DataArray(initial.values @ self.split_species,
                                          initial.coords)
        traj_subspecies = super().integrate(initial_subspecies, t_eval, **options)
        return xr.DataArray(self.merge_subspecies @ traj_subspecies.values, traj_subspecies.coords,
                            name=traj_subspecies.name)


def from_string(string: str, species: Optional[List[str]] = None) -> Union[CRN, PartitionedCRN]:
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
    crn: Union[CRN, PartitionedCRN]

    crn_def = crn_parser.parse(string)

    # create CRN from definition
    if crn_def.species_defs:
        crn = PartitionedCRN(species=species)
        for species_def in crn_def.species_defs.values():
            crn.define_subspecies(*species_def)
    else:
        crn = CRN(species=species)

    for reaction in crn_def.reactions:
        if isinstance(reaction, crn_parser.Reaction):
            crn.add_reaction(*reaction)
        else:
            crn.add_reaction(reaction.educts, reaction.products, reaction.forward)
            crn.add_reaction(reaction.products, reaction.educts, reaction.backward)

    return crn
