"""Chemical reaction networks
"""
from copy import deepcopy
import math
from typing import Callable, Iterable, Sequence
from itertools import chain
import warnings
import xarray as xr                      # type: ignore
import numpy as np
import pandas as pd                      # type: ignore
from scipy.integrate import solve_ivp    # type: ignore
from scipy.optimize import basinhopping  # type: ignore
import lmfit                             # type: ignore
from . import crn_parser
from .mars import Assay

Reactants = tuple[tuple[str, int], ...] # TODO: support generic tuple[tuple[T, int], ...]

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
    reactions: mapping of complex pairs to a two-tuple of rate constant names
    params: lmfit.Parameters instance of rate constants
    """
    # TODO: support open networks and buffered species

    species: pd.Index
    complexes: list[Reactants]
    reactions: dict[tuple[Reactants, Reactants], tuple[str, str]]
    params: lmfit.Parameters

    def __init__(self,
                 reactions: list[tuple[Reactants, Reactants, lmfit.Parameter]]|None = None,
                 species: Iterable[str]|None = None):
        """Create a chemical reaction network.

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
        self.params.add('t0', value=DEFAULT_INTEGRATION_START, min=DEFAULT_MIN_T0, vary=True)
        for reaction in reactions or []:
            self.add_reaction(*reaction)

    def __str__(self) -> str:
        def render(complexes: tuple[Reactants, Reactants], forward: str, backward: str = '') -> str:
            educts = self._render_reactants(complexes[0])
            products = self._render_reactants(complexes[1])
            if backward:
                return (f"{educts} <=> {products}; "
                        + f"{forward}={self.params[forward].value}, "
                        + f"{backward}={self.params[backward].value}")
            return f"{educts} -> {products}; {forward}={self.params[forward].value}"
        return '\n'.join(render(reaction, *name) for reaction, name in self.reactions.items())

    def _repr_html_(self) -> str:
        return (
            '<table>'
            + '\n'.join(
                f'''<tr>
                    <td style="text-align: right">{self._render_reactants(reaction[0])}</td>
                    <td style="text-align: center">&rlhar</td>
                    <td style="text-align: left">{self._render_reactants(reaction[1])}</td>
                    <td style="text-align: left">{fw} = {self.params[fw].value:.2g}</td>
                    <td style="text-align: left">{bw} = {self.params[bw].value:.2g}</td>
                </tr>''' if bw else
                f'''<tr>
                    <td style="text-align: right">{self._render_reactants(reaction[0])}</td>
                    <td style="text-align: center">&LongRightArrow;</td>
                    <td style="text-align: left">{self._render_reactants(reaction[1])}</td>
                    <td style="text-align: left" colspan="2">{fw} = {self.params[fw].value:.2g}</td>
                </tr>'''
                for reaction, (fw, bw) in self.reactions.items()
            )
            + '</table>'
        )

    @property
    def burst_reactions(self) -> dict[tuple[Reactants, Reactants], tuple[str, str]]:
        """Return subset of reactions with infinite rate constant"""
        return {
            reaction: (foward_rate, backward_rate)
            for reaction, (foward_rate, backward_rate) in self.reactions.items()
            if self.params[foward_rate].value == float('inf')
            or (backward_rate and self.params[backward_rate].value == float('inf'))
        }

    @property
    def stoichiometry_matrix(self) -> np.ndarray:
        """Stoichiometry matrix of the reaction system

        This returns an n x m matrix with one row for each reaction
        (in the order they are stored in self.reactions). Each column
        denotes the stoichiometric coefficient of a species (in the
        order they are stored in self.species). Positive coefficients
        denotes products whereas negative coefficients denote educts.
        A coefficient of zero implies either that a species is not
        involved in the reaction or that it acts catalytically.
        """
        def stoichiometry(species, reactants):
            for some_species, stoich in reactants:
                if some_species == species:
                    return stoich
            return 0
        return np.array([[stoichiometry(species, products) - stoichiometry(species, educts)
                          for species in self.species]
                         for (educts, products) in self.reactions])

    @property
    def equilibrium_constants(self) -> np.ndarray:
        """Equilibrium constants of the reactions

        This returns a vector of equilibrium constants for each
        reaction. Irreversible reactions have an equilibrium constant
        equal to infinity.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.array([
                self.params[forward]/self.params[backward] if backward else float('inf')
                for (forward, backward) in self.reactions.values()
            ])

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
        n = len(self.complexes)
        rate_constants = np.zeros((n, n))
        for (educts, products), (forward, backward) in self.reactions.items():
            j = self.complexes.index(educts)
            i = self.complexes.index(products)
            kf = self.params[forward]
            kr = self.params.get(backward, 0.)
            if burst:
                rate_constants[i, j] = 1 if kf.value == float('inf') else 0.
                rate_constants[j, i] = 1 if kr and kr.value == float('inf') else 0.
            else:
                rate_constants[i, j] = val if (val := kf.value) != float('inf') else 0.
                rate_constants[j, i] = val if kr and (val := kr.value) != float('inf') else 0.
        return rate_constants

    def scale_concentration_unit(self, scale_factor: float):
        """Scale reaction rate constants to new concentration unit.

        For example, if current rate constants are given in M^-1s^-1, the
        call crn.scale_concentration_unit(1e-9) will rescale those to
        nM^-1s^-1.
        """
        for reaction, (forward, backward) in self.reactions.items():
            self.params[forward].value *= scale_factor**(len(reaction[0])-1)
            self.params[backward].value *= scale_factor**(len(reaction[1])-1)

    def add_reaction(self, educts: Reactants, products: Reactants,
                     forward_rate: lmfit.Parameter, backward_rate: lmfit.Parameter|None = None):
        """Add a reaction to the network.

        Any novel species that occur among the reactants are automatically
        added to the set of species of the network.

        Parameters
        ----------
        educts: tuple of species name, stoichiometry tuples
        products: tuple of species name, stoichiometry tuples
        forward_rate: lmfit.Parameter of the forward rate constant
        backward_rate: optional lmfit.Paramter of the backward rate constant (default: None)
        """
        # collect species and complexes
        if (products, educts) in self.reactions:
            self.reactions[products, educts] = (self.reactions[products, educts][0],
                                                forward_rate.name)

        else:
            for reactants in [educts, products]:
                self.species = self.species.append(pd.Index([
                    name for name, _ in reactants
                    if name not in self.species
                ]))

            for compl in [educts, products]:
                if compl not in self.complexes:
                    self.complexes.append(compl)

            self.reactions[educts, products] = (forward_rate.name,
                                                (backward_rate.name if backward_rate else None))

        if forward_rate.name not in self.params:
            self.params.add(forward_rate)
        if backward_rate and backward_rate.name not in self.params:
            self.params.add(backward_rate)

    def __getitem__(self, name: str) -> lmfit.Parameter:
        return self.params[name]

    def __setitem__(self, name: str, value: float|lmfit.Parameter):
        self.params[name] = value

    def state(self, conc: xr.DataArray|dict[str, float|Sequence[float]]|None = None, /,
              **extra_conc: float|Sequence[float]) -> xr.DataArray:
        """Generate a state vector with given species concentrations.

        Create a state vector with the given species concentrations.
        Species of the CRN that are ommitted in the input are set to 0.
        Concentrations can either be floating point numbers or sequences
        of floating point numbers. In case of the latter, the function
        returns a 2D DataArray wich species concentrations for the provided
        number of distinct samples.

        >>> initial = crn.state(A=100, B=100)

        Parameters
        ----------
        conc: a 1D or 2D xarray.DataArray with a coordinate "species"
              or a dictionary from species labels to float or sequence
              of floats
            giving species concentrations.
        extra_conc:
            named keyword arguemnts of additional concentrations.

        Returns
        -------
        Either a 1D DataArray (if scalar concentrations where provided
        for all species) or a 2D DataArray (if any concentration was
        provided as sequence). Unspecified species are padded with 0.
        """
        # determine resultant DataArray shape
        conc = conc if conc is not None else {}
        if isinstance(conc, dict):
            n_samples = list(set(len(value) for value in chain(conc.values(), extra_conc.values())
                                 if isinstance(value, (Sequence, np.ndarray, xr.DataArray))))
            if len(n_samples) > 1:
                raise ValueError("Inconsistent length of samples given.")
            samples = [f'Sample X{idx+1}' for idx in range(n_samples[0])] if n_samples else []
        elif conc.ndim == 1:
            n_samples = list(set(len(value) for value in extra_conc.values()
                                 if isinstance(value, (Sequence, np.ndarray, xr.DataArray))))
            if len(n_samples) > 1:
                raise ValueError("Inconsistent length of samples given.")
            samples = [f'Sample X{idx+1}' for idx in range(n_samples[0])] if n_samples else []
        else:
            samples = conc.indexes[conc.dims[0]] if conc.ndim>1 else []

        # initialize state DataArray
        if isinstance(conc, dict) and conc and samples:
            state = xr.DataArray([value if isinstance(value, (Sequence, np.ndarray))
                                        else len(samples)*[value] for value in conc.values()],
                                 {'species': list(conc.keys()), 'sample': samples}).T
        elif isinstance(conc, dict) and samples:
            state = xr.DataArray(()).expand_dims({'sample': samples, 'species': []})
        elif isinstance(conc, dict) and conc:
            state = xr.DataArray(list(conc.values()), {'species': list(conc.keys())})
        elif isinstance(conc, dict):
            state = xr.DataArray(len(self.species)*[0.], {'species': self.species})
        elif conc.ndim == 1 and samples:
            state = conc.expand_dims({'sample': samples}, 0)
        else:
            state = conc

        # reindex state to include missing species
        state = state.reindex({'species': self.species}, fill_value=0.).copy()
        # set extra_conc values
        for species, val in extra_conc.items():
            state.loc[..., species] = val
        return state

    def parameter_map(self, assay: Assay) -> pd.DataFrame:
        """Map of parameters for assay samples

        The method generates a DataFrame with specialized parameters
        for each sample. Parameters are shared among samples that
        have identical setup in the Assay.sample_map. If the sample_map
        of two samples differ, they are assigned distinct parameters.
        """
        def generate_suffixes(samples):
            length = int(math.log(len(samples), 26) + 1)
            for idx, _ in enumerate(samples):
                suff = []
                for __ in range(length):
                    suff.append(ord('a') + (idx % 26))
                    idx //= 26
                yield ''.join(chr(s) for s in reversed(suff))

        if assay.setup is None:
            raise ValueError("No setup is defined for the assay.")
        if assay.sample_map is None:
            # This path should never be taken...
            raise RuntimeError("The assay does not define a sample_map.")

        # FIXME: this does not work if samples differ only by medium or buffer
        mapping = pd.DataFrame([[None for __ in self.params]
                                for _ in assay.setup.content],
                               columns=list(self.params),
                               index=assay.setup.indexes['content'])

        # generate parameter sets for each unique species combination
        sample_sets = pd.DataFrame(assay.sample_map.drop_duplicates()).replace([None], [''])
        sample_sets.loc[:, 'params'] = None
        for sample, suffix in zip(sample_sets.index, generate_suffixes(sample_sets.index)):
            params = lmfit.Parameters()
            for name, value in self.params.items():
                name = f'{name}_{suffix}'
                params[name] = deepcopy(value)
                params[name].name = name
            sample_sets.at[sample, 'params'] = params

        # assign parameter set to all matching samples
        for _, setup in sample_sets.iterrows():
            samples = (assay.sample_map.replace([None], [''])
                       == setup[assay.sample_map.columns]).all(axis=1)
            for sample in samples[samples == True].index:
                for name, value in zip(self.params, setup['params'].values()):
                    mapping.at[sample, name] = value

        # t0 is reset to the model's t0 for all samples
        mapping['t0'] = self.params['t0']
        return mapping

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

    def equilibrate(self, initial_condition: xr.DataArray|dict, **options) -> xr.DataArray:
        """Equilibrium state of the reaction network

        Uses gradient decent to find the equilibrium state for a given
        initial condition. The implementation follows the procedure
        discussed in https://chemistry.stackexchange.com/questions/153869/

        Parameters
        ----------
        initial_condition: xarray.DataArray
            state vector to equilibrate
        """
        # pylint: disable=invalid-name
        C = self.state(initial_condition).values
        N = self.stoichiometry_matrix
        lnK = np.log(self.equilibrium_constants)

        if np.isinf(lnK).any():
            raise ValueError("Only fully reversible CRNs can be equilibrated.")

        def equilib(X):
            Y = N.T @ X + C
            if out_of_bounds := Y[Y<0].sum():
                return (1-out_of_bounds)*1e14
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = np.nansum(N*np.log(Y).T, axis=1) - lnK
            Z = np.where(np.isnan(Z), 0, Z)
            return np.linalg.norm(Z)

        minimizer_kwargs = {'method': 'Nelder-Mead', 'options': {'xatol': 1e-21, 'maxiter': 1000}}
        minimizer_kwargs.update(options)

        result = basinhopping(equilib, np.zeros(N.shape[0],), niter=100,
                              minimizer_kwargs = minimizer_kwargs)
        return xr.DataArray(N.T @ result.x + C, {'species': self.species})

    def integrate(self, initial_condition: xr.DataArray|dict,  # pylint: disable=invalid-name
                  t_eval: Iterable[float]|float|None = None,
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
        def stratify_t_eval(times):
            # pylint: disable=too-many-return-statements
            if isinstance(times, tuple):
                return pd.Index(np.linspace(*((times + (DEFAULT_INTEGRATION_POINTS,))[:3]),
                                            dtype=float),
                                name="time")
            if isinstance(times, pd.Index):
                return times
            if isinstance(times, xr.DataArray):
                if times.ndim == 0:
                    return pd.Index(np.linspace(self.params['t0'].value,
                                                float(t_eval), DEFAULT_INTEGRATION_POINTS,
                                                dtype=float),
                                    name="time")
                if times.ndim == 1:
                    return times
                raise ValueError("t_eval must have either zero or one dimension.")
            if isinstance(times, Iterable):
                return pd.Index(times, name="time")
            if times is None:
                return pd.Index(np.linspace(self.params['t0'].value,
                                            DEFAULT_INTEGRATION_END,
                                            DEFAULT_INTEGRATION_POINTS, dtype=float),
                                name="time")
            return pd.Index(np.linspace(self.params['t0'].value, t_eval,
                                        DEFAULT_INTEGRATION_POINTS, dtype=float),
                            name="time")
        times: pd.Index = stratify_t_eval(t_eval)

        initial_condition = self.state(initial_condition)
        if any(param.value==float('inf') for param in self.params.values()):
            initial_condition = self.perform_burst_reactions(initial_condition)

        kinetics = self.rate_law()

        result = xr.DataArray(np.zeros(initial_condition.shape + times.shape),
                              [(dim, initial_condition.indexes[dim])
                               for dim in initial_condition.dims]+[times])
        if len(initial_condition.dims) == 1:
            result[0:] = solve_ivp(kinetics,
                                   (self.params['t0'], times[-1]),
                                   initial_condition,
                                   t_eval=times, vectorized=True, **options).y
        else:
            for idx, initial in enumerate(initial_condition):
                # TODO: parallelize using multiprocessing.Pool's
                result[idx, 0:] = solve_ivp(kinetics,
                                            (self.params['t0'], times[-1]),
                                            initial,
                                            t_eval=times, vectorized=True, **options).y
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
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: float|xr.DataArray = 1.,
            vary_t0: bool|None = None,
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
        options:
            Any remaining keyword arguments are pass to lmfit.minimize

        Result
        ------
            An lmfit MinimizerResult that contains (among others) the
            attribute params, which are the optimized parameters.
        """
        if vary_t0 is not None:
            warnings.warn("Argument vary_t0 to CRN.fit is no longer supported. "
                          "Set CRN.params['t0'].vary instead")
        conversion = conversion or (lambda conc: conc.sel(species=data.species))
        initial = initial[initial.sample.isin(data.sample)]
        original = deepcopy(self.params)
        params = self.params
        params['t0'].max = float(data.time[0]) # FIXME: respect injections
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

    subspecies: dict[str, dict[str, str]]
    subspecies_rests: dict[str, str]


    def __init__(self,
                 reactions: list[tuple[Reactants, Reactants, lmfit.Parameter]]|None = None,
                 species_defs : list[tuple[str, dict[str, lmfit.Parameter], str]]|None = None,
                 species: Iterable[str]|None = None):
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

    def define_subspecies(self, species: str, subspecies: dict[str, lmfit.Parameter],
                          rest: str='pure'):
        """Define subspecies of a given species

        Parameters
        ----------
            species: str
                The species that should be partitioned into subspecies
            subspecies: dict[str, lmfit.Parameter]
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

    def state(self, conc: xr.DataArray|dict[str, float|Sequence[float]]|None = None, /,
              **extra_conc: float|Sequence[float]) -> xr.DataArray:
        state = super().state(conc, **extra_conc)
        return xr.DataArray(state.values @ self.split_species, state.coords)

    def equilibrate(self, initial_condition: xr.DataArray|dict, **options) -> xr.DataArray:
        initial = self.state(initial_condition)
        initial_subspecies = xr.DataArray(initial.values @ self.split_species,
                                          initial.coords)
        eq_subspecies = super().equilibrate(initial_subspecies, **options)
        return xr.DataArray(self.merge_subspecies @ eq_subspecies.values, eq_subspecies.coords,
                            name=eq_subspecies.name)

    def integrate(self, initial_condition: xr.DataArray|dict,  # pylint: disable=invalid-name
                  t_eval: Iterable|float|None = None,
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


def from_string(string: str, species: list[str]|None = None) -> CRN|PartitionedCRN:
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
    crn: CRN|PartitionedCRN

    crn_def = crn_parser.parse(string)

    # create CRN from definition
    if crn_def.species_defs:
        crn = PartitionedCRN(species=species)
        for species_def in crn_def.species_defs.values():
            crn.define_subspecies(*species_def)
    else:
        crn = CRN(species=species)

    for reaction in crn_def.reactions:
        crn.add_reaction(reaction.educts, reaction.products, reaction.forward, reaction.backward)

    return crn
