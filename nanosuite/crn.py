"""Chemical reaction networks
"""
from __future__ import annotations
from concurrent import futures
from copy import deepcopy
from itertools import chain
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings
import lmfit                             # type: ignore
import numpy as np
import pandas as pd                      # type: ignore
from scipy.integrate import solve_ivp    # type: ignore
from scipy.optimize import basinhopping  # type: ignore
import xarray as xr                      # type: ignore
from . import crn_parser
from .utils import Cache


Reactants = tuple[tuple[str, int], ...] # TODO: support generic tuple[tuple[T, int], ...]

DEFAULT_INTEGRATION_START = 0
DEFAULT_INTEGRATION_END = 100
DEFAULT_INTEGRATION_POINTS = 501
DEFAULT_MIN_T0 = -np.inf


def do_integration(crn: CRN, initial: xr.DataArray, params: lmfit.Parameters, times: pd.Index,
                   options: dict) -> np.ndarray:
    """Integrate crn for initial condition at given times

    Internally, this method uses the method of van der Schaft et al.
    (2011) SIAM J Appl Math 73(2):953-973.
    """
    # pylint: disable=invalid-name

    Z = crn.complex_graph
    A = crn.get_complex_adjacency(params)
    L = np.diag(np.sum(A, axis=0)) - A

    def kinetics(_, state):
        # Z.T @ log(state) with convention 0*inf = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = np.log(state, out=-np.inf*np.ones_like(state), where=state != 0)
            tmp = np.nansum(Z.T*tmp, axis=1)
        return -Z @ L @ np.exp(tmp)

    if 'atol' not in options:
        options['atol'] = 1e-8*initial.max() or 1e-8
    if 'rtol' not in options:
        options['rtol'] = 1e-8

    result = solve_ivp(kinetics, (params['t0'], times[-1]), initial, t_eval=times, **options)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y

def dist_to_equilib(X, N, C, lnK):
    """Compute distance to equilibrium distribution

    This is used by CRN.equilibrate to minimize the
    distance to the equilibrium in an iterative
    gradient decent.
    """
    # pylint: disable=invalid-name
    Y = N.T @ X + C.data
    if out_of_bounds := Y[Y<0].sum():
        return (1-out_of_bounds)*1e14
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.nansum(N*np.log(Y).T, axis=1) - lnK
    Z = np.where(np.isnan(Z), 0, Z)
    return np.linalg.norm(Z)


class ParameterMap(lmfit.Parameters):
    """Mapping between samples and parameters

    A ParameterMap establishes relationships between samples and
    Parameter objects. ParameterMap's are usually obtained by
    calling CRN.parametrize_for. There should normally be no need
    for the user to manually create a ParameterMap.

    ParameterMap.mapping is a dataframe that maps from general
    parameter names to sample-specific parameter names.
    ParameterMap.general_params holds the original parameters.
    """
    mapping: pd.DataFrame
    general_params: dict[str, lmfit.Parameter]
    zero: lmfit.Parameter = lmfit.Parameter('__ZERO__', 0, vary=False)

    class ParameterProxy:
        """A proxy object to access all specialized parameters

        ParameterProxy's are returned by ParameterMap.__getitem__ when accessing
        a general parameter that is overloaded with specializations. The proxy
        allows to set attributes of all specializations of the general parameter
        either from a sequence of values or from a single value.
        """
        # pylint: disable=too-few-public-methods
        params: ParameterMap
        specializations: pd.Series

        def __init__(self, parameter_map: ParameterMap, specializations: pd.Series):
            self.__dict__.update(specializations=specializations, params=parameter_map)

        def __setattr__(self, attr: str, val: Any) -> None:
            if isinstance(val, xr.DataArray):
                val = val.data
            if isinstance(val, Sequence|np.ndarray):
                if len(val) != len(self.specializations):
                    raise ValueError(f"Must provide {len(self.specializations)} values")
                for special, value in zip(self.specializations, val):
                    if self.params[special] is self.params.zero:
                        continue  # don't alter ParameterMap.zero
                    setattr(self.params[special], attr, value)
            else:
                for special in self.specializations:
                    if self.params[special] is self.params.zero:
                        continue  # don't alter ParameterMap.zero
                    setattr(self.params[special], attr, val)

        def __getattr__(self, attr: str) -> np.ndarray:  # FIXME: return DataFrame or DataArray
            return np.array([getattr(self.params[special], attr)
                            for special in self.specializations])

    def __init__(self, sample_map: pd.DataFrame|None = None, usersyms: Mapping|None = None):
        super().__init__(usersyms)

        index = sample_map.index if sample_map is not None else pd.Index([])
        self.mapping = pd.DataFrame([], index=index)
        self.general_params = {}

    def __reduce__(self) -> tuple:
        return self.__class__, (), {'mapping': self.mapping,
                                    'general': self.general_params,
                                    'params': super().__reduce__()}

    def __setstate__(self, state):
        # I first need to set self.mapping, so that specialized parameters
        # are not added as mapping columns when setting lmfit.Parameters
        self.mapping = state['mapping']
        self.general_params = state['general']
        return super().__setstate__(state['params'][2])

    def __getitem__(self, key: str) -> lmfit.Parameter:
        if key in self:
            return super().__getitem__(key)
        if key in self.general_params:
            return self.ParameterProxy(self, self.mapping[key])
        if not key:
            return self.zero
        raise KeyError(key)

    def update(self, other: ParameterMap):
        if (self.mapping.shape != other.mapping.shape
            or (self.mapping.values != other.mapping.values).any()):
            raise ValueError("Can only update from parameter maps with equal sample map.")
        super().update(other)
        self.general_params.update(other.general_params)

    def __deepcopy__(self, memo: Any) -> ParameterMap:
        result = super().__deepcopy__(memo)
        result.mapping = deepcopy(self.mapping)
        result.general_params = deepcopy(self.general_params)
        return result

    def __setitem__(self, name: str, value: lmfit.Parameter|None) -> None:
        super().__setitem__(name, value)
        if isinstance(name, lmfit.Parameter):
            name = name.name
        if name not in self.mapping.values:
            # only add a column to the mapping if name is a general parameter
            self.mapping[name] = name
            self.mapping = self.mapping.copy()

    def __add__(self, other: ParameterMap) -> ParameterMap:
        raise RuntimeError("Not implemented yet.")

    def set(self, **kwds: int|float|lmfit.Parameter) -> None:
        raise RuntimeError("Not implemented yet.")

    def specify(self, samples: pd.Index, name: str, new_name: str) -> None:
        """Overload general parameter for specific samples"""
        if name not in self.mapping.columns:
            raise KeyError(f"Unknown parameter '{name}'")
        if all(self.mapping[name] == name):
            self.mapping[name] = ""
            self.general_params[name] = self.pop(name)
        if name not in self.mapping.columns:
            self.mapping[name] = 0  # Can I ever get into this code path?

        param = deepcopy(self.general_params[name])
        param.name = new_name
        self[new_name] = param
        self.mapping.drop(columns=[new_name], inplace=True)
        for idx in samples:
            self.mapping.at[idx, name] = new_name

    def specification_for(self, sample: xr.DataArray) -> dict[str, lmfit.Parameter]:
        """Return specific parameters for a given sample

        Parameters
        ----------
        samples: xarray.DataArray
            sample for the requested parameter specialization

        Returns
        -------
            A dictionary that maps general parameter names
            to the specific parameters defined for the given sample
        """
        if len(self.mapping) == 0:
            return self
        content_dim = next(iter(sample.coords))
        specification = self.mapping.loc[sample.coords[content_dim].values]
        return {general: self.get(specification[general], self.zero)
                for general in self.mapping.columns}

    def fix_outside(self, samples: xr.DataArray) -> None:
        """Fix all parameters that do not occur in the given samples

        Parameters
        ----------
        samples: xarray.DataArray
            samples to vary parameters for
        """
        content_dim = next(iter(samples.coords))
        subset = self.mapping.loc[samples.coords[content_dim].data].values
        for name in self:
            if name not in subset:
                self[name].vary = False


class CRN:
    """Chemical reaction network

    CRN represents a chemical reaction network, i.e. reactions among
    multisets of species. Kinetics are assumed to follow mass action
    kinetics.
    There are several ways to create a new reaction network. As a
    simple example, the reaction

        A + B <=> C

    with forward rate constant k_forward and backward rate constant
    k_backward can be created with:

    >>> from lmfit import Parameter
    >>> crn = CRN({
    ...     ((("A", 1), ("B", 1)), (("C", 1),)):
    ...         (Parameter('k_forward', 0.1), Parameter('k_backward', 0.1)),
    ... })

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
    params: ParameterMap
    """
    # TODO: support open networks and buffered species

    species: pd.Index
    complexes: list[Reactants]
    reactions: dict[tuple[Reactants, Reactants], tuple[str, str]]
    params: ParameterMap

    def __init__(self,
                 reactions: dict[tuple[Reactants, Reactants],
                                 tuple[lmfit.Parameter,
                                 lmfit.Parameter|None]]|None = None,
                 species: Iterable[str]|None = None):
        """Create a chemical reaction network.

        Parameters
        ----------
        reactions: list of tuples of educts, products and a rate constant
            See class documentation for details.
        species: list of strings
            If species names are provided, they determine the order of
            the species in the state vector used in crn.integrate and
            crn.rate_law. If not provided, they are ordered as they occur
            in the CRN definition.
        """
        self.species = pd.Index(species or [])
        self.complexes = []
        self.reactions = {}
        self.params = ParameterMap()
        self.params.add('t0', value=DEFAULT_INTEGRATION_START, min=DEFAULT_MIN_T0, vary=True)
        for reaction, rates in (reactions or {}).items():
            self.add_reaction(*reaction, *rates)

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
        """Return subset of reactions with infinite rate constant

        DEPRECATED: This property has been deprecated in version 0.3.0
        """
        warnings.warn("CRN.burst_reactions is deprecated and will be removed.",
                      DeprecationWarning, stacklevel=2)
        return {
            reaction: (forward_rate, backward_rate)
            for reaction, (forward_rate, backward_rate) in self.reactions.items()
            if forward_rate in self.params and self.params[forward_rate].value == float('inf')
            or (backward_rate and backward_rate in self.params
                and self.params[backward_rate].value == float('inf'))
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

    def get_complex_adjacency(self, params: dict[str, lmfit.Parameter],
                              burst: bool=False) -> np.ndarray:
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
            kf = float(params[forward])
            kr = float(params.get(backward, 0.))
            if burst:
                rate_constants[i, j] = 1 if kf == float('inf') else 0.
                rate_constants[j, i] = 1 if kr == float('inf') else 0.
            else:
                rate_constants[i, j] = kf if kf != float('inf') else 0.
                rate_constants[j, i] = kr if kr != float('inf') else 0.
        return rate_constants

    def get_equilibrium_constants(self,
                                  params: dict[str, lmfit.Parameter]|None = None) -> np.ndarray:
        """Equilibrium constants of the reactions
    
        This returns a vector of equilibrium constants for each
        reaction. Irreversible reactions have an equilibrium constant
        equal to infinity.

        Parameters
        ----------
        params: optional Parameter dictionary
            Reaction rate constants used to calculate equilibrium constants.
            Uses self.params if not given.
        """
        params = params if params is not None else self.params
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.array([
                params[forward]/b if backward and (b:=params[backward]) != 0 else float('inf')
                for (forward, backward) in self.reactions.values()
            ])

    def scale_concentration_unit(self, scale_factor: float) -> ParameterMap:
        """Scale reaction rate constants to new concentration unit.

        Returns
        -------
        ParameterMap with concentrations scaled by the given factor.
        For example, if current rate constants are given in M^-1s^-1, the
        call crn.scale_concentration_unit(1e-9) will rescale those to
        nM^-1s^-1.
        """
        parameter_map = self.params.copy()

        for (f_reaction, b_reaction), (f_rate, b_rate) in self.reactions.items():
            for reaction, param in [(f_reaction, f_rate), (b_reaction, b_rate)]:
                if not param:
                    continue
                factor = scale_factor**(len(reaction)-1)
                if not parameter_map.mapping.empty and param in parameter_map.mapping:
                    for specific in self.params.mapping[param]:
                        parameter_map[specific].value *= factor
                else:
                    parameter_map[param].value *= factor
        return parameter_map

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
        if forward_rate.value == float('inf') and backward_rate and backward_rate.value != 0:
            raise ValueError("Reactions with infinite rate constant cannot be reversible.")
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
        if backward_rate is not None and backward_rate.name not in self.params:
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

        >>> model = from_string("A + B <=> C; k1, k2")
        >>> initial = model.state(A=100, B=100)

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

    def parametrize_for(self, sample_map: pd.DataFrame,
                        **parameter_dependencies: list[str]) -> ParameterMap:
        """Return a parametrization for the given sample_map

        Keyword arguments are parameter names, which are to be specialized
    	following the provided sample_map dependent on the species given
        as argument value. For example:

        >>> model = from_string("A + B <=> C; k1, k2")
        >>> params = model.parametrize_for(assay.sample_map,
        ...                                k1=['A', 'B'], k2=['A'])  # doctest: +SKIP

        Calling the method sets new model.params for the given sample_map and
        parameter dependencies.

        # TODO: The parameter_map currently does not respect buffer and media
        or any descriptor other than chemical species.
        """
        # We define a mapping of sample indices to parameters
        # the default behaviour is for all samples to reference the general parameter
        parameter_map = ParameterMap(sample_map)
        parameter_map.add_many(*self.params.values())

        # For each rate with declared species_class dependencies, the subset of
        # dependent species is determined from the sample_map and partitioned into
        # unique sample_sets

        # For each of these sample_sets, we set the named rate parameter of the mapping
        # to a new specialized parameter object. The parameter name is suffixed with the
        # names of the dependencies (if k1 depends on Probe, the local parameter_map will
        # be named k1_Probe_1, k1_Probe_2, etc.

        # If a parameter gets specialized, it might require to also specialize
        # parameters that depend on it via expressions. E.g. if k_back is constrained
        # to equal k_forward / exp(-dG) and dG becomes specialized, then
        # k_back needs to become specialized too -- and the equation for each
        # k_back needs to be rewritten to use the specialized dG parameter.
        # This could even happen iteratively, if a third parameter expression
        # depends on k_back...
        # We first add all these implicit dependencies to the parameter_dependencies

        while True:
            indirect = {par.name: list(set.union(*[set(parameter_dependencies[n]) for n in dep]))
                        for par in self.params.values()
                        if par.name not in parameter_dependencies
                        and (dep := set(par._expr_deps) & set(parameter_dependencies))}  # pylint: disable=protected-access
            if not indirect:
                break
            parameter_dependencies.update(indirect)

        for name, dependencies in parameter_dependencies.items():
            dependencies = sorted(dependencies)
            sample_sets = sample_map[dependencies].drop_duplicates().replace([None], [''])

            for _, species in sample_sets.iterrows():
                if not any(species):
                    continue
                suffix = '_'.join(dep.replace(' ', '_') for dep in species)
                samples = (sample_map[dependencies]
                                     [sample_map[dependencies] == species].dropna()
                                                                          .index)
                parameter_map.specify(samples, name, f'{name}_{suffix}')

        # Now we go back and rewrite parameter expressions to use specialized parameters
        expr_deps = [(par.name, list(dep)) for par in parameter_map.general_params.values()
                     if (dep := set(par._expr_deps) & set(parameter_dependencies))]  # pylint: disable=protected-access
        for name, deps in expr_deps:
            expr = [parameter_map.general_params[name].expr
                    for sample in parameter_map.mapping.index]
            substitutions = parameter_map.mapping[deps]
            for general in substitutions:
                expr = [re.sub(f'\\b{general}\\b', special, ex)
                        for ex, special in zip(expr, substitutions[general])]
            parameter_map[name].expr = expr

        return parameter_map

    def equilibrate(self, initial_condition: xr.DataArray|dict, **options) -> xr.DataArray:
        """Equilibrium state of the reaction network

        Uses gradient decent to find the equilibrium state for a given
        initial condition. The implementation follows the procedure
        discussed in https://chemistry.stackexchange.com/questions/153869/

        Parameters
        ----------
        initial_condition: 1D or 2D xarray.DataArray
            state vector to equilibrate

        Returns
        -------
        A 1D or 2D xarray.DataArray with the equilibrium corresponding to
        the initial state
        """
        # pylint: disable=invalid-name
        initial_condition = self.state(initial_condition)
        N = self.stoichiometry_matrix

        kwargs: dict[str, Any] = {'method': 'Nelder-Mead',
                                  'options': {'xatol': 1e-21, 'maxiter': 100}}
        kwargs.update(options)

        if initial_condition.ndim == 1:
            params = self.params.specification_for(initial_condition)
            lnK = np.log(self.get_equilibrium_constants(params))
            kwargs['args'] = (N, initial_condition, lnK)
            result = basinhopping(dist_to_equilib, np.zeros(N.shape[0]), niter=100,
                                  minimizer_kwargs = kwargs)
            return xr.DataArray(N.T @ result.x + initial_condition, initial_condition.coords)

        if initial_condition.ndim == 2:
            with futures.ProcessPoolExecutor() as executor:
                def schedule_computation(sample):
                    params = self.params.specification_for(sample)
                    lnK = np.log(self.get_equilibrium_constants(params))
                    return executor.submit(basinhopping, dist_to_equilib, np.zeros(N.shape[0]),
                                           niter=100,
                                           minimizer_kwargs = kwargs | {'args': (N, sample, lnK)})

                jobs = [schedule_computation(sample) for sample in initial_condition]
                equilibrium = [N.T @ job.result().x + C
                               for job, C in zip(jobs, initial_condition)]

            return xr.DataArray(equilibrium, initial_condition.coords)

        raise ValueError("Initial condition mut be 1 or 2 dimensional")

    def integrate(self, initial_condition: xr.DataArray|dict,
                  t_eval: Iterable[float]|float|None = None,
                  cache: Cache|None = None, **options) -> xr.DataArray:
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

        if not cache:
            cache = Cache()

        times: pd.Index = stratify_t_eval(t_eval)
        initial_condition = self.state(initial_condition)
        if any(param.value==float('inf') for param in self.params.values()):
            initial_condition = self.perform_burst_reactions(initial_condition)

        if len(initial_condition.dims) == 1:
            traj: Iterable = do_integration(self, initial_condition, self.params, times, options)
        else:
            with futures.ProcessPoolExecutor() as executor:
                @cache.compute
                def schedule_computation(sample):
                    params = self.params.specification_for(sample)
                    return executor.submit(do_integration, self, sample, params, times, options)

                jobs = [schedule_computation(sample) for sample in initial_condition]
                traj = [job.result() for job in jobs]

        return xr.DataArray(traj, [(dim, initial_condition.indexes[dim])
                                   for dim in initial_condition.dims] + [times],
                            name="concentration")

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

        iterations = 10*len(self.reactions)
        for _ in range(iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                if len(state.dims) == 1:
                    A = self.get_complex_adjacency(self.params, True)
                    L = np.diag(np.sum(A, axis=0)) - A
                    rates = Z @ L @ np.exp(np.nansum(Z.T*np.log(state.values), axis=1))
                    fraction = min(x/y for x, y in zip(state, rates)
                                   if y > 0).values if rates.any() else 0
                else:
                    params = self.params.specification_for(state)
                    A = self.get_complex_adjacency(params, True)
                    L = np.diag(np.sum(A, axis=0)) - A
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
            **options) -> lmfit.minimizer.MinimizerResult:
        """Fit model parameters to experimental data

        Parameters
        ----------
        data: xarray.DataArray with rfu over time
        initial: xarray.DataArray with concentrations of species
        conversion: optional function that converts concentrations to RFU values
            The conversion must accept DataArrays of concentrations
            over time and must return a DataArray of RFU values over
            time. Can be obtained from mars.Assay.convert.
        error: optional xr.DataArray with rfu over time or float (default 1.)
            Standard deviations of measured data
        options:
            Any remaining keyword arguments are pass to lmfit.minimize

        Result
        ------
            An lmfit MinimizerResult that contains (among others) the
            attribute params, which are the optimized parameters.
        """
        conversion = conversion or (lambda conc: conc.sel(species=data.species))
        initial = initial[initial.sample.isin(data.sample)]

        cache = Cache(2*len(initial))

        def objective(params):
            self.params = params
            model = conversion(self.integrate(initial, t_eval=data.time, cache=cache))
            return (data-model)/error

        original = self.params
        params = self.params.copy()
        params.fix_outside(data)
        params['t0'].max = float(data.time[0]) # TODO: respect injections
        fit = lmfit.minimize(objective, params, **options)
        self.params = original
        return fit

    @staticmethod
    def _render_reactants(multiset) -> str:
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

    >>> from lmfit import Parameter
    >>> crn = PartitionedCRN({
    ...     ((("biomarker_mutant", 1), ("probe", 1)), (("biomarker_mutant", 1), ('signal', 1))):
    ...         (Parameter('k', 0.1), None),
    ... })

    >>> crn.define_subspecies("biomarker",
    ...     subspecies={"mutant": Parameter('k_mutant', 0.01)}, rest="biomarker_wildtype")

    >>> initial = crn.state(biomarker=100)
    >>> trajectory = crn.integrate(initial)
    >>> total = trajectory.sel(species="biomarker")
    >>> mutant = trajectory.sel(species="biomarker_mutant")
    >>> wildtype = trajectory.sel(species="biomarker_wildtype")


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
                 reactions: dict[tuple[Reactants, Reactants],
                                 tuple[lmfit.Parameter,
                                 lmfit.Parameter|None]]|None = None,
                 species_defs : list[tuple[str, dict[str, lmfit.Parameter], str]]|None = None,
                 species: Iterable[str]|None = None):
        super().__init__(reactions, species)
        self.subspecies = {}
        self.subspecies_rests = {}

        for species_def in species_defs or []:
            self.define_subspecies(*species_def)

    def split_species(self, index: pd.Index|None = None) -> np.ndarray:
        """Split matrix distributing species into subspecies concentrations"""
        all_subspecies = set(chain(*self.subspecies.values()))

        def partition(species, subspecies) -> float:
            if species in self.subspecies:
                if subspecies in self.subspecies[species]:
                    name = self.subspecies[species][subspecies]
                    if index is not None and len(self.params.mapping):
                        name = str(self.params.mapping[name].loc[index])
                    return self.params[name]
                return 0.
            return int(species==subspecies and subspecies not in all_subspecies)

        result = np.array([[partition(species, subspecies)
                            for subspecies in self.species]
                           for species in self.species])
        for subspecies, species in self.subspecies_rests.items():
            i = self.species.tolist().index(species)
            j = self.species.tolist().index(subspecies)
            result[...,j] = 0.
            result[i, j] = 1 - result[i].sum()
            result[i, i] = 1.
        return result

    @property
    def merge_subspecies(self) -> np.ndarray:
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

        if len(state.dims) == 1:
            return xr.DataArray(state.values @ self.split_species(), state.coords)
        content_dim = next(iter(state.coords))
        return xr.DataArray([sample.values @ self.split_species(sample.coords[content_dim].data)
                                 for sample in state],
                                state.coords)

    def equilibrate(self, initial_condition: xr.DataArray|dict, **options) -> xr.DataArray:
        initial = self.state(initial_condition)
        initial_subspecies = xr.DataArray(initial.values @ self.split_species(),
                                          initial.coords)
        eq_subspecies = super().equilibrate(initial_subspecies, **options)
        return xr.DataArray(self.merge_subspecies @ eq_subspecies.values, eq_subspecies.coords,
                            name=eq_subspecies.name)

    def integrate(self, initial_condition: xr.DataArray|dict,  # pylint: disable=invalid-name
                  t_eval: Iterable|float|None = None,
                  cache: Cache|None = None, **options) -> xr.DataArray:
        """Generate trajectory for given initial condition(s).

        This converts the given initial condition to subspecies concentrations
        which are then integrated using CRN.integrate. Trajectories are merged
        back into total species concentrations.
        """
        initial = self.state(initial_condition)
        traj_subspecies = super().integrate(initial, t_eval, cache, **options)
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
