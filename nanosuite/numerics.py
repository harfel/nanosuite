"""Numerical algorithms for chemical reaction networks

These methods work in conjunction with crn.CRN and crn.PartitionedCRN.
"""
from __future__ import annotations
from concurrent import futures
from threading import Lock
from typing import Iterable, TYPE_CHECKING
import warnings
import lmfit                             # type: ignore
import numpy as np
import pandas as pd                      # type: ignore
from scipy.integrate import solve_ivp    # type: ignore
from scipy import optimize               # type: ignore
import xarray as xr                      # type: ignore
from .utils import Cache

if TYPE_CHECKING:
    from typing import Any, Callable
    from .crn import CRN, ParameterMap


class DummyExecutor(futures.Executor):
    """Executor that runs jobs sequentially in the main thread

    Meant to simplify debugging. Do not use in production.
    """
    def __init__(self, warn: bool = True):
        if warn:
            warnings.warn("Using DummyExecutor. Do not use in production.")
        self._shutdown = False
        self._shutdown_lock = Lock()

    def submit(self, fn, /, *args, **kwargs):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown")

            f = futures.Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:  # pylint: disable=broad-exception-caught
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._shutdown_lock:
            self._shutdown = True


class Trajectory:
    """Trajectory of a CRN

    Parameters
    ----------
    """
    def __init__(self, crn: CRN):
        # FIXME: elevate eval and fit options to constructor?
        self.crn = crn

    def eval(self,
             initial: xr.DataArray|dict,
             t_eval: Iterable[float]|float|None = None,
             cache: Cache|None = None, **options) -> xr.DataArray:
        """Evaluate trajectory at given time points

    Internally, the method uses scipy.integrate.solve_ivp.
    Optional keyword arguments (method, atol, rtol, etc.) are
    passed to solve_ivp.

    Parameters
    ----------
    initial_condition: 1D or 2D xarray.DataArray
        the last coord must denote species concentrations

        If initial_condition is a 1D vector, Trajectory.eval returns
        a 2D DataArray of states. If initial_condition is a 2D DataArray,
        the return value is a 3D DataArray with trajectories for
        each initial condition.

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
        2D or 3D DataArray of trajectories. See help(Trajectory).
        """
        def stratify_t_eval(times):
            # pylint: disable=too-many-return-statements
            from .crn import (DEFAULT_INTEGRATION_END,  # pylint: disable=import-outside-toplevel
                              DEFAULT_INTEGRATION_POINTS)

            if isinstance(times, tuple):
                times = tuple(float(t) for t in times[:2]) + tuple(int(t) for t in times[2:])
                return pd.Index(np.linspace(*((times + (DEFAULT_INTEGRATION_POINTS,))[:3]),
                                            dtype=float),
                                name="time")
            if isinstance(times, pd.Index):
                return times
            if isinstance(times, xr.DataArray):
                if times.ndim == 0:
                    return pd.Index(np.linspace(self.crn.params['t0'].value,
                                                float(t_eval), DEFAULT_INTEGRATION_POINTS,
                                                dtype=float),
                                    name="time")
                if times.ndim == 1:
                    return times
                raise ValueError("t_eval must have either zero or one dimension.")
            if isinstance(times, Iterable):
                return pd.Index(times, name="time")
            if times is None:
                return pd.Index(np.linspace(self.crn.params['t0'].value,
                                            DEFAULT_INTEGRATION_END,
                                            DEFAULT_INTEGRATION_POINTS, dtype=float),
                                name="time")
            return pd.Index(np.linspace(self.crn.params['t0'].value, t_eval,
                                        DEFAULT_INTEGRATION_POINTS, dtype=float),
                            name="time")

        initial = self.crn.state(initial)
        if initial.ndim not in [1, 2]:
            raise ValueError("Initial condition must be 1D or 2D")
        if any(param.value==float('inf') for param in self.crn.params.values()):
            initial = self.crn.perform_burst_reactions(initial)

        if not cache:
            cache = Cache()

        times: pd.Index = stratify_t_eval(t_eval)
        traj: Iterable[xr.DataArray]

        if len(initial.dims) == 1:
            params = self.crn.params.specification_for(initial)
            traj = self._do_integration(initial, params, times, options)
        else:
            with futures.ProcessPoolExecutor() as executor:
                def integrate(sample):
                    params = self.crn.params.specification_for(sample)
                    return schedule_computation(sample.data,
                                                {k: v.value for k, v in params.items()},
                                                times, options)

                @cache.compute
                def schedule_computation(init, pardict, times, opts):
                    return executor.submit(self._do_integration, init, pardict, times, opts)

                jobs = [integrate(sample) for sample in initial]
                traj = [job.result() for job in jobs]

        traj = xr.DataArray(traj,
                            [(dim, initial.indexes[dim]) for dim in initial.dims]
                            + [times],
                            name='concentration')
        return self.crn.post_process_state(traj)

    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray|dict,
            observe: str|Callable[[xr.DataArray], xr.DataArray]|None = None,
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: xr.DataArray|None = None,
            **options) -> lmfit.minimizer.MinimizerResult:
        """Fit model parameters to experimental data

        Parameters
        ----------
        data: xarray.DataArray with rfu over time

        observe: string or function
            If observe is a string, it specifies a species name of the model
            that is mapped against the provided data. If it is a function,
            it must map model states to the data domain

        conversion: optional function to convert concentrations to RFU values

            *Deprecated since 0.3.1: use observe instead*

            The conversion must accept DataArrays of concentrations
            over time and must return a DataArray of RFU values over
            time. Can be obtained from mars.Assay.convert.

        error: optional xr.DataArray with rfu over time or float
            Standard deviations of measured data. If provided, the fit minimizes
            the WSSR = (model-data)**2/error**2. Otherwise, it minimizes the
            relative deviation (data/model - 1)**2

        options:
            Any remaining keyword arguments are pass to lmfit.minimize

        Result
        ------
            An lmfit MinimizerResult that contains (among others) the
            attribute params, which are the optimized parameters.
        """
        if conversion:
            warnings.warn("Argument conversion is deprecated. Use observe instead.",
                          DeprecationWarning, stacklevel=2)
            if not observe:
                observe = conversion
        convert = (lambda conc: conc.sel(species=observe)) if isinstance(observe, str) else observe

        options = {'xtol': 1e-7} | options

        if data.ndim == 2:
            content_dim = data.dims[0]
            initial = initial[initial.coords[content_dim].isin(data.coords[content_dim])]

        initial = self.crn.state(initial)

        cache = Cache(2*len(initial))

        def wssr(model):
            return (model-data)/error

        def reldev(model):
            return data/model - 1

        residual = wssr if error is not None else reldev

        def objective(params):
            self.crn.params = params
            model = convert(self.eval(initial, t_eval=data.time, cache=cache))
            return residual(model)

        original = self.crn.params
        params = original.copy()
        params.fix_outside(data)
        for p in params.values():
            if p.value == np.inf:
                p.vary = False
        params['t0'].max = float(data.time[0]) # TODO: respect injections
        fit = lmfit.minimize(objective, params, **options)
        self.crn.params = original
        return fit

    def _do_integration(self, initial: xr.DataArray,
                        params: lmfit.Parameters, times: pd.Index, options: dict) -> np.ndarray:
        """Integrate crn for initial condition at given times

        Internally, this method uses the method of van der Schaft et al.
        (2011) SIAM J Appl Math 73(2):953-973.
        """
        # pylint: disable=invalid-name
        options = {'atol': float(1e-6*initial.max()) or 1e-10, 'rtol': 1e-6} | options

        Z = self.crn.complex_graph
        A = self.crn.get_complex_adjacency(params)
        L = np.diag(np.sum(A, axis=0)) - A

        def kinetics(_, state):
            # Z.T @ log(state) with convention 0*inf = 0
            with np.errstate(divide='ignore', invalid='ignore'):
                tmp = np.log(state, out=-np.inf*np.ones_like(state), where=state != 0)
                tmp = np.nansum(Z.T*tmp, axis=1)
            return -Z @ L @ np.exp(tmp)

        result = solve_ivp(kinetics, (params['t0'], times[-1]), initial,
                           t_eval=times, **options)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y


class Equilibrium:
    """Equilibrium of a CRN

    This represents the equilibrium distribution of a reversible CRN.
    The main purpose of the model is to fit reaction rate constants
    to experimental data. When doing so, care must be taken to fix
    one of the kinetic rate parameters that constitute the equilibrium
    constant. E.g.

    >>> model = from_string("A + B <=> C; k_f=1, k_r")
    >>> model.params['k_f'].vary = False
    >>> eq = Equilibrium(model, model.state(A=100, B=100))
    """
    def __init__(self, crn: CRN):
        self.crn = crn
        self._simplex = None

    def eval(self, initial: xr.DataArray|dict, basinhopping: int = 100, **options) -> xr.DataArray:
        """Compute equilibrium state

        Uses gradient decent to find the equilibrium state for a given
        initial condition. The implementation follows the procedure
        discussed in https://chemistry.stackexchange.com/questions/153869/

        Parameters
        ----------
        Any keyword argument is passed through to lmfit.minimize

        Returns
        -------
        A 1D or 2D xarray.DataArray with the equilibrium corresponding to
        the initial state
        """
        # pylint: disable=invalid-name
        initial = self.crn.state(initial)
        if initial.ndim not in [1, 2]:
            raise ValueError("Initial condition must be 1D or 2D")
        if any(param.value==float('inf') for param in self.crn.params.values()):
            initial = self.crn.perform_burst_reactions(initial)

        N = self.crn.stoichiometry_matrix

        kwargs: dict[str, Any] = {'method': 'Nelder-Mead',
                                  'options': {'xatol': 1e-12, 'maxiter': 500}}
        # TODO: would it be good to start with the final_simplex of the last evaluation?
        # if self._simplex is not None:
        #     print(self._simplex.T)
        #     kwargs['options']['initial_simplex'] = self._simplex
        kwargs.update(options)

        if initial.ndim == 1:
            params = self.crn.params.specification_for(initial)
            lnK = np.log(self.crn.get_equilibrium_constants(params))
            kwargs['args'] = (N, initial, lnK)
            result = optimize.basinhopping(self._dist_to_equilib, np.zeros(N.shape[0]),
                                           niter=basinhopping, minimizer_kwargs = kwargs)
            # self._simplex = result['lowest_optimization_result']['final_simplex'][0]
            equilibrium = N.T @ result.x + initial

        else:
            # with DummyExecutor(warn=False) as executor:
            with futures.ProcessPoolExecutor() as executor:
                def schedule_computation(sample):
                    params = self.crn.params.specification_for(sample)
                    lnK = np.log(self.crn.get_equilibrium_constants(params))
                    return executor.submit(optimize.basinhopping,
                                           self._dist_to_equilib,
                                           np.zeros(N.shape[0]),
                                           niter=basinhopping,
                                           minimizer_kwargs = kwargs | {'args': (N, sample, lnK)})

                jobs = [schedule_computation(sample) for sample in initial]
                results = [job.result() for job in jobs]
                if not all(res['success'] for res in results):
                    warnings.warn('\n'.join(res.lowest_optimization_result.message
                                            for res in results if not res['success']))
                equilibrium = [N.T @ result.x + C for result, C in zip(results, initial)]
                # FIXME: where to set self._simplex?

        return xr.DataArray(self.crn.post_process_state(equilibrium),
                            initial.coords, name=initial.name)

    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray|dict,
            observe: str|Callable[[xr.DataArray], xr.DataArray],
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: xr.DataArray|None = None,
            **options) -> lmfit.minimizer.MinimizerResult:
        """Fit rate constants to match experimental equilibrium

        Parameters
        ----------
        data: xarray.DataArray
            1D experimental equilibrium data

        observe: string or function
            If observe is a string, it specifies a species name of the model
            that is mapped against the provided data. If it is a function,
            it must map model states to the data domain

        conversion: callable

            *Deprecated since 0.3.1: use observe instead*

            Function to convert calculated equilibrium into observable,
            typically along the line of
            lambda eq: eq.sel(species='Signal')

        error: optional xr.DataArray with rfu over time or float
            Standard deviations of measured data. If provided, the fit minimizes
            the WSSR = (model-data)**2/error**2. Otherwise, it minimizes the
            relative deviation (data/model - 1)**2

        Returns
        -------
        lmfit.FitResult
        """
        initial = self.crn.state(initial)

        if conversion:
            warnings.warn("Argument conversion is deprecated. Use observe instead.",
                          DeprecationWarning, stacklevel=2)
            if not observe:
                observe = conversion
        convert = (lambda conc: conc.sel(species=observe)) if isinstance(observe, str) else observe

        if data.ndim == 1 and len(data) > 1:
            # FIXME: only if self.initial has sample dimension
            content_dim = data.dims[0]
            initial = initial.loc[data.coords[content_dim]]

        def wssr(model):
            return (model-data)/error

        def reldev(model):
            return data/model - 1

        residual = wssr if error is not None else reldev

        def objective(params, **opts):
            nonlocal initial
            self.crn.params = params
            eq = self.eval(initial, **opts)
            initial = eq
            return residual(convert(eq))**2

        orig_params = self.crn.params
        params = orig_params.copy()
        params.fix_outside(data)
        opts = {'method': 'nelder-mead'} | options
        fit = lmfit.minimize(objective, params, **opts)
        self.crn.params = orig_params
        return fit

    @staticmethod
    def _dist_to_equilib(X, N, C, lnK):
        """Compute distance to equilibrium distribution
    
        This is used by CRN.equilibrate to minimize the
        distance to the equilibrium in an iterative
        gradient decent.
        """
        # pylint: disable=invalid-name
        if not C.any():
            return 0
        Y = N.T @ X + C.data
        if out_of_bounds := Y[Y<0].sum():
            return (1-out_of_bounds)*1e14
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.nansum(N*np.log(Y).T, axis=1) - lnK
        Z = np.where(np.isnan(Z), 0, Z)
        return np.linalg.norm(Z)
