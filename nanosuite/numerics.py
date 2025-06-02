"""Numerical algorithms for chemical reaction networks

These methods work in conjunction with crn.CRN and crn.PartitionedCRN.
"""
from __future__ import annotations
from concurrent import futures
from typing import Iterable, TYPE_CHECKING
import lmfit
import numpy as np
import pandas as pd                      # type: ignore
from scipy.integrate import solve_ivp    # type: ignore
from scipy.optimize import basinhopping  # type: ignore
import xarray as xr                      # type: ignore
from .utils import Cache

if TYPE_CHECKING:
    from typing import Any, Callable
    from .crn import CRN, ParameterMap


class Trajectory:
    """Trajectory of a CRN

    Parameters
    ----------
    initial_condition: 1D or 2D xarray.DataArray
        the last coord must denote species concentrations

        If initial_condition is a 1D vector, Trajectory.eval returns
        a 2D DataArray of states. If initial_condition is a 2D DataArray,
        the return value is a 3D DataArray with trajectories for
        each initial condition.
    """
    def __init__(self, crn: CRN, initial: xr.DataArray|dict):
        self.crn = crn

        initial = crn.state(initial)
        if initial.ndim not in [1, 2]:
            raise ValueError("Initial condition must be 1D or 2D")
        if any(param.value==float('inf') for param in crn.params.values()):
            self.initial = crn.perform_burst_reactions(initial)
        else:
            self.initial = initial

    def eval(self,
             t_eval: Iterable[float]|float|None = None,
             cache: Cache|None = None, **options) -> xr.DataArray:
        """Evaluate trajectory at given time points

    Internally, the method uses scipy.integrate.solve_ivp.
    Optional keyword arguments (method, atol, rtol, etc.) are
    passed to solve_ivp.

    Parameters
    ----------
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

        if not cache:
            cache = Cache()

        times: pd.Index = stratify_t_eval(t_eval)

        if len(self.initial.dims) == 1:
            params = self.crn.params.specification_for(self.initial)
            traj = self._do_integration(self.initial, params, times, options)
        else:
            with futures.ProcessPoolExecutor() as executor:
                @cache.compute
                def schedule_computation(sample):
                    params = self.crn.params.specification_for(sample)
                    return executor.submit(self._do_integration, sample, params, times, options)

                jobs = [schedule_computation(sample) for sample in self.initial]
                traj = [job.result() for job in jobs]

        traj = xr.DataArray(traj,
                            [(dim, self.initial.indexes[dim]) for dim in self.initial.dims]
                            + [times],
                            name=self.initial.name)
        return self.crn.post_process_state(traj)

    def fit(self,
            data: xr.DataArray,
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: float|xr.DataArray = 1.,
            **options) -> lmfit.minimizer.MinimizerResult:
        """Fit model parameters to experimental data

        Parameters
        ----------
        data: xarray.DataArray with rfu over time
        conversion: optional function to convert concentrations to RFU values
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

        if data.ndim == 2:
            content_dim = data.dims[0]
            initial = self.initial[self.initial.coords[content_dim].isin(data.coords[content_dim])]
        else:
            initial = self.initial

        cache = Cache(2*len(initial))

        def objective(params):
            self.crn.params = params
            model = conversion(self.eval(t_eval=data.time, cache=cache))
            return (data-model)/error

        original = self.crn.params
        params = original.copy()
        params.fix_outside(data)
        params['t0'].max = float(data.time[0]) # TODO: respect injections
        fit = lmfit.minimize(objective, params, **options)
        self.crn.params = original
        return fit

    def _do_integration(self, initial: xr.DataArray,
                        params: ParameterMap, times: pd.Index, options: dict) -> np.ndarray:
        """Integrate crn for initial condition at given times

        Internally, this method uses the method of van der Schaft et al.
        (2011) SIAM J Appl Math 73(2):953-973.
        """
        # pylint: disable=invalid-name

        Z = self.crn.complex_graph
        A = self.crn.get_complex_adjacency(params)
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
    def __init__(self, crn: CRN, initial: xr.DataArray|dict):
        self.crn = crn

        initial = crn.state(initial)
        if initial.ndim not in [1, 2]:
            raise ValueError("Initial condition must be 1D or 2D")
        if any(param.value==float('inf') for param in crn.params.values()):
            self.initial = crn.perform_burst_reactions(initial)
        else:
            self.initial = initial

    def eval(self, **options) -> xr.DataArray:
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
        N = self.crn.stoichiometry_matrix

        kwargs: dict[str, Any] = {'method': 'Nelder-Mead',
                                  'options': {'xatol': 1e-21, 'maxiter': 100}}
        kwargs.update(options)

        if self.initial.ndim == 1:
            params = self.crn.params.specification_for(self.initial)
            lnK = np.log(self.crn.get_equilibrium_constants(params))
            kwargs['args'] = (N, self.initial, lnK)
            result = basinhopping(self._dist_to_equilib, np.zeros(N.shape[0]), niter=100,
                                  minimizer_kwargs = kwargs)
            equilibrium = N.T @ result.x + self.initial

        else:
            with futures.ProcessPoolExecutor() as executor:
                def schedule_computation(sample):
                    params = self.crn.params.specification_for(sample)
                    lnK = np.log(self.crn.get_equilibrium_constants(params))
                    return executor.submit(basinhopping,
                                           self._dist_to_equilib,
                                           np.zeros(N.shape[0]),
                                           niter=100,
                                           minimizer_kwargs = kwargs | {'args': (N, sample, lnK)})

                jobs = [schedule_computation(sample) for sample in self.initial]
                equilibrium = [N.T @ job.result().x + C
                               for job, C in zip(jobs, self.initial)]

        return xr.DataArray(self.crn.post_process_state(equilibrium),
                            self.initial.coords, name=self.initial.name)

    def fit(self,
            data: xr.DataArray,
            conversion: Callable[[xr.DataArray], xr.DataArray],
            **options) -> lmfit.minimizer.MinimizerResult:
        """Fit rate constants to match experimental equilibrium

        Parameters
        ----------
        data: xarray.DataArray
            1D experimental equilibrium data

        conversion: callable
            Function to convert calculated equilibrium into observable,
            typically along the line of
            lambda eq: eq.sel(species='Signal')

        Returns
        -------
        lmfit.FitResult
        """
        orig_initial = self.initial

        if data.ndim == 2:
            content_dim = data.dims[0]
            self.initial = self.initial.loc[data.coords[content_dim]]

        def objective(params, **opts):
            self.crn.params = params
            eq = self.eval(**opts)
            self.initial = eq
            return ((data - conversion(eq))**2).data

        orig_params = self.crn.params
        params = orig_params.copy()
        params.fix_outside(data)
        params['t0'].vary = False  # TODO: make this the default and only vary in CRN.fit
        opts = {'method': 'nelder-mead'}
        opts.update(**options)
        fit = lmfit.minimize(objective, params, **opts)
        self.initial = orig_initial
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
        Y = N.T @ X + C.data
        if out_of_bounds := Y[Y<0].sum():
            return (1-out_of_bounds)*1e14
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.nansum(N*np.log(Y).T, axis=1) - lnK
        Z = np.where(np.isnan(Z), 0, Z)
        return np.linalg.norm(Z)
