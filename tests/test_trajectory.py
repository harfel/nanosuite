import pytest
import numpy as np
import xarray as xr
import nanosuite as ns


def test_eval():
    model = ns.crn.from_string("""A -> Z""")
    traj = model.trajectory(model.state(A=1))

    result = traj.eval(t_eval=10)

    assert result.sel(species='Z', time=10) == pytest.approx(1-np.exp(-10))

def test_eval_teval_is_optional():
    """Ensure that scalar t_eval is optional"""
    model = ns.crn.from_string("A + B -> C")
    traj = model.trajectory(model.state(A=10, B=10))

    result = traj.eval()

    assert result.time[0] == ns.crn.DEFAULT_INTEGRATION_START
    assert result.time[-1] == ns.crn.DEFAULT_INTEGRATION_END
    assert len(result.time) == ns.crn.DEFAULT_INTEGRATION_POINTS

def test_eval_teval_accepts_float():
    """Ensure that scalar t_eval is taken as end value"""
    model = ns.crn.from_string("A + B -> C")
    traj = model.trajectory(model.state(A=10, B=10))

    result = traj.eval(t_eval=66)

    assert result.time[0] == ns.crn.DEFAULT_INTEGRATION_START
    assert result.time[-1] == 66
    assert len(result.time) == ns.crn.DEFAULT_INTEGRATION_POINTS

def test_eval_scalar_teval_starts_at_t0():
    """Ensure that scalar t_eval uses t0 as start value"""
    model = ns.crn.from_string("A + B -> C")
    model['t0'].value = -5
    traj = model.trajectory(model.state(A=10, B=10))
    
    result = traj.eval(t_eval=66)

    assert result.time[0] == -5
    assert result.time[-1] == 66
    assert len(result.time) == ns.crn.DEFAULT_INTEGRATION_POINTS

def test_eval_teval_accepts_tuple():
    """Ensure that tuple t_eval is taken as start and end values"""
    model = ns.crn.from_string("A + B -> C")
    traj = model.trajectory(model.state(A=10, B=10))
    
    result = traj.eval(t_eval=(5, 66))

    assert result.time[0] == 5
    assert result.time[-1] == 66
    assert len(result.time) == ns.crn.DEFAULT_INTEGRATION_POINTS
    assert result.sel(species='C')[0] > 0

def test_eval_teval_accepts_iterable():
    """Ensure that scalar t_eval is taken as time points"""
    model = ns.crn.from_string("A + B -> C")
    traj = model.trajectory(model.state(A=10, B=10))

    result = traj.eval(t_eval=[2, 4, 8, 16, 32])

    assert len(result.time) == 5
    assert result.time[0] == 2
    assert result.time[-1] == 32

def test_eval_teval_accepts_dataarrays():
    """Ensure that xarrays are taken directly as time dimension"""
    model = ns.crn.from_string("A + B -> C")
    traj = model.trajectory(model.state(A=10, B=10))
    times = xr.DataArray([0, 1, 2, 3, 4], {'testtime': [0, 1, 2, 3, 4]})

    result = traj.eval(t_eval=times)

    assert (result.testtime == times).all()

def test_eval_teval_accepts_scalar_arrays():
    """Ensure that scalar arrays are taken as end point of a time interval"""
    model = ns.crn.from_string("A + B -> C")
    model['t0'].value = -1
    traj = model.trajectory(model.state(A=10, B=10))
    times = xr.DataArray([0, 1, 2, 3, 4], {'time': [0, 1, 2, 3, 4]})

    result = traj.eval(t_eval=times.max())

    assert result.time[0] == -1
    assert result.time[-1] == 4
    assert len(result.time) == ns.crn.DEFAULT_INTEGRATION_POINTS

def test_fit():
    model = ns.crn.from_string("""A -> Z; k""")
    model.params['t0'].vary = False
    initial = model.state(A=1)
    traj = model.trajectory(initial)
    data = xr.DataArray([0.8], {'time': [10]})
    k = -np.log(0.2)/10

    result = traj.fit(data, conversion=lambda state: state.sel(species='Z'))

    assert result.params['k'].value == pytest.approx(k)

