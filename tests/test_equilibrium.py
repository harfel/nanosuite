import pytest
import xarray as xr
import nanosuite as ns

def test_equilibrium_eval():
    model = ns.crn.from_string("""A <=> Z""")
    initial = model.state(A=100)
    eq = model.equilibrium(initial)
    assert eq.eval().sel(species='A') == pytest.approx(50)

def test_equilibrium_fit():
    model = ns.crn.from_string("""A <=> Z""")
    model.params['kf1'].vary = False
    initial = model.state(A=1)
    eq = model.equilibrium(initial)
    data = xr.DataArray([0.2], {'species': ['A']})

    fit = eq.fit(data, lambda eq: eq.sel(species='A'))

    assert fit.params['kb1'] == pytest.approx(0.25, 1e-4)
