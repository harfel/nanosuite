import pytest
import pandas as pd
import xarray as xr
import nanosuite as ns


def test_eval():
    model = ns.crn.from_string("""A <=> Z""")
    eq = model.equilibrium(model.state(A=100))

    result = eq.eval()

    assert result.sel(species='A') == pytest.approx(50)

def test_eval_reversible():
    """Ensure eval is accurate"""
    model = ns.crn.from_string("A <=> B ; kf=2, kb=1")
    eq = model.equilibrium(model.state(A=10))

    result = eq.eval()

    rate_ratio = model.params['kf'].value / model.params['kb'].value
    conc_ratio = result.sel(species='B') / result.sel(species='A')
    assert result.sum() == 10
    assert conc_ratio == pytest.approx(rate_ratio)

def test_eval_multiple_state():
    """Permit equilbrium to be calculated for multiple states"""
    model = ns.crn.from_string("A + B <=> C; kf, kb")
    initial = model.state(A=[1, 10, 100], B=1)
    A0 = initial.sel(species='A')
    B0 = initial.sel(species='B')
    K = model.params['kf'].value / model.params['kb'].value
    Ceq = (A0+B0+1/K)/2 - ((A0-B0)**2 + 2*(A0+B0)/K + 1/K**2)**0.5/2

    result = model.equilibrium(initial).eval()

    assert (result.sel(species='C') == pytest.approx(Ceq)).all()

def test_eval_with_sample_map():
    model = ns.crn.from_string("""
        A <=> B; kf=1, kr=1
    """)

    initial = model.state(A=3*[110])

    sample_map = pd.DataFrame([
        ['A', 'B1'],
        ['A', 'B2'],
        ['A', 'B3'],
    ], columns=['A', 'B'], index=initial.coords['sample'])

    model.params = model.parametrize_for(sample_map, kr = ['B'])
    model.params['kr'].value = [0.1, 1., 10.]

    result = model.equilibrium(initial).eval()

    assert result.sel(species='A').values == pytest.approx([10., 55., 99.99999999])

def test_eval_ignores_unspecified_samples():
    model = ns.crn.from_string("""
        A <=> B; kf=1, kr=1
    """)

    initial = model.state(A=[10, 0])
    sample_map = pd.DataFrame([
        ['A1', 'B'],
        [None, None],
    ], columns=['A', 'B'], index=initial.coords['sample'])
    model.params = model.parametrize_for(sample_map, kr = ['A'])
    model.params['kr'].value = [1., 1.]

    result = model.equilibrium(initial).eval()

    assert result.sel(species='A').values == pytest.approx([5, 0])

def test_eval_with_subspecies():
    model = ns.crn.from_string("""
        A contains reactive with p_A = 0.5
        A [reactive] <=> B ; kf = 1, kb = 3
    """)
    state = model.state(A=8)

    result = model.equilibrium(state).eval()

    conc_ratio = result.sel(species='B') / result.sel(species='A_reactive')
    rate_ratio = model.params['kf'].value / model.params['kb'].value
    assert conc_ratio == pytest.approx(rate_ratio)
    assert result.sel(species='A') == 7

def test_fit():
    model = ns.crn.from_string("""A <=> Z""")
    model.params['kf1'].vary = False
    initial = model.state(A=1)
    eq = model.equilibrium(initial)
    data = xr.DataArray([0.2], {'species': ['A']})

    fit = eq.fit(data, lambda eq: eq.sel(species='A'))

    assert fit.params['kb1'] == pytest.approx(0.25, 1e-4)
