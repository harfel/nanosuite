"""Unit tests for crn
"""
import math
import pytest
import xarray as xr
from nanosuite import crn


def test_params_add():
    """Ensure that parameters can be added to CRN.params"""
    test_crn = crn.from_string("""
        A + B -> C; k_f
        C -> A + B; k_b
    """)
    params = test_crn.params
    params.add('dG', value=-12.5)
    params['k_b'].expr = 'k_f/exp(-dG)'
    assert 'dG' in params
    assert params['k_b'].value == params['k_f']/math.exp(-params['dG'].value)

@pytest.mark.skip("FIXME: fix this issue later")
def test_repr_html():
    """Ensure correct HTML representation"""
    test_crn = crn.from_string("""
        A + B -> C; k1=1.1
        C + D -> E + B; k2=1.2
        A [impure] + D -> E; k=inf
    """)
    rep = ''.join(line.strip() for line in test_crn._repr_html_().split('\n')) # pylint: disable=protected-access
    expected = '''<table><tr>
            <td style="text-align: right">A + B</td>
            <td style="text-align: center">&LongRightArrow;</td>
            <td style="text-align: left">C</td>
            <td style="text-align: left">k1 = 1.1</td>
        </tr>
        <tr>
            <td style="text-align: right">C + D</td>
            <td style="text-align: center">&LongRightArrow;</td>
            <td style="text-align: left">E + B</td>
            <td style="text-align: left">k2 = 1.2</td>
        </tr><tr>
            <td style="text-align: right">A [impure] + D</td>
            <td style="text-align: center">&LongRightArrow;</td>
            <td style="text-align: left">E</td>
            <td style="text-align: left">k = inf</td>
        </tr></table>'''
    assert rep == ''.join(line.strip() for line in expected.split('\n'))

def test_from_string_irreversible():
    """Ensure correct parsing of irreversible reactions"""
    test_crn = crn.from_string("""
        A + B -> C
        C + D -> E
    """)
    assert len(test_crn.reactions) == 2

def test_from_string_reversible():
    """Ensure correct parsing of reversible reactions"""
    test_crn = crn.from_string("""
        A + B <=> C
    """)
    assert len(test_crn.reactions) == 2
    assert 'kf1' in test_crn.params
    assert 'kb1' in test_crn.params

def test_from_string_rate():
    """Ensure correct parsing of reaction rates"""
    test_crn = crn.from_string("""
        A + B -> C; k = 10
    """)
    assert test_crn.params['k'].value == 10

def test_from_string_rate_name():
    """Ensure correct parsing of rate constant names"""
    test_crn = crn.from_string("""
        A + B -> C; k
    """)
    assert test_crn.params['k'] == 1

@pytest.mark.parametrize("value", [10, 1.0, 1e-1, 'inf'])
def test_from_string_rate_value(value):
    """Ensure correct parsing of rate constant values"""
    test_crn = crn.from_string(f"""
        A + B -> C; {value}
    """)
    assert test_crn.params['k1'] == float(value)

def test_implicit_rate_names():
    """Ensure the correct number of rate constants is defined"""
    test_crn = crn.from_string("""
        A -> W
        B -> X
        C -> Y; k0
        D -> Z; k0
    """)
    assert 'k0' in test_crn.params
    assert 'k1' in test_crn.params
    assert 'k2' in test_crn.params
    assert len(test_crn.params) == 3

def test_inconsistent_rate_names():
    """Ensure that rates cannot be assigned inconsistent values"""
    with pytest.raises(ValueError):
        crn.from_string("""
            A -> X; k = 1
            A -> Y; k = 2
        """)

@pytest.mark.parametrize("reactions, initial, outcome", [
    ("""A -> Z; k=inf""", [1., 0.], [0., 1.]),
    ("""A + B -> Y; k0=inf
        A + C -> Z; k1=inf""", [1., 1., 0., 3., 0.], [0., 0.75, 0.25, 2.25, 0.75]),
    ("""A -> B; k0=inf
        B -> Z; k1=inf""", [1., 0., 0.], [0., 0., 1.]),
    ("""A -> Y; k0=inf
        A -> Z; k1=inf""", [1., 0., 0.], [0., .5, .5])])
def test_burst_reactions(reactions, initial, outcome):
    """Ensure currect treatment of burst reactions"""
    test_crn = crn.from_string(reactions)
    initial = xr.DataArray(initial, {'species': test_crn.species})
    traj = test_crn.integrate(initial)
    assert (abs(traj.sel(time=0.) - outcome) < 1e-5).all()

@pytest.mark.parametrize("reactions, initial", [
    ("""A <=> B; kf=inf, kb=inf""", [1., 0.]),
    ("""A -> B; k=inf
        B -> A; k=inf""", [1., 0.]),
    ("""A -> B; k=inf
        B -> C; k
        C -> A; k""", [1., 0., 0.]),
    ])
def test_circular_burst_reactions(reactions, initial):
    """Ensure that circular burst reactions raise ValueError"""
    test_crn = crn.from_string(reactions)
    initial = xr.DataArray(initial, {'species': test_crn.species})
    with pytest.raises(ValueError):
        test_crn.integrate(initial)

def test_burst_after_initialization():
    test_crn = crn.from_string("""
        A -> X; k
    """)
    test_crn['k'].value = float('inf')
    assert len(test_crn.burst_reactions) == 1

def test_burst_reaction_name_consistancy():
    """Ensure reactions to be instantanous if their rate constant name repeats"""
    test_crn = crn.from_string("""
        A -> X; k=inf
        A -> Y; k
    """)
    assert len(test_crn.burst_reactions) == 2

def test_impurity_1():
    """Ensure correct treatment of impurities"""
    test_crn = crn.from_string("""
       A contains impure

       A -> X
       A [impure] -> Y;  k=0.1
    """)
    initial = xr.DataArray([1., 0., 0.], {'species': test_crn.species})
    traj = test_crn.integrate(initial)
    assert (traj.sel(species='X') == 10*traj.sel(species='Y')).all()

@pytest.mark.skip("FIXME: write test")
def test_impurity_2():
    """Ensure correct treatment of impurities"""
    test_crn = crn.from_string("""
       A contains impure with p = 0.5

       A [impure] -> X
    """)

@pytest.mark.skip("FIXME: write test")
def test_impurity_3():
    """Ensure correct treatment of impurities"""
    test_crn = crn.from_string("""
       A contains impure with p=0.5 rest pure

       A -> X
       A [impure] -> Y;  k=0.1
    """)

@pytest.mark.skip("FIXME: write test")
def test_impurity_4():
    """Ensure correct treatment of impurities"""
    test_crn = crn.from_string("""
       A contains imp1 with p=0.25,
         contains imp2 with p=0.25
       A [imp1] -> X;  k=inf
       A [imp1] -> Y;  k=inf
       A [imp2] -> Z;  k=inf
    """)

@pytest.mark.skip("FIXME: write test")
def test_impurity_5():
    """Ensure correct treatment of impurities"""
    test_crn = crn.from_string("""
       A contains imp1 with p=0.25, 
         contains imp2 with p=0.25
       A [imp1] -> X;  k=inf
       A [imp1] -> Y;  k=0.1
       A [imp2] -> Z;  k=inf
    """)
