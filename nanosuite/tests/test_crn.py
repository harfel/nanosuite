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

def test_repr_html():
    """Ensure correct HTML representation"""
    test_crn = crn.from_string("""
        A + B -> C; k1=1.1
        C + D -> B + E; k2=1.2
        A [impure] + D -> E; k=inf
    """)
    rep = ''.join(line.strip() for line in test_crn._repr_html_().split('\n')) # pylint: disable=protected-access
    expected = '''<table><tr>
            <td style="text-align: right">A_pure + B</td>
            <td style="text-align: center">&LongRightArrow;</td>
            <td style="text-align: left">C</td>
            <td style="text-align: left">k1 = 1.1</td>
        </tr>
        <tr>
            <td style="text-align: right">C + D</td>
            <td style="text-align: center">&LongRightArrow;</td>
            <td style="text-align: left">B + E</td>
            <td style="text-align: left">k2 = 1.2</td>
        </tr><tr>
            <td style="text-align: right">A_impure + D</td>
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
    """Ensure reaction can be made burst by changing their rate constant"""
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

def test_impurities():
    """Ensure correct split into subspecies"""
    test_crn = crn.from_string("""
       A contains impure with p_impure = 0.1

       A -> X;           k = 10.
       A [impure] -> Y;  k
    """)
    initial = test_crn.state(A=1.)
    traj = test_crn.integrate(initial)
    assert (abs(traj.sel(species='X') - 9*traj.sel(species='Y')) < 1e-15 ).all()

def test_impurities_can_burst():
    """Ensure correct behaviour of impure burst reactions"""
    test_crn = crn.from_string("""
       A contains impure with 0.5

       A [impure] -> X; k=inf
    """)
    initial = test_crn.state(A=1.0)
    traj = test_crn.integrate(initial)
    assert traj.sel(species="A", time=0) == 0.5
    assert traj.sel(species="A_impure", time=0) == 0.
    assert traj.sel(species="A", time=100) == 0.5


# FIXME: write test for impure reactions with multiple initial states
