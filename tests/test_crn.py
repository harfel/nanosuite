"""Unit tests for crn
"""
import math
import pytest
import xarray as xr
from nanosuite import crn
import lmfit


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

@pytest.mark.parametrize("reaction", ["2A -> B", "2 A -> B", "2*A -> B", "A -> 2B", "A -> 2*B"])
def test_stoichiometries(reaction):
    """Ensure that species can occur in higher stoichiometries"""
    crn.from_string(reaction)

def test_from_string_forbids_nonint_stoichiometries():
    """Forbid noninteger notation of stoichiometries"""
    with pytest.raises(ValueError):
        crn.from_string("1.0 A -> B")

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

@pytest.mark.parametrize("value", [10, '10.', 100, 1.0, 1e-1, 'inf'])
def test_from_string_rate_value(value):
    """Ensure correct parsing of rate constant values"""
    test_crn = crn.from_string(f"""
        A + B -> C; {value}
    """)
    assert test_crn.params['k1'] == float(value)

def test_from_string_repeated_reversible_rates():
    """Ensure correct treatment of common rate constants in reversible reactions."""
    model = crn.from_string("""
        A + B <=> C; kf1 = 1e7, 1e3
        C + D <=> E; kf1, kb = 1e3
        """)
    assert len(model.params) == 4
    k1 = model.reactions[(('A', 1), ('B', 1)), (('C', 1),)]
    k2 = model.reactions[(('C', 1), ('D', 1)), (('E', 1),)]
    assert k1 == k2

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
    assert len(test_crn.params) == 4

def test_forbid_t0_as_rate_name():
    """Forbid using t0 as a rate parameter"""
    with pytest.raises(ValueError):
        crn.from_string("A -> B; t0=1")

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

       A -> X;           k = 10
       A [impure] -> Y;  k
    """)
    initial = test_crn.state(A=1.)
    traj = test_crn.integrate(initial)
    assert (abs(traj.sel(species='X') - 9*traj.sel(species='Y')) < 1e-15).all()

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

def test_impurities_support_parallel_systems():
    """Ensure that impurities are compatible with parallel systems"""
    test_crn = crn.from_string("""
    A contains impure with p=0.1

    A [impure] -> X
    """)

    initial = xr.DataArray([
        [1.],
        [2.],
        [3.],
    ], {
        'system': ['1', '2', '3'],
        'species': ['A']
    })
    traj = test_crn.integrate(initial)

    assert (traj.sel(time=0, species="A_impure") - [0.1, 0.2, 0.3] < 1e-15).all()

@pytest.mark.parametrize("string", ["<>", "->", "A ->", "A B"])
def test_from_string_raises_valueerror(string):
    """Ensure that malformed input raises exceptions"""
    with pytest.raises(crn.crn_parser.CRNError):
        crn.from_string(string)

@pytest.mark.parametrize("value", ["0", "1", "1.", "0.1", ".01", "1.23e-31", "6e2"])
def test_from_string_real_formats(value):
    """Ensure parsing of parameter values"""
    crn.from_string(f"""A -> B; k={value}""")

def test_add_reaction_respects_catalysts():
    network = crn.CRN()
    network.add_reaction(educts=(('A', 1), ('C', 1)), products=(('Z', 1), ('C', 1)), rate=lmfit.Parameter('k'))
    assert len(network.species) == 3

def test_integrate_teval_is_optional():
    """Ensure that scalar t_eval is optional"""
    model = crn.from_string("A + B -> C")

    traj = model.integrate(model.state(A=10, B=10))

    assert traj.time[0] == crn.DEFAULT_INTEGRATION_START
    assert traj.time[-1] == crn.DEFAULT_INTEGRATION_END
    assert len(traj.time) == crn.DEFAULT_INTEGRATION_POINTS

def test_integrate_teval_accepts_float():
    """Ensure that scalar t_eval is taken as end value"""
    model = crn.from_string("A + B -> C")

    traj = model.integrate(model.state(A=10, B=10), t_eval=66)

    assert traj.time[0] == crn.DEFAULT_INTEGRATION_START
    assert traj.time[-1] == 66
    assert len(traj.time) == crn.DEFAULT_INTEGRATION_POINTS

def test_integrate_scalar_teval_starts_at_t0():
    """Ensure that scalar t_eval uses t0 as start value"""
    model = crn.from_string("A + B -> C")
    model['t0'].value = -5

    traj = model.integrate(model.state(A=10, B=10), t_eval=66)

    assert traj.time[0] == -5
    assert traj.time[-1] == 66
    assert len(traj.time) == crn.DEFAULT_INTEGRATION_POINTS

def test_integrate_teval_accepts_tuple():
    """Ensure that tuple t_eval is taken as start and end values"""
    model = crn.from_string("A + B -> C")

    traj = model.integrate(model.state(A=10, B=10), t_eval=(5, 66))

    assert traj.time[0] == 5
    assert traj.time[-1] == 66
    assert len(traj.time) == crn.DEFAULT_INTEGRATION_POINTS
    assert traj.sel(species='C')[0] > 0

def test_integrate_teval_accepts_iterable():
    """Ensure that scalar t_eval is taken as time points"""
    model = crn.from_string("A + B -> C")
    traj = model.integrate(model.state(A=10, B=10), t_eval=[2, 4, 8, 16, 32])
    assert len(traj.time) == 5
    assert traj.time[0] == 2
    assert traj.time[-1] == 32

def test_integrate_teval_accepts_dataarrays():
    """Ensure that xarrays are taken directly as time dimension"""
    model = crn.from_string("A + B -> C")
    times = xr.DataArray([0, 1, 2, 3, 4], {'testtime': [0, 1, 2, 3, 4]})
    traj = model.integrate(model.state(A=10, B=10), t_eval=times)
    assert (traj.testtime == times).all()

def test_integrate_teval_accepts_scalar_arrays():
    """Ensure that scalar arrays are taken as end point of a time interval"""
    model = crn.from_string("A + B -> C")
    model['t0'].value = -1
    times = xr.DataArray([0, 1, 2, 3, 4], {'time': [0, 1, 2, 3, 4]})
    traj = model.integrate(model.state(A=10, B=10), t_eval=times.max())
    assert traj.time[0] == -1
    assert traj.time[-1] == 4
    assert len(traj.time) == crn.DEFAULT_INTEGRATION_POINTS

def test_perform_burst_reactions_works_with_multiple_samples():
    """Ensure that burst reactions can be performed for a set of samples."""
    model = crn.from_string("A + B -> C; k=inf")
    init = xr.DataArray([
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
    ], {
        'sample': "a b c d e f g h i".split(),
        'species': "A B".split(),
    })
    traj = model.integrate(init)
    assert (traj.sel(time=0, species='B') == [0, 0, 0, 0, 0, 1, 2, 3, 4]).all()

@pytest.mark.parametrize("conc, extra_conc", [
    (xr.DataArray([1, 2], {'species': ['A', 'B']}), {}),
    ({'A': 1, 'B': 2},                              {}),
    (xr.DataArray([1], {'species': ['A']}),         {'B': 2}),
    ({'A': 1},                                      {'B': 2}),
])
def test_state_accepts_scalar_values(conc, extra_conc):
    model = crn.from_string("A + B -> C")
    state = model.state(conc, **extra_conc)
    assert state.sel(species='A') == 1
    assert state.sel(species='B') == 2

@pytest.mark.parametrize("conc, extra_conc", [
    (xr.DataArray([[1, 2], [1, 3]], {'samples': ['Sample X1', 'Sample X2'], 'species': ['A', 'B']}),
     {}),
    ({'A': 1, 'B': [2, 3]}, {}),
    (xr.DataArray([1], {'species': ['A']}), {'B': [2, 3]}),
    ({'A': 1}, {'B': [2, 3]}),
    (None, {'A': 1, 'B': [2, 3]})
])
def test_state_accepts_value_sequences(conc, extra_conc):
    model = crn.from_string("A + B -> C")
    state = model.state(conc, **extra_conc)
    assert (state.sel(species='A') == 1).all()
    assert state.sel(species='B')[0] == 2
    assert state.sel(species='B')[1] == 3
