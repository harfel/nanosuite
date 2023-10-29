import math
from nanosuite import crn


def test_params_add():
    test_crn = crn.from_string("""
        A + B -> C; k_f
        C -> A + B; k_b
    """)
    params = test_crn.params
    params.add('dG', value=-12.5)
    params['k_b'].expr = 'k_f/exp(-dG)'
    assert 'dG' in params
    assert params['k_b'].value == params['k_f']/math.exp(-params['dG'].value)

def test_from_string_irreversible():
    test_crn = crn.from_string("""
        A + B -> C
        C + D -> E
    """)
    assert len(test_crn.reactions) == 2

def test_from_string_rate():
    test_crn = crn.from_string("""
        A + B -> C; k = 10
    """)
    assert test_crn.params['k'].value == 10

def test_from_string_rate_name():
    test_crn = crn.from_string("""
        A + B -> C; k
    """)
    assert test_crn.params['k'] == 1

def test_from_string_rate_value():
    test_crn = crn.from_string("""
        A + B -> C; 10
    """)
    assert test_crn.params['k1'] == 10


def test_from_string_impure():
    test_crn = crn.from_string("""
        A [impure] + B -> C
    """)
    assert len(test_crn.side_reactions) == 1

def test_from_string_impure_fraction():
    test_crn = crn.from_string("""
        A [impure] + B -> C;    p=0.05
    """)
    assert test_crn.params['p'].value == 0.05
