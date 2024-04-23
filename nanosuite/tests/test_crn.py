"""Unit tests for crn
"""
import math
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
        C + D -> E + B; k2=1.2
        A [impure] + D -> E; p=0.0
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
            <td style="text-align: left">B + E</td>
            <td style="text-align: left">k2 = 1.2</td>
        </tr><tr>
            <td style="text-align: right">A [impure] + D</td>
            <td style="text-align: center">&LongRightArrow;</td>
            <td style="text-align: left">E</td>
            <td style="text-align: left">p = 0</td>
        </tr></table>'''
    assert rep == ''.join(line.strip() for line in expected.split('\n'))

def test_from_string_irreversible():
    """Ensure correct parsing of irreversible reactions"""
    test_crn = crn.from_string("""
        A + B -> C
        C + D -> E
    """)
    assert len(test_crn.reactions) == 2

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

def test_from_string_rate_value():
    """Ensure correct parsing of rate constant values"""
    test_crn = crn.from_string("""
        A + B -> C; 10
    """)
    assert test_crn.params['k1'] == 10

def test_from_string_reversible():
    """Ensure correct parsing of reversible reactions"""
    test_crn = crn.from_string("""
        A + B <=> C
    """)
    assert len(test_crn.reactions) == 2
    assert 'kf1' in test_crn.params
    assert 'kb1' in test_crn.params

def test_instantaneous():
    test_crn = crn.from_string("""
       A -> Z;  k=inf
    """)
    raise RuntimeError("FIXME: write test")

def test_impurity_1():
    test_crn = crn.from_string("""
       A contains impure

       A -> X
       A [impure] -> Y;  k=0.1
    """)
    raise RuntimeError("FIXME: write test")

def test_impurity_2():
    test_crn = crn.from_string("""
       A contains impure with p = 0.5

       A [impure] -> X
    """)
    raise RuntimeError("FIXME: write test")

def test_impurity_3():
    test_crn = crn.from_string("""
       A contains impure with p=0.5 rest pure

       A -> X
       A [impure] -> Y;  k=0.1
    """)
    raise RuntimeError("FIXME: write test")

def test_impurity_4():
    test_crn = crn.from_string("""
       A contains imp1 with p=0.25,
         contains imp2 with p=0.25
       A [imp1] -> X;  k=inf
       A [imp1] -> Y;  k=inf
       A [imp2] -> Z;  k=inf
    """)
    raise RuntimeError("FIXME: write test")

def test_impurity_5():
    test_crn = crn.from_string("""
       A contains imp1 with p=0.25, 
         contains imp2 with p=0.25
       A [imp1] -> X;  k=inf
       A [imp1] -> Y;  k=0.1
       A [imp2] -> Z;  k=inf
    """)
    raise RuntimeError("FIXME: write test")

