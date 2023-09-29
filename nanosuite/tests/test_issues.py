import pytest
import xarray as xr
from .. import crn

def test_issue_0001():
    crn_A = crn.from_string("""
        A + B -> C; k1 = 10
        C + D -> X + A; k2 = 5
        A [impure] + C -> Z; p=0.03
    """)

    crn_B = crn.from_string("""
        A + B -> C; k1
        C + D -> X + A; k2
        A [impure] + C -> Z; p
    """)

    crn_B.params = crn_A.params
    assert crn_B.params['k1'] == 10
    assert crn_B.params['k2'] == 5
    assert crn_B.params['p'] == 0.03


def test_issue_0002():
    impure_crn = crn.from_string("""
        A [impure] + B -> X
        A + B -> Y
    """)
    initial = xr.DataArray(
        [10, 10],
        {'species': ['A', 'B']}
    )
    try:
        impure_crn.integrate(initial)
    except ValueError as x:
        pytest.fail(str(x))
