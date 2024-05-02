"""Collection of regression tests that expose issues

Newly discovered issues should be reflected in one or several functions in this module.
These are not unit tests, but will typically initiate the writing of additional unit
tests to pinpoint the issue. 
"""
import pytest
import xarray as xr
from nanosuite import crn

def test_issue_0001():
    """Ensure that CRN params can be assigned via CRN.params."""
    crn_a = crn.from_string("""
        A + B -> C; k1 = 10
        C + D -> X + A; k2 = 5
        A [impure] + C -> Z; p=0.03
    """)

    crn_b = crn.from_string("""
        A + B -> C; k1
        C + D -> X + A; k2
        A [impure] + C -> Z; p
    """)

    crn_b.params = crn_a.params
    assert crn_b.params['k1'] == 10
    assert crn_b.params['k2'] == 5
    assert crn_b.params['p'] == 0.03
