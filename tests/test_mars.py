"""Unit tests for mars
"""
import os
import pytest
import pandas as pd
import xarray as xr
from nanosuite.mars import Assay

def test_deactivate():
    """Ensure that wells can be activated and deactivated"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'))
    assay.activate(assay.full_plate.well)

    assay.deactivate(["B03", "C10"])
    assert len(assay.plate.content) == len(assay.full_plate.content) - 2
    assert assay.plate.attrs['deactivated_cells'] == "B03, C10"

    assay.activate("B03")
    assert len(assay.plate.content) == len(assay.full_plate.content) - 1
    assert assay.plate.attrs['deactivated_cells'] == "C10"

def test_import_reactivated():
    """Ensure that deactivated wells are properly imported from Excel"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'))
    assert not all(assay.active_wells)

def test_avg():
    """Ensure correct average calculation"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'))
    for sample in pd.Series(assay.plate.sample.data).unique():
        assert (assay.mean.sel(sample=sample)
                == assay.plate.sel(sample=sample).mean(axis=0)).all()

@pytest.mark.parametrize("assayfile", ['testdata_001_RUC.xlsx', 'testdata_002_RUC.xlsx'])
def test_ensure_all_testcases_can_be_loaded(assayfile):
    """Ensure correct handling of different header field formatting"""
    Assay(os.path.join(os.path.dirname(__file__), 'data', assayfile))

def test_plate_setup():
    """Ensure correct setup of assay content"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'))
    content = assay.plate_setup(1e-9, A=10, B=20, C=list(range(1, 19)))
    assert (content[0] == [1e-8, 2e-8, 1e-9]).all()
    assert (abs(content[-1] - [1e-8, 2e-8, 1.8e-8]) < 1e-23).all()

def test_convert_accepts_one_arg():
    """Ensure Assay.convert can be called with one argument for pos_rfu"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'),
                  {'positive': ['Sample X17', 'Sample X18']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"))

def test_convert_accepts_two_args():
    """Ensure Assay.convert can be called with two arguments for pos_rfu and neg_rfu"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'),
                  {'negative': ['Sample X11', 'Sample X12'],
                   'positive': ['Sample X17', 'Sample X18']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"),
                                     assay.plate.sel(group="negative"))

def test_convert_accepts_four_args():
    """Ensure Assay.convert can be called with four arguments"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'),
                  {'negative': ['Sample X11', 'Sample X12'],
                   'positive': ['Sample X17', 'Sample X18']})
    neg_conc = 1e-9*xr.DataArray([10], {'content': ['Probe']})
    pos_conc = 1e-9*xr.DataArray([10], {'content': ['Signal']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"),
                                     assay.plate.sel(group="negative"),
                                     neg_conc, pos_conc)
