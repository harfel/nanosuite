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

def test_excel_time_units():
    assay_1 = Assay(os.path.join(os.path.dirname(__file__), 'data', 'testdata_003_RUC.xlsx'))
    assay_2 = Assay(os.path.join(os.path.dirname(__file__), 'data', 'testdata_004_RUC.xlsx'))
    assert (assay_1.plate.time == assay_2.plate.time).all()

@pytest.mark.parametrize("assayfile", ['testdata_001_RUC.xlsx', 'testdata_002_RUC.xlsx',
                                       'testdata_003_RUC.xlsx', 'testdata_004_RUC.xlsx'])
def test_ensure_all_testdata_can_be_loaded(assayfile):
    """Ensure correct handling of different header field formatting"""
    Assay(os.path.join(os.path.dirname(__file__), 'data', assayfile))

def test_plate_setup():
    """Ensure correct setup of assay content"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'))
    content = 1e-9*assay.plate_setup(A=10, B=20, C=list(range(1, 19)))
    assert (content[0] == [1e-8, 2e-8, 1e-9]).all()
    assert (abs(content[-1] - [1e-8, 2e-8, 1.8e-8]) < 1e-23).all()

def test_convert_accepts_one_arg():
    """Ensure Assay.convert can be called with one argument for pos_rfu"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'),
                  groups={'positive': ['Sample X17', 'Sample X18']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"))

    coord = assay.plate_setup(Signal=5).sel(species="Signal")

    assert from_rfu(assay.plate).dims == ('content', 'time')
    assert to_rfu(coord).dims == ('content', 'time')

def test_convert_accepts_two_args():
    """Ensure Assay.convert can be called with two arguments for pos_rfu and neg_rfu"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'),
                  groups={'negative': ['Sample X11', 'Sample X12'],
                          'positive': ['Sample X17', 'Sample X18']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"),
                                     assay.plate.sel(group="negative"))

    coord = assay.plate_setup(Signal=5).sel(species="Signal")

    assert from_rfu(assay.plate).dims == ('content', 'time')
    assert to_rfu(coord).dims == ('content', 'time')

def test_convert_accepts_four_args():
    """Ensure Assay.convert can be called with four arguments"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../nanosuite/examples/edc_RFU.xlsx'),
                  groups={'negative': ['Sample X11', 'Sample X12'],
                          'positive': ['Sample X17', 'Sample X18']})
    neg_conc = 1e-9*xr.DataArray([10], {'species': ['Probe']})
    pos_conc = 1e-9*xr.DataArray([10], {'species': ['Signal']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"),
                                     assay.plate.sel(group="negative"),
                                     neg_conc, pos_conc)

    assert from_rfu(assay.plate).dims == ('content', 'species', 'time')
    assert to_rfu(assay.plate_setup(Signal=5)).dims == ('content', 'time')
