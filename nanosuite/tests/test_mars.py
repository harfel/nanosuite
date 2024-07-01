"""Unit tests for mars
"""
import os
import pandas as pd
from nanosuite.mars import Assay

def test_deactivate():
    """Ensure that wells can be activated and deactivated"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../examples/edc_RFU.xlsx'))
    assay.activate(assay.full_plate.well)

    assay.deactivate(["B03", "C10"])
    assert len(assay.plate.content) == len(assay.full_plate.content) - 2
    assert assay.plate.attrs['deactivated_cells'] == "B03, C10"

    assay.activate("B03")
    assert len(assay.plate.content) == len(assay.full_plate.content) - 1
    assert assay.plate.attrs['deactivated_cells'] == "C10"

def test_import_reactivated():
    """Ensure that deactivated wells are properly imported from Excel"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../examples/edc_RFU.xlsx'))
    assert not all(assay.active_wells)

def test_avg():
    """Ensure correct average calculation"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../examples/edc_RFU.xlsx'))
    for sample in pd.Series(assay.plate.sample.data).unique():
        assert (assay.mean().sel(sample=sample)
                == assay.plate.sel(sample=sample).mean(axis=0)).all()

def test_varying_header_fields():
    """Ensure correct handling of different header field formatting"""
    Assay(os.path.join(os.path.dirname(__file__), './data/testdata_001_RUC.xlsx'))

def test_content():
    """Ensure correct setup of assay content"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../examples/edc_RFU.xlsx'))
    content = assay.plate_setup(1e-9, A=10, B=20, C=list(range(1, 19)))
    assert (content[0] == [1e-8, 2e-8, 1e-9]).all()
    assert (abs(content[-1] - [1e-8, 2e-8, 1.8e-8]) < 1e-23).all()
