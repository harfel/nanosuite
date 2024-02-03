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

def test_avg():
    """Ensure correct average calculation"""
    assay = Assay(os.path.join(os.path.dirname(__file__), '../examples/edc_RFU.xlsx'))
    for sample in pd.Series(assay.plate.sample.data).unique():
        assert (assay.mean().sel(sample=sample)
                == assay.plate.sel(sample=sample).mean(axis=0)).all()
