"""Unit tests for mars
"""
from pathlib import Path
import pytest
import pandas as pd
import xarray as xr
from nanosuite.mars import Assay

rfu_file = Path(__file__).parent / '../nanosuite/examples/edc_RFU.xlsx'
setup_file = Path(__file__).parent / '../nanosuite/examples/edc_setup.xlsx'

def test_init_with_rfu():
    assay = Assay(rfu_file=rfu_file)
    assert assay.plate.shape == (42, 961)

def test_init_with_setup_file():
    assay = Assay(setup_file=setup_file, rfu_file=rfu_file)
    content = assay.plate.sel(sample='Sample X14').content
    assert str(content.group.data[0]) == 'Calibration'
    assert (assay.setup.sel(sample=content.sample, species='Signal') == 4e-9).all()

def test_init_with_groups():
    assay_1 = Assay(rfu_file=rfu_file, setup_file=setup_file)
    assay_2 = Assay(rfu_file=rfu_file, groups={
        "Responses": slice("Sample X1", "Sample X9"),
        "Negative": ["Sample X10", "Sample X11"],
        "Calibration": slice("Sample X12", "Sample X16"),
        "Positive": ["Sample X17", "Sample X18"],
    })

    assert all(assay_1.plate.content == assay_2.plate.content)

def test_init_with_setup_array():
    assay_1 = Assay(rfu_file=rfu_file, setup_file=setup_file)
    assay_2 = Assay(rfu_file=rfu_file, setup=1e-9*xr.DataArray(
        [[0, 10, 15, 0],
         [0, 10, 15, 0],
         [0, 10, 15, 0],
         [0.005, 10, 15, 0],
         [0.01, 10, 15, 0],
         [0.05, 10, 15, 0],
         [0.1, 10, 15, 0],
         [0.5, 10, 15, 0],
         [1, 10, 15, 0],
         [0, 0, 0, 0],
         [0, 10, 0, 0],
         [0, 10, 0, 0],
         [0, 8, 13, 2],
         [0, 6, 11, 4],
         [0, 4, 9, 6],
         [0, 2, 7, 8],
         [0, 0, 5, 10],
         [0, 0, 5, 10]],
        {'content': pd.MultiIndex.from_tuples([("Responses", "Sample X1"),
                                                ("Responses", "Sample X2"),
                                                ("Responses", "Sample X3"),
                                                ("Responses", "Sample X4"),
                                                ("Responses", "Sample X5"),
                                                ("Responses", "Sample X6"),
                                                ("Responses", "Sample X7"),
                                                ("Responses", "Sample X8"),
                                                ("Responses", "Sample X9"),
                                                ("Negative", "Sample X10"),
                                                ("Negative", "Sample X11"),
                                                ("Calibration", "Sample X12"),
                                                ("Calibration", "Sample X13"),
                                                ("Calibration", "Sample X14"),
                                                ("Calibration", "Sample X15"),
                                                ("Calibration", "Sample X16"),
                                                ("Positive", "Sample X17"),
                                                ("Positive", "Sample X18")],
                                               names=['group', 'sample']),
         'species': ["Input", "Probe", "Fuel", "Signal"]}
    ))
    assert (abs(assay_1.setup-assay_2.setup)<1e-21).all()

def test_init_setup_file_and_groups_are_exclusive():
    with pytest.raises(ValueError):
        assay = Assay(setup_file=setup_file, rfu_file=rfu_file, groups={'Unknown': "Sample X1"})

def test_init_setup_file_and_setup_are_exclusive():
    with pytest.raises(ValueError):
        assay = Assay(setup_file=setup_file, rfu_file=rfu_file, setup=xr.DataArray([1]))

def test_init_setup_and_groups_are_exclusive():
    with pytest.raises(ValueError):
        assay = Assay(rfu_file=rfu_file, setup=xr.DataArray([1]), groups={'Unknown': "Sample X1"})

def test_deactivate():
    """Ensure that wells can be activated and deactivated"""
    assay = Assay(rfu_file)
    assay.activate(assay.full_plate.well)

    assay.deactivate(["B03", "C10"])
    assert len(assay.plate.content) == len(assay.full_plate.content) - 2
    assert assay.plate.attrs['deactivated_cells'] == "B03, C10"

    assay.activate("B03")
    assert len(assay.plate.content) == len(assay.full_plate.content) - 1
    assert assay.plate.attrs['deactivated_cells'] == "C10"

def test_import_reactivated():
    """Ensure that deactivated wells are properly imported from Excel"""
    assay = Assay(rfu_file)
    assert not all(assay.active_wells)

def test_avg():
    """Ensure correct average calculation"""
    assay = Assay(rfu_file)
    for sample in pd.Series(assay.plate.sample.data).unique():
        assert (assay.mean.sel(sample=sample)
                == assay.plate.sel(sample=sample).mean(axis=0)).all()

def test_excel_time_units():
    assay_1 = Assay(Path(__file__).parent / 'data/testdata_003_RUC.xlsx')
    assay_2 = Assay(Path(__file__).parent / 'data/testdata_004_RUC.xlsx')
    assert (assay_1.plate.time == assay_2.plate.time).all()

@pytest.mark.parametrize("assayfile", ['testdata_001_RUC.xlsx', 'testdata_002_RUC.xlsx',
                                       'testdata_003_RUC.xlsx', 'testdata_004_RUC.xlsx'])
def test_ensure_all_testdata_can_be_loaded(assayfile):
    """Ensure correct handling of different header field formatting"""
    Assay(Path(__file__).parent / 'data' / assayfile)

def test_convert_accepts_one_arg():
    """Ensure Assay.convert can be called with one argument for pos_rfu"""
    assay = Assay(rfu_file,
                  groups={'positive': ['Sample X17', 'Sample X18']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"))

    coord = assay.plate_setup(Signal=5).sel(species="Signal")

    assert from_rfu(assay.plate).dims == ('content', 'time')
    assert to_rfu(coord).dims == ('content', 'time')

def test_convert_accepts_two_args():
    """Ensure Assay.convert can be called with two arguments for pos_rfu and neg_rfu"""
    assay = Assay(rfu_file,
                  groups={'negative': ['Sample X11', 'Sample X12'],
                          'positive': ['Sample X17', 'Sample X18']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"),
                                     assay.plate.sel(group="negative"))

    coord = assay.plate_setup(Signal=5).sel(species="Signal")

    assert from_rfu(assay.plate).dims == ('content', 'time')
    assert to_rfu(coord).dims == ('content', 'time')

def test_convert_accepts_four_args():
    """Ensure Assay.convert can be called with four arguments"""
    assay = Assay(rfu_file,
                  groups={'negative': ['Sample X11', 'Sample X12'],
                          'positive': ['Sample X17', 'Sample X18']})
    neg_conc = 1e-9*xr.DataArray([10], {'species': ['Probe']})
    pos_conc = 1e-9*xr.DataArray([10], {'species': ['Signal']})
    from_rfu, to_rfu = assay.convert(assay.plate.sel(group="positive"),
                                     assay.plate.sel(group="negative"),
                                     neg_conc, pos_conc)

    assert from_rfu(assay.plate).dims == ('content', 'species', 'time')
    assert to_rfu(assay.plate_setup(Signal=5)).dims == ('content', 'time')
