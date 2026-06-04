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
    assert assay.all_rfu.shape == (54, 961)
    assert assay.rfu.shape == (42, 961)

def test_init_with_setup_file():
    assay = Assay(setup_file=setup_file, rfu_file=rfu_file)
    sample = assay.rfu.sel(sample='Sample X14').sample
    assert (assay.setup.sel(sample=sample, species='Signal') == 4e-9).all()

def test_init_with_groups():
    assay_1 = Assay(rfu_file=rfu_file, setup_file=setup_file)
    assay_2 = Assay(rfu_file=rfu_file, groups={
        "Responses": slice("Sample X1", "Sample X9"),
        "Negative": ["Sample X10", "Sample X11"],
        "Calibration": slice("Sample X12", "Sample X16"),
        "Positive": ["Sample X17", "Sample X18"],
    })

    assert all(assay_1.rfu.group == assay_2.rfu.group)

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
        {'sample': pd.Index(["Sample X1", "Sample X2", "Sample X3", "Sample X4", "Sample X5",
                             "Sample X6", "Sample X7", "Sample X8", "Sample X9", "Sample X10",
                             "Sample X11", "Sample X12", "Sample X13", "Sample X14", "Sample X15",
                             "Sample X16", "Sample X17", "Sample X18"]),
         'species': ["Input", "Probe", "Fuel", "Signal"]}
    ))
    assert (abs(assay_1.setup-assay_2.setup)<1e-21).all()

def test_init_with_setup_dict():
    assay_1 = Assay(rfu_file=rfu_file, setup_file=setup_file)
    assay_2 = Assay(rfu_file=rfu_file, setup={
        "Input": [0, 0, 0, 5e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Probe": 9*[1e-8] + [0, 1e-8, 1e-8, 8e-9, 6e-9, 4e-9, 2e-9, 0, 0],
        "Fuel": 9*[1.5e-8] + [0, 0, 0, 1.3e-8, 1.1e-8, 9e-9, 7e-9, 5e-9, 5e-9],
        "Signal": 12*[0] + [2e-9, 4e-9, 6e-9, 8e-9, 1e-8, 1e-8],
    })
    assert (abs(assay_1.setup-assay_2.setup)<1e-21).all()

def test_init_setup_file_and_groups_are_exclusive():
    with pytest.raises(ValueError):
        Assay(setup_file=setup_file, rfu_file=rfu_file, groups={'Unknown': "Sample X1"})

def test_init_setup_file_and_setup_are_exclusive():
    with pytest.raises(ValueError):
        Assay(setup_file=setup_file, rfu_file=rfu_file, setup=xr.DataArray([1]))

def test_init_setup_dict_with_groups():
    assay_1 = Assay(rfu_file=rfu_file, setup_file=setup_file)
    assay_2 = Assay(rfu_file=rfu_file, setup={
        "Input": [0, 0, 0, 5e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Probe": 9*[1e-8] + [0, 1e-8, 1e-8, 8e-9, 6e-9, 4e-9, 2e-9, 0, 0],
        "Fuel": 9*[1.5e-8] + [0, 0, 0, 1.3e-8, 1.1e-8, 9e-9, 7e-9, 5e-9, 5e-9],
        "Signal": 12*[0] + [2e-9, 4e-9, 6e-9, 8e-9, 1e-8, 1e-8],
    }, groups={
        "Responses": slice("Sample X1", "Sample X9"),
        "Negative": ["Sample X10", "Sample X11"],
        "Calibration": slice("Sample X12", "Sample X16"),
        "Positive": ["Sample X17", "Sample X18"],
    })
    assert (abs(assay_1.setup-assay_2.setup)<1e-21).all()

def test_init_supports_groups():
    assay = Assay(rfu_file=rfu_file, groups={'Positive': ['Sample X17', 'Sample X18'],
                                             'Negative': ['Sample X10', 'Sample X11']})
    assert len(assay.all_rfu[assay.all_rfu.group == 'Positive']) == 6
    assert len(assay.mean[assay.mean.group == 'Positive']) == 2

def test_init_imports_deactivated():
    """Ensure that deactivated wells are properly imported from Excel"""
    assay = Assay(rfu_file)
    assert assay.all_rfu.active.any() and not assay.all_rfu.active.all()

def test_deactivate():
    """Ensure that wells can be activated and deactivated"""
    assay = Assay(rfu_file)
    assay.activate(assay.all_rfu.well)

    assay.deactivate(["B03", "C10"])
    assert tuple(assay.all_rfu.sel(active=False).well.data) == ('B03', 'C10')
    assert assay.rfu.attrs['deactivated_cells'] == ["B03", "C10"]

    assay.activate("B03")
    assert assay.all_rfu.sel(active=False).well.data == 'C10'
    assert assay.rfu.attrs['deactivated_cells'] == ["C10"]

def test_mean():
    """Ensure correct average calculation"""
    assay = Assay(rfu_file)
    for sample in assay.rfu.sample.to_series().unique():
        assert (assay.mean.sel(sample=sample)
                == assay.rfu.sel(sample=sample)
                            .sel(active=True).mean(axis=0)).all()

def test_excel_time_units():
    assay_1 = Assay(Path(__file__).parent / 'data/testdata_003_RUC.xlsx')
    assay_2 = Assay(Path(__file__).parent / 'data/testdata_004_RUC.xlsx')
    assert (assay_1.rfu.time == assay_2.rfu.time).all()

def test_rfu_contains_only_active_wells():
    assay = Assay(rfu_file)
    assert not assay.all_rfu.active.all()
    assert assay.rfu.active.all()

@pytest.mark.parametrize("assayfile", ['testdata_001_RUC.xlsx', 'testdata_002_RUC.xlsx',
                                       'testdata_003_RUC.xlsx', 'testdata_004_RUC.xlsx'])
def test_ensure_all_testdata_can_be_loaded(assayfile):
    """Ensure correct handling of different header field formatting"""
    Assay(Path(__file__).parent / 'data' / assayfile)

def test_convert_accepts_one_arg():
    """Ensure Assay.convert can be called with one argument for pos_rfu"""
    assay = Assay(rfu_file,
                  groups={'positive': ['Sample X17', 'Sample X18']},
                  setup={'Signal': 5})
    from_rfu, to_rfu = assay.convert(assay.rfu[assay.rfu.group=="positive"])

    assert from_rfu(assay.rfu).dims == ('well', 'time')
    assert to_rfu(assay.setup).dims == ('sample', 'time')

def test_convert_accepts_two_args():
    """Ensure Assay.convert can be called with two arguments for pos_rfu and neg_rfu"""
    assay = Assay(rfu_file,
                  groups={'negative': ['Sample X11', 'Sample X12'],
                          'positive': ['Sample X17', 'Sample X18']},
                  setup={'Signal': 5})
    from_rfu, to_rfu = assay.convert(assay.rfu[assay.rfu.group=="positive"],
                                     assay.rfu[assay.rfu.group=="negative"])

    assert from_rfu(assay.rfu).dims == ('well', 'time')
    assert to_rfu(assay.setup).dims == ('sample', 'time')

def test_convert_accepts_four_args():
    """Ensure Assay.convert can be called with four arguments"""
    assay = Assay(rfu_file,
                  groups={'negative': ['Sample X11', 'Sample X12'],
                          'positive': ['Sample X17', 'Sample X18']},
                  setup={'Signal': 5})
    neg_conc = 1e-9*xr.DataArray([10], {'species': ['Probe']})
    pos_conc = 1e-9*xr.DataArray([10], {'species': ['Signal']})
    from_rfu, to_rfu = assay.convert(assay.rfu[assay.rfu.group=="positive"],
                                     assay.rfu[assay.rfu.group=="negative"],
                                     neg_conc, pos_conc)

    assert from_rfu(assay.rfu).dims == ('well', 'species', 'time')
    assert to_rfu(assay.setup).dims == ('sample', 'time')

def test_convert_average():
    assay = Assay(rfu_file, setup_file)
    control = assay.rfu[assay.rfu.group=='Positive']
    from_rfu, to_rfu = assay.convert(control, method='average', transient=assay.rfu.time[-1]//2)

    assert from_rfu(assay.rfu).dims == ('well', 'time')
    assert to_rfu(assay.setup).dims == ('sample', 'time')

def test_to_concentrations_accepts_scalar_pos_conc():
    assay = Assay(rfu_file, setup_file)
    concs = assay.to_concentrations(1e-6)
    assert (concs < 1e-6).all()

def test_to_concentrations_accepts_vector_pos_conc():
    assay = Assay(rfu_file, setup_file)
    pos = xr.DataArray([1, 0.5, 0.25], {'species': ['A', 'B', 'C']})
    concs = assay.to_concentrations(pos)
    assert (concs < 1).all()

def test_to_concentrations_accepts_vector_concs():
    assay = Assay(rfu_file, setup_file)
    neg = xr.DataArray([0, 0.5, 0.75], {'species': ['A', 'B', 'C']})
    pos = xr.DataArray([1, 0.5, 0.25], {'species': ['A', 'B', 'C']})
    concs = assay.to_concentrations(pos, neg)
    assert (concs < 1).all()

def test_to_concentrations_does_not_convert_controls():
    assay = Assay(rfu_file, setup_file)
    concs = assay.to_concentrations()
    assert len(concs.sample) == 14

@pytest.mark.skip("TODO: add this feature")
def test_to_concentrations_works_with_average_conversions():
    assay = Assay(rfu_file, setup_file)
    concs = assay.to_concentrations(1e-8, method='average')
    assert not xr.ufuncs.isinf(concs).any()
