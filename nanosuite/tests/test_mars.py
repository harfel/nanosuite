import xarray as xr
import numpy as np
import pandas as pd
from nanosuite.mars import Assay


class MockAssay(Assay):
    def __init__(self):     # pylint: disable=super-init-not-called
        times = np.linspace(0, 100, 101)
        content = [('Sample X1', 'B01'), ('Sample X1', 'B02'), ('Sample X1', 'B03'),
                   ('Sample X2', 'B04'), ('Sample X2', 'B05'), ('Sample X2', 'B06'),
                   ('Sample X3', 'B07'), ('Sample X3', 'B08'), ('Sample X3', 'B09')]
        self.full_plate = xr.DataArray(np.random.rand(len(content), len(times)),
        {'content': pd.MultiIndex.from_tuples(content, names=["sample", "well"]), 'time': times})
        self.active_wells = xr.ones_like(self.full_plate, dtype=bool)


def test_deactivate():
    assay = MockAssay()

    assay.deactivate(["B02", "B05"])
    assert len(assay.plate.content) == len(assay.full_plate.content) - 2

    assay.activate("B05")
    assert len(assay.plate.content) == len(assay.full_plate.content) - 1
