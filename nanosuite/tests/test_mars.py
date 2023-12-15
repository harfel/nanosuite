from typing import Optional
import xarray as xr
import numpy as np
import pandas as pd
from nanosuite.mars import Assay


class MockAssay(Assay):
    path: str = 'N/A'

    def __init__(self, data: Optional[xr.DataArray] = None, samples: int = 1, repeats: int = 1):    # pylint: disable=super-init-not-called
        if data is None:
            times = np.linspace(0, 100, 11)
            data = np.random.rand(repeats*samples, len(times))
        else:
            times = np.linspace(0, 100, len(data[0]))
            samples = len(data)//repeats
        content = [(f'Sample X{idx+1}', f'{chr(65+idx)}{rep+1:02}')
                   for idx in range(samples) for rep in range(repeats)]
        self.full_plate = xr.DataArray(data,
                                       {'content': pd.MultiIndex.from_tuples(
                                            content, names=["sample", "well"]
                                        ), 'time': times})

        self.active_wells = ~self.full_plate.well.isin([])
        self.plate = self.full_plate[self.active_wells]


def test_deactivate():
    assay = MockAssay(samples=2, repeats=3)

    assay.deactivate(["A01", "B03"])
    assert len(assay.plate.content) == len(assay.full_plate.content) - 2

    assay.activate("B03")
    assert len(assay.plate.content) == len(assay.full_plate.content) - 1

def test_avg():
    assay = MockAssay([[0, 1, 2, 3, 4],
                       [1, 2, 3, 4, 5],
                       [2, 3, 4, 5, 6],
                       [5, 4, 3, 2, 1],
                       [4, 3, 2, 1, 0],
                       [3, 2, 1, 0,-1]], repeats=3)

    assert (assay.mean().sel(sample="Sample X1") == [1, 2, 3, 4, 5]).all()
    assert (assay.std(ddof=0).sel(sample="Sample X1") == (2/3)**.5).all()

    assay.deactivate("A02")

    assert (assay.mean().sel(sample="Sample X1") == [1, 2, 3, 4, 5]).all()
    assert (assay.std(ddof=0).sel(sample="Sample X1") == 1).all()

def test_avg_sorting():
    assay = MockAssay(samples=15, repeats=3)
    avg = assay.mean()
    assert (avg.sample == [f'Sample X{idx}' for idx in range(1, 16)]).all()
