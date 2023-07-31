"""Tools to work with BMG Labtech MARS plate reader data analysis files.
"""
from typing import cast
import numpy as np
import pandas as pd
import xarray as xr
from openpyxl import load_workbook

class Assay:
    """Access to MARS data.

    Assay instances have the following attributes:

    Attributes
    ----------
    path: str
        Path of the associated xlsx file (read only).

    times: 1D numpy.array
        Times at which measurements have been taken.

    wells: 2D xarray DataArray
        Fluorescence values of all active wells and time points.

    all_wells: 2D xarray DataArray
        Fluorescence values of all wells and time points.

    deactivated: list of well indices
        Wells that had been blanked by the user.
    """
    def __init__(self, path: str,):
        """Plate reader data as saved by MARS.

        Parameters
        ----------
        path: str
              file path
        """
        deactivated_info_cell = 11, 1
        time_row = 14
        well_col = 0
        content_col = 1
        sample_first_row = 15

        self.path = path

        workbook = load_workbook(self.path)
        worksheet = workbook["Table All Cycles"]

        sample_last_row = worksheet.max_row

        # read deactivated wells from header info
        self.deactivated = [
            well.strip()
            for well in cast(str, worksheet.cell(*deactivated_info_cell).value)
                            .split(':')[-1].split(';')
        ]

        self.times = np.array([cell.value
                               for cell in np.array(worksheet[time_row][2:])])

        content = pd.MultiIndex.from_tuples([
            (row[content_col].value, row[well_col].value)
            for idx, row in enumerate(worksheet[sample_first_row: sample_last_row])
        ], names=("sample", "well"))

        self.all_wells = xr.DataArray(
            [[cell.value for cell in worksheet[y][2:]]
             for y in range(sample_first_row, sample_last_row+1)],
            [("content", content), ("time", self.times)],
            name="RFU",
            attrs={
                cell[0].value.partition(': ')[::2]
                for cell in worksheet['A1':'A9']
            },
        ).astype(float)

        self.wells = self.all_wells.where(~self.all_wells.well.isin(self.deactivated))

    def __repr__(self) -> str:
        return f'<Assay "{self.path}">'
