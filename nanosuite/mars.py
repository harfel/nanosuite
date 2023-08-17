"""Tools to work with BMG Labtech MARS plate reader data analysis files.
"""
from typing import cast, Callable, Dict, List, Optional, Tuple
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

    plate: 2D xarray DataArray
        Fluorescence values of all active wells and time points.

    full_plate: 2D xarray DataArray
        Fluorescence values of all wells and time points.

    deactivated: list of well indices
        Wells that had been blanked by the user.
    """
    def __init__(self, path: str, groups: Optional[Dict[str, List[str]]] = None):
        """Plate reader data as saved by MARS.

        Parameters
        ----------
        path: str
              file path
        """
        deactivated_info_cell = 11, 1
        attr_first_row = 1
        attr_last_row = 9
        time_row = 14
        sample_first_row = 15
        well_col = 0
        content_col = 1

        self.path = path

        groups = groups or {}

        groups_reverse = {
            sample: group
            for group, samples in groups.items()
            for sample in samples
        }

        workbook = load_workbook(self.path)
        worksheet = workbook["Table All Cycles"]

        sample_last_row = worksheet.max_row

        # read deactivated wells from header info
        deactivated_cells = worksheet.cell(*deactivated_info_cell).value
        self.deactivated = [
            well.strip()
            for well in cast(str, deactivated_cells) .split(':')[-1].split(';')
        ] if deactivated_cells else []

        self.times = np.array([cell.value
                               for cell in np.array(worksheet[time_row][2:])])

        content = pd.MultiIndex.from_tuples([
            (groups_reverse.get(cast(str, row[content_col].value), 'Unknown'),
             row[content_col].value,
             row[well_col].value)
            for idx, row in enumerate(
                worksheet.iter_rows(min_row=sample_first_row,
                                    max_row=sample_last_row)
            )
        ], names=("group", "sample", "well"))

        self.full_plate = xr.DataArray(
            [[cell.value for cell in worksheet[y][2:]]
             for y in range(sample_first_row, sample_last_row+1)],
            [("content", content), ("time", self.times)],
            name="RFU",
            attrs=dict(
                cast(str, cell[0].value).partition(': ')[::2]
                for cell in worksheet.iter_rows(min_row=attr_first_row,
                                                max_row=attr_last_row)
            ),
        ).astype(float)

        mask = ~self.full_plate.well.isin(self.deactivated)
        self.plate = self.full_plate[mask]

    def __repr__(self) -> str:
        return f'<Assay "{self.path}">'

    def _repr_html_(self) -> str:
        return self.plate._repr_html_() # pylint: disable=protected-access

    def calibrate(
        self, positive: str, negative: str, pos_conc: float, neg_conc: float
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """Compute transforms between raw and normalized RFU values

        Transforms are based on the linear relations

        (rfu-neg)/(pos-neg) = (conc-neg_conc)/(pos_conc-neg_conc)

        where the left hand side captures the linear interpolation
        between repeat-averaged negative and positive control RFU
        values and the right hand side captures the linear
        interpolation between the scalar concentrations neg_pos and
        right_pos.

        Parameters
        ----------
        positive: str -- element of self.plate.coords['sample']
            Label of the positive control sample
        negative: str -- element of self.plate.coords['sample']
            Label of the negative control sample
        pos_conc: float
            Concentration associated with positive control
        neg_conc: float
            Concentration associated with negative control

        Returns
        -------
        Two functions from_rfu and to_rfu with signatures

        fromRFU(rfu: np.ndarray) -> np.ndarray
        toRFU(rfu: np.ndarray) -> np.ndarray
        """
        # TODO: support different calibration methods
        neg = self.plate.loc[negative].mean(axis=0).to_numpy()
        pos = self.plate.loc[positive].mean(axis=0).to_numpy()
        def from_rfu(rfu):
            # FIXME: make sure that this works for all supported rfu.ndims
            return neg_conc + (rfu-neg) * (pos_conc-neg_conc)/(pos-neg)
        def to_rfu(conc):
            return neg + (conc-neg_conc) * (pos-neg)/(pos_conc-neg_conc)
        return from_rfu, to_rfu
