"""Tools to work with BMG Labtech MARS plate reader data analysis files.
"""
from openpyxl import load_workbook
import numpy as np


class Assay:
    """Access to MARS data.

    Assay instances have the following attributes:

    Attributes
    ----------
    times: 1D numpy.array
        Times at which measurements have been taken.
    wells: 2D numpy.array
        Fluorescence values at each well and time point.
    contents:
        mapping from strings to well indices. See Assay.__init__.
    path: file path
        Path of the associated xlsx file (read only).
    deactivated: list of well indices
        Wells that had been blanked by the user.
    
    """
    def __init__(self, path, resolution=1, contents=None):
        """Plate reader data as saved by MARS.

        Parameters
        ----------
        path: file path
        resolution: int (defaults to 1)
            If set to n, every nth data point is added to the assay
        contents: mapping of strings to well indices
            If not given, the mapping is autimatically inferred from
            the content column.
        """
        deactivated_info_cell = 11, 1
        time_row = 14
        content_col = 1
        sample_first_row = 15

        self.path = path

        workbook = load_workbook(self.path)
        worksheet = workbook["Table All Cycles"]

        sample_last_row = worksheet.max_row

        # read deactivated wells from header info
        self.deactivated = [
            well.strip()
            for well in worksheet.cell(*deactivated_info_cell).value.split(':')[-1].split(';')
        ]
		# FIXME: Assay.deactivated should be indices into Assay.wells

        # time and raw read information (incl. deactivated wells)
        self.times = np.array([cell.value
                               for cell in np.array(worksheet[time_row][2::resolution])])
        self.wells = np.array([
            [cell.value for cell in worksheet[y][2::resolution]]
            for y in range(sample_first_row, sample_last_row+1)
        ], dtype=float)

        # mapping of content to well indices (excl. deactivated wells)
        if not contents:
            contents = {}
            for idx, row in enumerate(worksheet[sample_first_row: sample_last_row]):
                if row[0].value in self.deactivated:
                    continue
                content = row[content_col].value
                contents[content] = contents.get(content, []) + [idx]
        self.contents = contents

        self._mean = None
        self._std = None

    def __repr__(self):
        return f'<Assay "{self.path}">'

    @property
    def mean(self):
        """Average fluorescence of all wells with identical content."""
        if self._mean is None:
            self._mean = np.array([
                np.mean(self.wells[idx], axis=0)
                for idx in self.contents.values()
            ])
        return self._mean

    @property
    def std(self):
        """Fluorescence standard deviation of all wells with identical content."""
        if self._std is None:
            self._std = np.array([
                np.std(self.wells[idx], axis=0)
                for idx in self.contents.values()
            ])
            # replace 0 std values by smallest positive value
            self._std[self._std==0] = np.min(self._std, where=self._std!=0, initial=np.inf)
        return self._std
