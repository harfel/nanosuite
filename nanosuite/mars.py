"""Tools to work with BMG Labtech MARS plate reader data analysis files.
"""
import warnings
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import lmfit # type: ignore


class Assay:
    # TODO: API to activate/deactivate wells
    """Access to MARS data.

    Assay instances have the following attributes:

    Attributes
    ----------
    path: str
        Path of the associated xlsx file (read only).

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
        # TODO: support slices instead of lists in groups
        deactivated_info_cell = 11, 1
        info_vals = ["user", "path", "test ID", "test name",
                     "date", "ID1", "ID2", "ID3"]

        self.path = path
        groups = groups or {}

        df_groups = pd.DataFrame(
            [(k, val) for k, vals in groups.items() for val in vals],
            columns=['group', 'sample'])

        # Create header df and extract data
        df_total = pd.read_excel(self.path, header=None)
        df_header = df_total.iloc[:deactivated_info_cell[0], :1]

        # extract info from headers, into dictionary
        attributes = {}
        n_inf = 0
        for inf in info_vals:
            attributes[inf] = str(df_header.iloc[n_inf]).split(": ")[1].split("\n")[0]
            n_inf += 1
        self.deactivated = [
            cell.strip()
            for cell in
            df_header.iloc[10, 0].rsplit(': ', maxsplit=1)[-1].split('; ')
        ]
        attributes["deactivated_cells"] = ', '.join(self.deactivated)

        # Create main df and eval if it needs to be transposed
        df_main = df_total.iloc[len(df_header)+1:,]

        if isinstance(df_main.iloc[1, 2], str):
            df_main = df_main.T

        df_main.columns = pd.Index(df_main.iloc[0])
        df_main = df_main[1:]
        df_main.index = pd.Index(np.arange(1, len(df_main) + 1))

        # extract coordinates from dataframe
        times = df_main.iloc[:1, 2:].values.flatten().astype(float)
        main_array = df_main.iloc[1:, 2:].values


        df_content = df_main[["Content", "Well"]][1:]
        df_content.columns = pd.Index(["sample", "well"])
        if df_groups.empty:
            df_content["group"] = 'Unknown'
        else:
            df_content = df_content.merge(df_groups, on='sample')
        df_multicontent = pd.MultiIndex.from_frame(df_content)

        # from dataframe to xarray
        self.full_plate = xr.DataArray(main_array,
            [("content", df_multicontent), ("time", times)],
            name="RFU",
            attrs=attributes,).astype(float)

        mask = ~self.full_plate.well.isin(self.deactivated)
        self.plate = self.full_plate[mask]

    def __repr__(self) -> str:
        return f'<Assay "{self.path}">'

    def _repr_html_(self) -> str:
        return self.plate._repr_html_() # pylint: disable=protected-access

    @staticmethod
    def _default_error(_):
        return 1

    def calibrate(
        self, pos_conc: xr.DataArray, neg_conc: xr.DataArray,
        pos_rfu: Optional[xr.DataArray]=None,
        neg_rfu: Optional[xr.DataArray]=None, *,
        method: str='direct',
        error: Optional[Callable[[float], float]]=None,
    ) -> Tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between modelled RFU values and concentrations

        Parameters
        ----------
        pos_conc, neg_conc: xr.DataArray
            Concentration vector of the positive and negative controls
        pos_rfu, neg_rfu: optional xr.DataArray
            RFU values of positive and negative controls
            (for direct method only)
        method: 'direct' (default) or 'relaxation'
        error: optional function mapping float on float values
            Error model used for fitting when using the 'relaxation' method 

        Returns
        -------
        Two functions from_rfu and to_rfu with signatures

        fromRFU(rfu: xr.DataArray) -> xr.DataArray
        toRFU(rfu: xr.DataArray) -> xr.DataArray

        See documentation of the specialized calibration methods for detail.
        """
        if method == 'direct':
            if error is not None:
                warnings.warn("Argument error is ignored with method 'direct'.")
            return self.calibrate_direct(pos_conc, neg_conc, pos_rfu, neg_rfu)
        if method == 'relaxation':
            if pos_rfu or neg_rfu:
                warnings.warn("Arguments pos_rfu and neg_rfu are ignored with method 'relaxation'.")
            return self.calibrate_relaxation(pos_conc, neg_conc, error=error)
        raise ValueError(f"""Unsupported calibration method: '{method}'
        Needs to be one of: direct [default], relaxation""")

    def calibrate_relaxation(
        self, pos_conc: xr.DataArray, neg_conc: xr.DataArray,
        error: Optional[Callable[[float], float]]=None,
    ) -> Tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between modelled RFU values and concentrations

        Parameters
        ----------
        pos_conc, neg_conc: xr.DataArray
            Concentration vector of the positive and negative controls
        error: optional function mapping float on float values
            If provided, the function should return the expected
            measurement error for a given fluorescence.

        Returns
        -------
        Two functions from_rfu and to_rfu with signatures

        fromRFU(rfu: xr.DataArray) -> xr.DataArray
        toRFU(rfu: xr.DataArray) -> xr.DataArray


        Same as Assay.calibrate_direct, but calibration is based on a
        double exponential relaxation model that is fitted against
        the controls. Specifically, the model assumes that fluoresence
        quickly equilibrates towards an equilibrium value that itself
        slowly equilibrates over time.
        """
        # TODO: report fit statistics
        pos_rfu = str(pos_conc.sample.data)
        neg_rfu = str(neg_conc.sample.data)
        error = error if error is not None else lambda rfu: 1.

        controls = self.plate[self.plate.sample.isin([pos_rfu, neg_rfu])]

        def double_relaxation(time, r_1, r_2, rfu_0, rfu_1, rfu_inf):   # pylint: disable=too-many-arguments
            if r_1 == r_2:
                return (rfu_inf + (rfu_0 - rfu_inf)*np.exp(-r_1*time)
                        + r_1*(rfu_1 - rfu_inf)*time*np.exp(-r_1*time))
            return (rfu_inf + (rfu_0 - rfu_inf)*np.exp(-r_1*time)
                    + r_1*(rfu_1 - rfu_inf)/(r_1 - r_2)*(np.exp(-r_2*time)-np.exp(-r_1*time)))

        def well_model(time, params):
            negative = double_relaxation(
                time,
                params['r1'], params['r2'],
                params['N0'], params['N1'], params['Ninf']
            )
            positive = double_relaxation(
                time,
                params['r1'], params['r2'],
                params['P0'], params['P1'], params['Pinf']
            )
            return np.array([
                {
                    pos_rfu: positive,
                    neg_rfu: negative,
                }[str(sample.data)]
                for sample in controls.sample
            ])

        def residuals_for(data):
            def residuals(params):
                model = well_model(data.coords['time'], params)
                return (data-model)/error(data)
            return residuals

        split = controls.shape[-1]//5
        params = lmfit.create_params(
            r1 = {'value': float(10/controls.time[split]) , 'min': 0., 'vary': True},
            r2 = {'value': float(1/controls.time[-1]), 'min': 0., 'vary': True},
            N0 = {'value': float(controls[:3].mean(axis=0)[0]), 'min': 0.},
            N1 = {'value': float(controls[:3].mean(axis=0)[split]), 'min': 0.},
            Ninf = {'value': float(controls[:3].mean(axis=0)[-1]), 'min': 0.},
            P0 = {'value': float(controls[3:].mean(axis=0)[0]), 'min': 0.},
            P1 = {'value': float(controls[3:].mean(axis=0)[split]), 'min': 0.},
            Pinf = {'value': float(controls[3:].mean(axis=0)[-1]), 'min': 0.},
        )
        fit = lmfit.minimize(residuals_for(controls), params)

        neg_model = double_relaxation(
            controls.time, fit.params['r1'], fit.params['r2'],
            fit.params['N0'], fit.params['N1'], fit.params['Ninf']
        )

        pos_model = double_relaxation(
            controls.time, fit.params['r1'], fit.params['r2'],
            fit.params['P0'], fit.params['P1'], fit.params['Pinf']
        )

        return self.calibrate_direct(pos_conc, neg_conc, pos_model, neg_model)

    def calibrate_direct(
        self, pos_conc: xr.DataArray, neg_conc: xr.DataArray,
        pos_rfu=None, neg_rfu=None
    ) -> Tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between raw RFU values and concentrations

        Transforms are based on the linear relations

        (rfu-neg_rfu)/(pos_rfu-neg_rfu) = (conc-neg_conc)/(pos_conc-neg_conc)

        where the left hand side isa linear interpolation from negative
        to positive control RFU values and the right hand side expresses
        the reaction coordinate from neg_conc to pos_conc.

        Parameters
        ----------
        pos_conc: xarray.DataArray
            Concentrations associated with positive control
        neg_conc: xarray.DataArray
            Concentrations associated with negative control
        pos_rfu: optional xarray.DataArray
            RFU values of the positive control            
        neg_rfu: optional xarray.DataArray
            RFU values of the negative control

        If pos_rfu or neg_rfu are not provided, the method uses
        the mean fluorescence values of all wells associated with
        pos_conc.sample and neg_conc.sample.

        Returns
        -------
        Two functions from_rfu and to_rfu with signatures

        fromRFU(rfu: xr.DataArray) -> xr.DataArray
        toRFU(rfu: xr.DataArray) -> xr.DataArray
        """
        pos_rfu = (
            pos_rfu
            if pos_rfu is not None
            else self.plate.sel(sample=pos_conc.sample).mean(axis=0)
        )
        neg_rfu = (
            neg_rfu
            if neg_rfu is not None
            else self.plate.sel(sample=neg_conc.sample).mean(axis=0)
        )
        def from_rfu(rfu):
            return neg_conc + (pos_conc-neg_conc) * (rfu-neg_rfu)/(pos_rfu-neg_rfu)
        def to_rfu(conc):
            rfu = (neg_rfu + (pos_rfu-neg_rfu)
                   * ((conc-neg_conc)/(pos_conc-neg_conc))
                        .where(pos_conc != neg_conc)
                        .mean(axis=conc.get_axis_num("species")))
            rfu.name = "RFU"
            return rfu.transpose()
        return from_rfu, to_rfu
