"""Tools to work with BMG Labtech MARS plate reader data analysis files.
"""
import warnings
from typing import cast, Callable, Iterable, Iterator, Optional
import numpy as np
import pandas as pd
import xarray as xr
import lmfit # type: ignore


def _unique(it: Iterable) -> Iterator:
    """Iteration filter that drops adjacent repeated elements"""
    last = None
    for current in it:
        if current == last:
            continue
        last = current
        yield current


class Assay:
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

    active_wells: 1D xarray DataArray
        Wells that have not been blanked by the user.
    """
    def __init__(self, path: str, groups: Optional[dict[str, list[str]|slice]] = None):
        """Plate reader data as saved by MARS.

        Parameters
        ----------
        path: str
            file path

        groups: dict
            mapping of group names to a list or slice of sample names
            e.g. {'System 1': slice("Sample X1", "Sample X5"), "Control": ["Sample X6"]}
        """
        deactivated_info_cell = 11, 1
        info_vals = ["user", "path", "test ID", "test name",
                     "date", "ID1", "ID2", "ID3"]

        self.path = path
        groups = groups or {}

        # Create header df and extract data
        df_total = pd.read_excel(self.path, header=None)
        df_header = df_total.iloc[:deactivated_info_cell[0], :1]

        # extract info from headers, into dictionary
        attributes = {}
        n_inf = 0
        for inf in info_vals:
            attributes[inf] = str(df_header.iloc[n_inf]).split(": ")[1].split("\n")[0]
            n_inf += 1
        deactivated = [
            cell.strip()
            for cell in
            str(df_header.iloc[10, 0]).rsplit(': ', maxsplit=1)[-1].split('; ')
        ]
        attributes["deactivated_cells"] = ', '.join(deactivated)

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

        samples = df_main['Content'][1:]

        df_groups = pd.DataFrame(
            [(k, val) for k, vals in groups.items() for val in
             cast(Iterable, samples.loc[samples[samples==vals.start].index[0]
                                        :samples[samples==vals.stop].index[-1]].unique()
                            if isinstance(vals, slice) else vals)],
            columns=['group', 'sample'])

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

        self.active_wells = ~self.full_plate.well.isin(deactivated)
        self.plate = self.full_plate[self.active_wells]

    def __repr__(self) -> str:
        return f'<Assay "{self.path}">'

    def _repr_html_(self) -> str:
        return self.plate._repr_html_() # pylint: disable=protected-access

    def deactivate(self, wells: str|list[str]) -> None:
        """Deactivate a well or list of wells

        Activating and deactivating wells will set a new Assay.plate --
        invalidating any reference to the previous plate attribute.
        """
        if isinstance(wells, str):
            wells = [wells]
        self.active_wells = self.active_wells.where(~self.active_wells.well.isin(wells), False)
        self.plate = self.full_plate[self.active_wells]
        self.plate.attrs['deactivated_cells'] = ', '.join(
            self.full_plate[~self.active_wells].well.values)

    def activate(self, wells: str|list[str]) -> None:
        """Activate a well or list of wells

        Activating and deactivating wells will set a new Assay.plate --
        invalidating any reference to the previous plate attribute.
        """
        if isinstance(wells, str):
            wells = [wells]
        self.active_wells = self.active_wells.where(~self.active_wells.well.isin(wells), True)
        self.plate = self.full_plate[self.active_wells]
        self.plate.attrs['deactivated_cells'] = ', '.join(
            self.full_plate[~self.active_wells].well.values)

    def mean(self) -> xr.DataArray:
        """Return sample means

        Returns
        -------
        DataArray of average fluorescence of all active wells that belong to
        the same sample.
        """
        avg = self.plate.groupby('sample').mean(dim='content')
        # We need to reorder the result to match the original content index.
        # This is because of https://github.com/pydata/xarray/issues/757
        samples = list(_unique(self.plate.sample.values))
        original_order = xr.DataArray(range(len(avg)), {'sample': samples})
        return avg.sortby(original_order)

    def std(self, ddof: int = 0) -> xr.DataArray:
        """Return sample standard deviation

        Parameters
        ----------
        ddof: optional int (default = 0)
            Difference in number of degrees of freedom. Return value is
            calculated as 1/(N-ddof) sum_{i=1}^N(x_i - <x>) where N is the
            number of samples.

        Returns
        -------
        DataArray of fluorescence standard deviation of all active wells that
        belong to the same sample.
        """
        avg = self.plate.groupby('sample').std(dim='content', ddof=ddof)
        # Same situation as in Array.mean
        samples = list(_unique(self.plate.sample.values))
        original_order = xr.DataArray(range(len(avg)), {'sample': samples})
        return avg.sortby(original_order)

    def calibrate(
        self, pos_conc: xr.DataArray, neg_conc: xr.DataArray,
        pos_rfu: Optional[xr.DataArray]=None,
        neg_rfu: Optional[xr.DataArray]=None, *,
        method: str='direct',
        error: Optional[Callable[[float], float]]=None,
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
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
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
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
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
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
