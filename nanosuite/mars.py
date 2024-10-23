"""Tools to work with BMG Labtech MARS plate reader data analysis files.
"""
import warnings
from typing import cast, Callable, Dict, Iterable, Optional, Union
import numpy as np
import pandas as pd
import xarray as xr
import scipy # type: ignore
import lmfit # type: ignore


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
    path: str
    full_plate: xr.DataArray
    plate: xr.DataArray
    _mean: xr.DataArray | None = None
    _std: xr.DataArray | None = None


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
        self.path = path
        groups = groups or {}

        # Create header df and extract data
        df_total = pd.read_excel(self.path, header=None)
        content_start = df_total[df_total[0]=='Well'].index[0]
        df_header = df_total.iloc[:content_start, :1]

        # extract info from headers, into dictionary
        attributes = {}
        for field in df_header[0]:
            if isinstance(field, float):
                break
            try:
                ridx = field.rindex(': ')
                key = field[:ridx].strip()
                val = field[ridx+2:].strip()
                attributes[key] = val
            except ValueError:
                continue

        deactivated = attributes.pop(
            'Grey fields contain deactivated wells:   /   Disabled by user', '').split('; ')
        attributes['deactivated_cells'] = ', '.join(deactivated)

        # Create main df and eval if it needs to be transposed
        df_main = df_total.iloc[len(df_header):,]

        if df_main.iloc[1,0] == 'Content':
            df_main = df_main.T

        df_main.columns = pd.Index(df_main.iloc[0])
        df_main = df_main[1:]
        df_main.index = pd.Index(np.arange(1, len(df_main) + 1))

        # extract coordinates from dataframe
        # TODO: Assay should define a time_unit and respect the one in the execl file
        times = df_main.iloc[:1, 2:].values.flatten().astype(float)  # FIXME: time can be str formatted
        main_array = df_main.iloc[1:, 2:].values

        samples = df_main['Content'][1:]

        df_content = df_main[['Content', 'Well']][1:]
        df_content['group'] = "Unknown"
        df_content.columns = pd.Index(['sample', 'well', 'group'])
        df_content = df_content.reindex(columns=['group', 'sample', 'well'])
        # set df_content['group'] from groups dict
        for group, group_samples in groups.items():
            if isinstance(group_samples, slice):
                start = samples[samples==group_samples.start].index[0]
                end = samples[samples==group_samples.stop].index[-1]
                group_samples = cast(list, samples.loc[start:end].unique())
            for sample in group_samples:
                df_content.loc[df_content['sample']==sample, 'group'] = group
        df_multicontent = pd.MultiIndex.from_frame(df_content)

        # from dataframe to xarray
        self.full_plate = xr.DataArray(main_array,
            {"content": df_multicontent, "time": times},
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
        self._mean = None
        self._std = None

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
        self._mean = None
        self._std = None

    @property
    def mean(self) -> xr.DataArray:
        """Return sample means

        Returns
        -------
        DataArray of average fluorescence of all active wells that belong to
        the same sample.
        """
        if self._mean is None:
            samples = pd.Series(self.plate.sample.data).unique()
            self._mean = xr.DataArray(
                [self.plate.sel(sample=sample).mean(dim='content').data for sample in samples],
                {'content': self.plate.indexes['content'].droplevel('well').unique(),
                 'time': self.plate.time},
                attrs=self.plate.attrs, name=self.plate.name
            )
        return self._mean

    @property
    def std(self) -> xr.DataArray:
        """Return sample standard deviation

        Returns
        -------
        DataArray of fluorescence standard deviation of all active wells
        that belong to the same sample.
        """
        if self._std is None:
            samples = pd.Series(self.plate.sample.data).unique()
            self._std = xr.DataArray(
                [self.plate.sel(sample=sample).std(dim='content').data
                 for sample in samples],
                {'content': self.plate.indexes['content'].droplevel('well').unique(),
                 'time': self.plate.time},
                attrs=self.plate.attrs, name=self.plate.name
            )
        return self._std

    def plate_setup(self, factor: float, conc: Optional[Dict[str, Union[float, Iterable]]] = None,
                    **kwargs: Union[float, Iterable]) -> xr.DataArray:
        """Define plate setup

        This method generates a correctly indexed xr.DataArray that
        describes the plate setup in terms of concentrations of chemical
        species for each sample.

        Parameters
        ----------
        factor: float
            Common concentration prefactor
        conc, kwargs: Dict[str, Union[float, Iterable]]
            Concentrations of chemical species. If a float is given,
            the species is uniform over all samples. If an iterable
            is given, it must have the same length as there are
            samples in the assay.

        Returns
        -------
        An xr.DataArray with dimensions 'content' (taken from Assay.plate)
        and 'species', where each sample has species concentrations as
        provided in the method arguments.
        """
        conc = conc if conc else {}
        conc.update(kwargs)
        array = xr.DataArray(
            dims=['content', 'species'],
            coords={'content': self.plate.indexes['content'].droplevel('well').unique(),
                    'species': list(conc.keys())})
        for species, values in conc.items():
            array.loc[..., species] = values
        return factor*array

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
        warnings.warn("Assay.calibrate is deprecated and will be removed in a future "
                      "version of nanosuite. Change your code to use Assay.convert.")

        if method == 'direct':
            if error is not None:
                warnings.warn("Argument error is ignored with method 'direct'.")
            return self.calibrate_direct(pos_conc, neg_conc, pos_rfu, neg_rfu)
        if method == 'relaxation':
            return self.calibrate_relaxation(pos_conc, neg_conc, pos_rfu, neg_rfu, error=error)
        raise ValueError(f"""Unsupported calibration method: '{method}'
        Needs to be one of: direct [default], relaxation""")

    def calibrate_relaxation(
        self, pos_conc: xr.DataArray, neg_conc: xr.DataArray,
        pos_rfu: Optional[xr.DataArray]=None,
        neg_rfu: Optional[xr.DataArray]=None, *,
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
        pos_rfu = (
            pos_rfu
            if pos_rfu is not None
            else self.plate[self.plate.sample.isin(pos_conc.sample)].mean(axis=0)
        )
        neg_rfu = (
            neg_rfu
            if neg_rfu is not None
            else self.plate[self.plate.sample.isin(neg_conc.sample)].mean(axis=0)
        )
        error = error if error is not None else lambda rfu: 1.

        controls = self.plate[self.plate.sample.isin(pos_rfu.sample)
                              | self.plate.sample.isin(neg_rfu.sample)]

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
                positive if sample in pos_conc.sample else negative
                for sample in controls.sample.data
            ])

        def residuals_for(data):
            def residuals(params):
                model = well_model(data.time, params)
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
        if len(pos_conc.sample) > 1:
            assert not any(pos_conc.std(axis=pos_conc.get_axis_num('sample')))
        if len(neg_conc.sample) > 1:
            assert not any(neg_conc.std(axis=neg_conc.get_axis_num('sample')))

        pos_rfu = (
            pos_rfu
            if pos_rfu is not None
            else self.plate[self.plate.sample.isin(pos_conc.sample)].mean(axis=0)
        )
        neg_rfu = (
            neg_rfu
            if neg_rfu is not None
            else self.plate[self.plate.sample.isin(neg_conc.sample)].mean(axis=0)
        )

        pos_conc = pos_conc.mean(axis=pos_conc.get_axis_num('sample'))
        neg_conc = neg_conc.mean(axis=neg_conc.get_axis_num('sample'))

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

    def convert(
        self, pos_rfu: xr.DataArray, neg_rfu: Optional[xr.DataArray] = None,
        pos_conc: Union[xr.DataArray, float] = 1., neg_conc: Union[xr.DataArray, float] = 0., /,
        method: Optional[str] = 'direct'
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between RFU values and concentrations

        E.g.
        >>> from_rfu, to_rfu = assay.convert(assay.plate.sample=="Sample X1",
                                             assay_plate.sample=="Sample X10")

        Parameters
        ----------
        pos_rfu: xr.DataArray
            Positive control samples.
        neg_rfu: optional xr.DataArray
            Negative control samples. Assumed to be zero if not provided.
        pos_conc, neg_conc: optional xr.DataArray
            Concentration vectors of the positive and negative controls
        method: 'direct' (default) or 'relaxation'
        error: optional function mapping float on float values # FIXME: not supported
            Error model used for fitting when using the 'relaxation' method 

        Returns
        -------
        Two functions from_rfu and to_rfu with signatures

        fromRFU(rfu: xr.DataArray) -> xr.DataArray
        toRFU(rfu: xr.DataArray) -> xr.DataArray

        See documentation of the specialized calibration methods for detail.
        """
        if method == 'direct':
            return self.convert_direct(pos_rfu, neg_rfu, pos_conc, neg_conc)
        if method == 'relaxation':
            return self.convert_relaxation(pos_rfu, neg_rfu, pos_conc, neg_conc)
        raise ValueError(f"Unsupported calibration method '{method}'.")

    def convert_direct(
        self, pos_rfu: xr.DataArray, neg_rfu: Optional[xr.DataArray] = None,
        pos_conc: Union[xr.DataArray, float] = 1., neg_conc: Union[xr.DataArray, float] = 0.
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between raw RFU values and concentrations

        Transforms are based on the linear relations

        (rfu-neg_rfu)/(pos_rfu-neg_rfu) = (conc-neg_conc)/(pos_conc-neg_conc)

        where the left hand side is a linear interpolation from negative
        to positive control RFU values and the right hand side expresses
        the reaction coordinate from neg_conc to pos_conc. The transformation
        is performed for each time point independently.

        Parameters
        ----------
        pos_rfu: xarray.DataArray
            RFU values of the positive control            
        neg_rfu: optional xarray.DataArray
            RFU values of the negative control
        pos_conc: xarray.DataArray or float (default 1)
            Concentrations associated with positive control
        neg_conc: xarray.DataArray or float (default 0)
            Concentrations associated with negative control

        Returns
        -------
        Two functions from_rfu and to_rfu with signatures

        fromRFU(rfu: xr.DataArray) -> xr.DataArray
        toRFU(rfu: xr.DataArray) -> xr.DataArray
        """
        neg_rfu = neg_rfu if neg_rfu is not None else 0*pos_rfu
        pos = pos_rfu.mean(dim='content') if len(pos_rfu.dims)>1 else pos_rfu
        neg = neg_rfu.mean(dim='content') if len(neg_rfu.dims)>1 else neg_rfu
        pos_conc = pos_conc if isinstance(pos_conc, xr.DataArray) else xr.DataArray(pos_conc)
        neg_conc = neg_conc if isinstance(neg_conc, xr.DataArray) else xr.DataArray(neg_conc)
        pos_conc, neg_conc = xr.concat([pos_conc, neg_conc], dim='control', fill_value=0.)

        def from_rfu(rfu) :
            return neg_conc + (pos_conc-neg_conc) * (rfu-neg)/(pos-neg)
        def to_rfu(conc):
            return neg + (pos-neg) * (conc-neg_conc)/(pos_conc-neg_conc)
        return from_rfu, to_rfu

    def convert_relaxation(
        self, pos_rfu: xr.DataArray, neg_rfu: Optional[xr.DataArray] = None,
        pos_conc: Union[xr.DataArray, float] = 1., neg_conc: Union[xr.DataArray, float] = 0.
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """FIXME: document"""
        # FIXME: make sure that controls can be any content subset, incl. multiple samples
        neg_rfu = neg_rfu if neg_rfu is not None else 0*pos_rfu
        #pos = pos_rfu.mean(dim='content') if len(pos_rfu.dims)>1 else pos_rfu
        #neg = neg_rfu.mean(dim='content') if len(neg_rfu.dims)>1 else neg_rfu
        pos_conc = pos_conc if isinstance(pos_conc, xr.DataArray) else xr.DataArray(pos_conc)
        neg_conc = neg_conc if isinstance(neg_conc, xr.DataArray) else xr.DataArray(neg_conc)
        pos_conc, neg_conc = xr.concat([pos_conc, neg_conc], dim='control', fill_value=0.)

        def double_relaxation(time, r_1, r_2, rfu_0, rfu_1, rfu_inf):   # pylint: disable=too-many-arguments
            if r_1 == r_2:
                return (rfu_inf + (rfu_0 - rfu_inf)*np.exp(-r_1*time)
                        + r_1*(rfu_1 - rfu_inf)*time*np.exp(-r_1*time))
            return (rfu_inf + (rfu_0 - rfu_inf)*np.exp(-r_1*time)
                    + r_1*(rfu_1 - rfu_inf)/(r_1 - r_2)*(np.exp(-r_2*time)-np.exp(-r_1*time)))

        def error(_):
            return 1 # FIXME: use optional argument value

        controls = xr.concat([pos_rfu, neg_rfu], dim='content') if neg_rfu is not None else pos_rfu

        pos_samples = pos_rfu.sample
        neg_samples = neg_rfu.sample if neg_rfu is not None else []

        def well_model(time, params):
            positive = double_relaxation(time, params['r1'], params['r2'],
                                         params['P0'], params['P1'], params['Pinf'])
            negative = double_relaxation(time, params['r1'], params['r2'],
                                         params['N0'], params['N1'], params['Ninf'])
            return np.array([positive if sample in pos_samples else negative
                             for sample in controls.sample.data])

        def residuals_for(data):
            def residuals(params):
                model = well_model(data.time, params)
                return (data-model)/error(data)
            return residuals

        split = controls.shape[-1]//5
        params = lmfit.create_params(
            r1 = {'value': float(10/controls.time[split]) , 'min': 0., 'vary': True},
            r2 = {'value': float(1/controls.time[-1]), 'min': 0., 'vary': True},
            N0 = {'value': float(controls[controls.sample.isin(neg_samples)][0, 0]), 'min': 0.},
            N1 = {'value': float(controls[controls.sample.isin(neg_samples)][0, split]), 'min': 0.},
            Ninf = {'value': float(controls[controls.sample.isin(neg_samples)][0, -1]), 'min': 0.},
            P0 = {'value': float(controls[controls.sample.isin(pos_samples)][0, 0]), 'min': 0.},
            P1 = {'value': float(controls[controls.sample.isin(pos_samples)][0, split]), 'min': 0.},
            Pinf = {'value': float(controls[controls.sample.isin(pos_samples)][0, -1]), 'min': 0.},
        )
        fit = lmfit.minimize(residuals_for(controls), params)

        neg_model = double_relaxation(controls.time, fit.params['r1'], fit.params['r2'],
                                      fit.params['N0'], fit.params['N1'], fit.params['Ninf'])
        pos_model = double_relaxation(controls.time, fit.params['r1'], fit.params['r2'],
                                      fit.params['P0'], fit.params['P1'], fit.params['Pinf'])

        return self.convert_direct(pos_model, neg_model, pos_conc, neg_conc)

    def compute_power_variance_model(self) -> Callable[[float], float]:
        """Compute power variance model 

        Fit a linear regression against the log transformed sample variance over
        log transformed means. Based in the regression parameters A,B, return a
        function that calculates $A*Y^B$ for some provided fluorescence Y.
        """
        x = np.log(self.mean()).data.flatten()
        y = np.log(self.std()**2).data.flatten()
        regresult = scipy.stats.linregress(x, y) # FIXME: what if regresult.intercept is negative?
        def power_variance(Y: float) -> float:  # pylint: disable=invalid-name
            """Return power variance var(Y) = A*Y^B"""
            return regresult.intercept * Y**regresult.slope
        return power_variance
