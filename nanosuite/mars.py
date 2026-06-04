"""Tools to work with BMG Labtech MARS plate reader data analysis files.

Due to proprietary data formats, MARS files cannot be read directly but have
to be exported to excel. The generated excel files can be loaded with
Assay(rfu_file=path_to_excel_rfu_data).
"""
from functools import cached_property
import re
from typing import cast, Callable, Iterable, Sequence
import warnings
import lmfit         # type: ignore
import numpy as np
import pandas as pd
import scipy         # type: ignore
import xarray as xr  # type: ignore


class Assay:
    """Access to MARS data.

    Assay instances provide access to RFU raw data, mean RFU and standard variance.
    Time points are converted to seconds, independent of the time unit given in the
    imported Excel file.

    Assay instances have the following attributes:

    Attributes
    ----------
    rfu_file: str
        Path of the associated xlsx file (read only)

    all_rfu: 2D xarray DataArray
        Fluorescence values of all wells (whether active or not) and time points

    setup: 2D xarray DataArray
        Initial concentrations in mol/liter for each sample

    FIXME: document sample_map
    """
    rfu_file: str
    all_rfu: xr.DataArray
    setup: xr.DataArray|None = None
    sample_map: pd.DataFrame|None = None


    def __init__(self, rfu_file: str, setup_file: str|None = None, *,
                 setup: xr.DataArray|None = None,
                 sample_map: pd.DataFrame|None = None,
                 groups: dict[str, Sequence[str]|slice]|xr.DataArray|None = None):
        """A plate reader assay

        Parameters
        ----------
        rfu_file: str
            path to MARS RFU data file

        setup_file: str
            path to experimental setup file

        setup: xarray.DataArray
            A 2D DataArray with dimensions sample and species denoting initial
            conditions in mol/liter. Only allowed if setup_file is not provided.

        groups: dict
            mapping of group names to a list or slice of sample names
            e.g. {'System 1': slice("Sample X1", "Sample X5"), "Control": ["Sample X6"]}
            Only allowed it neither setup_file nor setup are provided.
        """
        groups = groups or {}
        self.rfu_file = rfu_file
        self.all_rfu = self.read_rfu(self.rfu_file)

        self.sample_map = sample_map  # may be overwritten by setup_file

        # read assay setup if given
        if setup_file:
            if groups:
                raise ValueError("Arguments setup_file and groups are mutually exclusive")
            if setup:
                raise ValueError("Arguments setup_file and setup are mutually exclusive")
            if sample_map:
                raise ValueError("Arguments setup_file and sample_map are mutually exclusive")
            self.setup = self.read_setup(setup_file)
        elif isinstance(setup, xr.DataArray):
            self.setup = setup
        elif isinstance(setup, dict) or setup is None:
            self.setup = xr.DataArray(
                dims=['sample', 'species'],
                coords={'sample': self.all_rfu.sample.to_series().unique(),
                        'species': list(setup.keys()) if setup else []})
        if isinstance(setup, dict):
            for species, values in setup.items():
                self.setup.loc[..., species] = values

        if 'group' in self.setup.coords:
            # we copy setup groups to rfu groups
            self.annotate(group=self.setup.group)
        if groups:
            self.annotate(group=groups)

    def read_setup(self, setup_file: str) -> xr.DataArray:
        """Read plate setup from excel file

        Parameters
        ----------
        setup_file: str
            path to setup Excel file

        Returns
        -------
        DataArray of initial concentrations in mol/liter

        The excel file needs to contain a worksheet named
        "sample_preparations" with the following format:

        Group | Sample ID | Negative  | Positive   | ... | Gate  | Gate conc (nM) | ...
        ------+-----------+-----------+------------+-----+-------+----------------+-----
        ...   | Sample X1 | Sample X1 | Sample X24 | ... | Gate1 | 100            | ...

        There can be an arbitrary number of Buffer/Media components as well
        as chemical species. Concentrations of the latter can be provided in
        mM, uM, nM, pM or fM.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            df = pd.read_excel(setup_file, sheet_name='assay_settings')
            attrs = dict(df.to_dict('tight')['data'])
            df = pd.read_excel(setup_file, sheet_name='sample_preparations')
            # remove spurious whitespace
            df = df.replace(r'^\s+$', np.nan, regex=True).dropna(axis=0, how='all')

        samples = pd.Index(df['Sample ID'].ffill(), name='sample')
        df.set_index(samples, inplace=True)
        units = [match[1] for s in df.columns[5::2] if (match:=re.match(r'.*\(([munpfa]M)\)', s))]
        factors = {'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9, 'pM': 1e-12, 'fM': 1e-15, 'aM': 1e-18}
        sample_map = df[df.columns[-2*len(units)::2]].set_index(samples)
        sample_map.replace([np.nan], [None], inplace=True)
        sample_map.replace('[^a-zA-Zα-ωΑ-Ω0-9_]', '_', regex=True, inplace=True)
        self.sample_map = sample_map
        concs = df[df.columns[1-2*len(units)::2]].set_index(samples)
        concs = concs.rename(columns=dict(zip(concs.columns, self.sample_map.columns)))
        fac = np.array([factors[u] for u in units])
        try:
            concs *= fac
        except TypeError as exc:
            raise ValueError("Non-numerical concentrations encountered in setup_file") from exc
        for species_class in self.sample_map.columns:
            alternatives = pd.Series(self.sample_map[species_class].unique())
            for species in alternatives:
                if not species:
                    continue
                concs[species] = concs[self.sample_map[species_class]==species][species_class]
        concs.fillna(0., inplace=True)

        # TODO: should controls be optional?
        return xr.DataArray(
            concs,
            {'sample': samples, 'species': concs.columns},
            attrs=attrs,
            name='concentration',
        ).assign_coords(group=('sample', df['Group'].ffill().values),
                        positive=('sample', df['Positive']),
                        negative=('sample', df['Negative']))

    def read_rfu(self, rfu_file: str) -> xr.DataArray:
        """Read fluoresence data from Excel

        Parameters
        ----------
        rfu_file: str
            path to Excel file exported from MARS

        Returns
        -------
        An xarray.DataArray of all raw RFU values with 2 dimensions well and time.
        The well dimension is annotated with a sample coordinate and a coordinate active
        which denotes whether or not the well had been deactivated in MARS. The time
        dimension is annotated with coordinates seconds, minutes and hours.
        DataArray attributes are read from the header section of the excel file, and
        also contain the list of times where injections occurred. 
        """
        def _parse_time(label, times):
            def convert(string):
                match = re.match(r'((?P<h>\d+) h)? *((?P<min>\d+) min)? *((?P<s>\d+) s)?', string)
                return (60*60*float(match.group('h') or 0)
                        + 60*float(match.group('min') or 0)
                        + float(match.group('s') or 0))
            if label == 'Time':
                return np.array(list(map(convert, times)))
            factor = {
                'Time [s]': 1,
                'Time [min]': 60,
                'Time [h]': 60*60,
            }[label]
            return factor * times.values.flatten().astype(float)

        # Create header df and extract data
        df_total = pd.read_excel(rfu_file, header=None)
        content_start = df_total[df_total[0]=='Well'].index[0]
        df_header = df_total.iloc[:content_start, :1]

        # Extract info from headers, into dictionary
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
        attributes['deactivated_cells'] = deactivated

        # Create main df and eval if it needs to be transposed
        df_main = df_total.iloc[content_start:,]

        if df_main.iloc[1,0] == 'Content':
            df_main = df_main.T

        df_main.columns = pd.Index(df_main.iloc[0])
        df_main = df_main[1:]
        df_main.index = pd.Index(np.arange(1, len(df_main) + 1))

        # extract coordinates from dataframe
        times = _parse_time(df_main.iloc[0, 1], df_main.iloc[0, 2:])
        main_array = df_main.iloc[1:, 2:].values

        # The line below blindly assumes that injections happen over a single plate reader cycle.
        inj_indexes = sorted(set(np.where(main_array == 'Inj.')[1]))
        for index in reversed(inj_indexes):
            main_array = np.delete(main_array, index, 1)
            times = np.delete(times, index, 0)
        attributes['injections'] = pd.Index([times[index] for index in inj_indexes], name="time")

        # from dataframe to xarray
        all_rfu = xr.DataArray(main_array,
                               coords={'well': df_main['Well'][1:],
                                   'time': times,
                                   'sample': ('well', df_main['Content'][1:]),
                                   'active': ('well', [well not in deactivated
                                                       for well in df_main['Well'][1:]]),
                                   'seconds': ('time', times),
                                   'minutes': ('time', times/60),
                                   'hours': ('time', times/3600)},
                               dims=('well', 'time'),
                               attrs=attributes,
                               name="RFU").astype(float)

        return all_rfu

    def annotate(self, coords: dict[str, dict[str, Sequence[str]|slice]|xr.DataArray]|None = None,
                 /, **kwargs: dict[str, Sequence[str]|slice]|xr.DataArray) -> None:
        """Annotate samples

        This allows to annotate assay samples with arbitrary information.
        Annotations are added as coordinates to the well coordinate of
        assay.all_rfu and the sample coordinate of assay.setup. New coordinates
        are propagated through into assay.rfu, assay.mean and assay.std.

        >>> assay.annotate(system={'variant_a': slice('Sample X1', 'Sample X10'),
                                   'variant_b': slice('Sample X11', 'Sample X20')})
        >>> assay.setup[assay.setup.system='variant_a']

        Parameters
        ----------
        coords: mapping of coordinate names to sample-associated values.
            Values are either sequences with the same length as either
            assay.setup (to annotate samples) or assay.all_rfu (to annotate
            repeats); or they are mappings where the keys are annotations
            and values either sequences or slices of samples that should
            have the associated annotation.
        """
        if not coords:
            coords = {}
        coords.update(kwargs)
        coord_samples = self.all_rfu.sample.to_series().unique()

        for name, attrib in coords.items():
            coord = xr.DataArray(np.full(len(self.all_rfu), 'Unknown', dtype=object),
                                 self.all_rfu.well.coords)
            if isinstance(attrib, dict):
                for attr, target_samples in attrib.items():
                    if isinstance(target_samples, slice):
                        target_samples = (coord.sel(sample=target_samples)
                                               .sample.to_series().unique())
                    for sample in target_samples:
                        coord[coord.sample == sample] = attr
            elif len(attrib) == len(coord_samples):
                for sample, attr in zip(coord_samples, attrib):
                    coord[coord.sample == sample] = attr
            elif len(attrib) == len(coord):
                coord = attrib
            else:
                raise ValueError(f"attrib must be of length {len(coord)} or {len(coord_samples)}")
            coords[name] = coord

        self.all_rfu = self.all_rfu.assign_coords(coords)
        if self.setup is not None:
            self.setup = self.setup.assign_coords({name: ('sample', coord.groupby('sample')
                                                                         .map(lambda s: s[0])
                                                                         .drop_vars(['well'])
                                                                         .loc[self.setup.sample]
                                                                         .data)
                                                   for name, coord in coords.items()})

        try:
            del self.mean
        except AttributeError:
            pass
        try:
            del self.std
        except AttributeError:
            pass

    def __repr__(self) -> str:
        return f'<Assay "{self.rfu_file}">'

    def _repr_html_(self) -> str:
        return self.rfu._repr_html_() # pylint: disable=protected-access

    def deactivate(self, wells: str|list[str]) -> None:
        """Deactivate a well or list of wells

        Activating and deactivating wells will set a new Assay.all_rfu --
        invalidating any reference to the previous attribute.
        """
        if isinstance(wells, str):
            wells = [wells]
        self.all_rfu.active[self.all_rfu.well.isin(wells)] = False
        self.all_rfu.attrs['deactivated_cells'] = list(
            self.all_rfu.well[~self.all_rfu.active].data)
        for attr in ('rfu', 'mean', 'std'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def activate(self, wells: str|list[str]) -> None:
        """Activate a well or list of wells

        Activating and deactivating wells will set a new Assay.all_rfu --
        invalidating any reference to the previous attribute.
        """
        if isinstance(wells, str):
            wells = [wells]
        self.all_rfu.active[self.all_rfu.well.isin(wells)] = True
        self.all_rfu.attrs['deactivated_cells'] = list(
            self.all_rfu.well[~self.all_rfu.active].data)
        for attr in ('rfu', 'mean', 'std'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    @cached_property
    def mean(self) -> xr.DataArray:
        """Return sample means

        Returns
        -------
        DataArray of average fluorescence of all active wells that belong to
        the same sample. Deactivated wells are not included in the mean
        fluorescence.
        """
        def all_same(values):
            return (values == values[0]).all()

        # Retain coordinates that have homogeneous values in all samples
        homogeneous = {name: coord.groupby('sample').first()
                       for name, coord in self.rfu.coords.items()
                       if 'well' in coord.dims
                       and coord.groupby('sample').map(all_same).all('sample')}

        result = (self.rfu.sel(active=True)
                          .groupby('sample').mean('well')
                          .set_xindex('sample')
                          .assign_coords(homogeneous)
                          .loc[self.setup.sample])
        return result

    @cached_property
    def std(self) -> xr.DataArray:
        """Return sample standard deviation

        Returns
        -------
        DataArray of fluorescence sample standard deviations of all active
        wells that belong to the same sample. Deactivated wells are not included
        in the standard deviation.
        """
        def all_same(values):
            return (values == values[0]).all()

        # Retain coordinates that have homogeneous values in all samples
        homogeneous = {name: coord.groupby('sample').first().loc[self.setup.sample]
                       for name, coord in self.rfu.coords.items()
                       if 'well' in coord.dims
                       and coord.groupby('sample').map(all_same).all('sample')}

        return (self.rfu.sel(active=True)
                    .groupby('sample').std('well', ddof=1)
                    .set_xindex('sample')
                    .assign_coords(homogeneous)
                    .loc[self.setup.sample])

    @cached_property
    def rfu(self):
        """Raw RFU values of all activated wells"""
        return self.all_rfu.sel(active=True)

    @property
    def plate(self):
        """Fluorescence values of all active wells and time points

        *Deprecated since 0.3.0*: use Assay.rfu instead"""
        warnings.warn("Assay.plate is deprecated. Use Assay.rfu instead",
                      DeprecationWarning, stacklevel=2)
        return self.rfu

    @property
    def full_plate(self):
        """Fluorescence values of all wells and time points

        *Deprecated since 0.3.0*: use Assay.all_rfu instead"""
        warnings.warn("Assay.full_plate is deprecated. Use Assay.full_rfu instead",
                      DeprecationWarning, stacklevel=2)
        return self.rfu

    def plate_setup(self, conc: dict[str, float|Iterable]|None = None,
                    **kwargs: float|Iterable) -> xr.DataArray:
        """Define plate setup

        *Deprecated since 0.3.0*: Use Assay.setup

        This method generates a correctly indexed xr.DataArray that
        describes the plate setup in terms of concentrations of chemical
        species for each sample.

        Example
        -------
        To define the plate setup of four samples with 0, 2, 4, and 6 mM
        of species A and 1 mM of species B:

        >>> concentrations = 1e-3 * assay.plate_setup(A=[0, 2, 4, 6], B=1)

        Parameters
        ----------
        conc, kwargs: dict[str, float|Iterable]
            Concentrations of chemical species. If a float is given,
            the species is uniform over all samples. If an iterable
            is given, it must have the same length as there are
            samples in the assay.

        Returns
        -------
        An xarray DataArray with dimensions 'sample' (taken from Assay.plate)
        and 'species', where each sample has species concentrations as
        provided in the method arguments.
        """
        warnings.warn("Assay.plate_setup is deprecated."
                      " Initial concentrations are stored in Assay.setup",
                      DeprecationWarning, stacklevel=2)
        conc = conc if conc else {}
        conc.update(kwargs)
        array = xr.DataArray(
            dims=['sample', 'species'],
            coords={'sample': self.rfu.sample.to_series().unique(),
                    'species': list(conc.keys())})
        for species, values in conc.items():
            array.loc[..., species] = values
        return array

    def convert(
        self, pos_rfu: xr.DataArray, neg_rfu: xr.DataArray|None = None,
        pos_conc: xr.DataArray|float = 1., neg_conc: xr.DataArray|float = 0., /,
        method: str = 'direct', **method_options
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between RFU values and concentrations

        E.g.
        >>> from_rfu, to_rfu = assay.convert(assay.rfu.sample=="Sample X1",
                                             assay_rfu.sample=="Sample X10")

        Parameters
        ----------
        pos_rfu: xr.DataArray
            Positive control samples.
        neg_rfu: optional xr.DataArray
            Negative control samples. Assumed to be zero if not provided.
        pos_conc, neg_conc: optional xr.DataArray
            Concentration vectors of the positive and negative controls
        method: 'direct' (default), 'relaxation' or 'average'

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
        if method == 'average':
            return self.convert_average(pos_rfu, neg_rfu, pos_conc, neg_conc, **method_options)
        raise ValueError(f"Unsupported calibration method '{method}'.")

    def convert_direct(
        self, pos_rfu: xr.DataArray, neg_rfu: xr.DataArray|None = None,
        pos_conc: xr.DataArray|float = 1., neg_conc: xr.DataArray|float = 0.
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
        pos = pos_rfu.mean(axis=0) if len(pos_rfu.dims)>1 else pos_rfu
        neg = neg_rfu.mean(axis=0) if len(neg_rfu.dims)>1 else neg_rfu
        pos_conc = pos_conc if isinstance(pos_conc, xr.DataArray) else xr.DataArray(pos_conc)
        neg_conc = neg_conc if isinstance(neg_conc, xr.DataArray) else xr.DataArray(neg_conc)
        pos_conc, neg_conc = xr.concat([pos_conc, neg_conc], dim='content',
                                       fill_value=0., join='outer')

        def from_rfu(rfu) :
            return (neg_conc + (pos_conc-neg_conc) * (rfu-neg)/(pos-neg)
                    ).transpose(*rfu.dims[:-1], *pos_conc.dims, rfu.dims[-1])
        def to_rfu(conc):
            fraction = (conc-neg_conc)/(pos_conc-neg_conc)
            if 'species' in fraction.dims:
                fraction = fraction.mean(dim='species')
            return fraction*(pos-neg) + neg
        return from_rfu, to_rfu

    def convert_relaxation(
        self, pos_rfu: xr.DataArray, neg_rfu: xr.DataArray|None = None,
        pos_conc: xr.DataArray|float = 1., neg_conc: xr.DataArray|float = 0.
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transforms between RFU values and concentrations

        Same as Assay.convert_direct, but the conversion is based on a
        double exponential relaxation model that is fitted against
        the controls. Specifically, the model assumes that fluoresence
        quickly equilibrates towards an equilibrium value that itself
        slowly equilibrates over time.
        """
        pos_conc = pos_conc if isinstance(pos_conc, xr.DataArray) else xr.DataArray(pos_conc)
        neg_conc = (neg_conc
                    if isinstance(neg_conc, xr.DataArray)
                    else xr.DataArray(neg_conc or 0*pos_conc))
        pos_conc, neg_conc = xr.concat([pos_conc, neg_conc], dim='control', fill_value=0.)

        def double_relaxation(time, r_1, r_2, rfu_0, rfu_1, rfu_inf):   # pylint: disable=too-many-arguments
            if r_1 == r_2:
                return (rfu_inf + (rfu_0 - rfu_inf)*np.exp(-r_1*time)
                        + r_1*(rfu_1 - rfu_inf)*time*np.exp(-r_1*time))
            return (rfu_inf + (rfu_0 - rfu_inf)*np.exp(-r_1*time)
                    + r_1*(rfu_1 - rfu_inf)/(r_1 - r_2)*(np.exp(-r_2*time)-np.exp(-r_1*time)))

        controls = xr.concat([pos_rfu.mean(dim='content'), neg_rfu.mean(dim='content')],
                             dim='content') if neg_rfu is not None else pos_rfu.mean(dim='content')

        def well_model(time, params):
            positive = double_relaxation(time, params['r1'], params['r2'],
                                         params['P0'], params['P1'], params['Pinf'])
            negative = double_relaxation(time, params['r1'], params['r2'],
                                         params['N0'], params['N1'], params['Ninf'])
            return xr.concat([positive, negative],
                             dim='content') if neg_rfu is not None else positive

        def residuals_for(data):
            def residuals(params):
                model = well_model(data.time, params)
                return data - model
            return residuals

        split = controls.shape[-1]//5
        params = lmfit.create_params(
            r1 = {'value': float(10/controls.time[split]) , 'min': 0., 'vary': True},
            r2 = {'value': float(1/controls.time[-1]), 'min': 0., 'vary': True},
            N0 = {'value': float(neg_rfu.mean(dim='content')[0]) if neg_rfu is not None else 0,
                  'min': 0.},
            N1 = {'value': float(neg_rfu.mean(dim='content')[split]) if neg_rfu is not None else 0,
                  'min': 0.},
            Ninf = {'value': float(neg_rfu.mean(dim='content')[-1]) if neg_rfu is not None else 0,
                    'min': 0.},
            P0 = {'value': float(pos_rfu.mean(dim='content')[0]), 'min': 0.},
            P1 = {'value': float(pos_rfu.mean(dim='content')[split]), 'min': 0.},
            Pinf = {'value': float(pos_rfu.mean(dim='content')[-1]), 'min': 0.},
        )
        fit = lmfit.minimize(residuals_for(controls), params)

        neg_model = double_relaxation(controls.time, fit.params['r1'], fit.params['r2'],
                                      fit.params['N0'], fit.params['N1'], fit.params['Ninf'])
        pos_model = double_relaxation(controls.time, fit.params['r1'], fit.params['r2'],
                                      fit.params['P0'], fit.params['P1'], fit.params['Pinf'])

        return self.convert_direct(pos_model, neg_model, pos_conc, neg_conc)

    def convert_average(
        self, pos_rfu: xr.DataArray, neg_rfu: xr.DataArray|None = None,
        pos_conc: xr.DataArray|float = 1., neg_conc: xr.DataArray|float = 0.,
        transient: float = 0
    ) -> tuple[Callable[[xr.DataArray], xr.DataArray], Callable[[xr.DataArray], xr.DataArray]]:
        """Compute transformations between raw RFU values and concentrations

        TODO: improve documentation

        This method can be used if there are no dedicated controls.
        """
        eq = self.rfu[..., self.rfu.time >= transient].mean(dim='time')
        # ERROR: here I am using pos_rfu wrongly, thinking it is a subset of rfu
        pos_rfu = 0*self.rfu[self.rfu.sample.isin(pos_rfu.sample)] + eq
        if neg_rfu is not None:
            eq = self.rfu[..., self.rfu.time.isin(neg_rfu.time)].mean(dim='time')
            neg_rfu = 0*self.rfu[self.rfu.sample.isin(neg_rfu.sample)] + eq
        return self.convert_direct(pos_rfu, neg_rfu, pos_conc, neg_conc)

    def to_concentrations(self, pos_conc: xr.DataArray|float = 1,
                          neg_conc: xr.DataArray|float = 0, /,
                          method: str = 'direct', **method_options) -> xr.DataArray:
        """Convert RFU to concentrations

        This converts Assay.rfu to concentrations using the controls associated
        in Assay.setup as Assay.setup.positive and Assay.setup.negative. 
        The return value is a linear interpolation between the provided neg_conc
        and pos_conc.

        Parameters
        ----------
        neg_conc, pos_conc: xarray.DataArray or float (optional)
            Either scalar concentration values or DataArrays with
            a coordinate species, which is expected to be a subset
            of Assay.setup.species. If neg_conc is not provided
            it defaults to 0. If pos_conc is not provided it
            defaults to 1.

        method: optional
            Either 'direct' (default) or 'average'

        method_options
            are passed through to the respective method

        Returns
        -------
        An xarray.DataArray with dimensions (sample, species) with
        interpolated concentration values.
        """
        # TODO: allow missing negative controls
        # TODO: could return dataset with mean and std
        if method == 'average':
            raise RuntimeError("Assay.to_concentrations does not yet support method 'average'")

        if self.setup is None:
            raise ValueError('to_concentrations requires Assay.setup')

        pos_controls = self.setup.positive.dropna('sample')
        neg_controls = self.setup.negative.dropna('sample')

        if not pos_controls.sample.equals(neg_controls.sample):
            # TODO: just warn and proceed with overlap
            raise ValueError("All samples must either have no control or two controls.")

        if method == 'relaxation':
            warnings.warn("Direct conversion to concentrations is experimental. "
                          "Use the Assay.convert_relaxation interface.")

        controls = xr.concat([pos_controls, neg_controls],
                             pd.Index(['positive', 'negative'], name='control'))

        rfu = self.mean

        # pos_rfu and neg_rfu are xarrays with the same shape as rfu. for each sample
        # the array denotes the rfu values of the corresponding controls
        # (for direct conversion this is exactly what I want)
        neg_rfu = rfu.loc[controls.sel(control='negative')]
        pos_rfu = rfu.loc[controls.sel(control='positive')]
        from_rfu, _ = self.convert(pos_rfu, neg_rfu, pos_conc, neg_conc,
                                   method=method, **method_options)
        concs = from_rfu(self.mean.sel(sample=controls.sample))
        concs.name = 'contentation'
        if 'control' in concs.coords:
            return concs.drop_vars(['control'])
        return concs

    def compute_power_variance_model(self) -> Callable[[float], float]:
        """Compute power variance model

        Fit a linear regression against the log transformed sample variance over
        log transformed means. Based in the regression parameters A,B, return a
        function that calculates $A*Y^B$ for some provided fluorescence Y.
        """
        x = cast(xr.DataArray, np.log(self.mean)).values.flatten()
        y = cast(xr.DataArray, np.log(self.std**2)).values.flatten()
        regresult = scipy.stats.linregress(x, y)
        def power_variance(Y: float) -> float:  # pylint: disable=invalid-name
            """Return power variance var(Y) = A*Y^B"""
            return np.exp(regresult.intercept) * Y**regresult.slope
        return power_variance
