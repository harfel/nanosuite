"""Usability enhancements for running nanosuite in an interactive environment

This module requires IPython and matplotlib
"""
import base64
from copy import deepcopy
import io
from typing import Any, Callable
from IPython.display import display, HTML  # type: ignore
import lmfit                               # type: ignore
from matplotlib import colormaps           # type: ignore
from matplotlib.figure import Figure       # type: ignore
import xarray as xr                        # type: ignore
from . import crn, mars


def gradient(dataset):
    size = len(dataset)
    for idx, _ in enumerate(dataset):
        yield colormaps['rainbow'](idx/size)


####################################################################################################
#
# crn enhancements
#
####################################################################################################
class FitProgress:
    """Live visualization for CRN.fit"""
    def __init__(self, crn_, data, initial, conversion, error):
        self.crn = crn_
        self.data = data
        self.initial = initial[initial.sample.isin(data.sample)]
        self.conversion = conversion or (lambda conc: conc.sel(species=data.species))
        self.error = error
        self.hdisplay = None

    def __enter__(self):
        self.hdisplay = display(HTML('<div/>'), display_id=True)
        return self

    def __exit__(self, typ, value, traceback):
        pass # self.hdisplay.update(HTML('<div/>'))

    def __call__(self, params, num_it, residuals, *args, **kwargs):
        original = deepcopy(self.crn.params)
        self.crn.params = params
        traj = self.conversion(self.crn.integrate(self.initial, t_eval=self.data.time))
        self.crn.params = original

        fig = Figure()
        ax = fig.gca()
        for experiment, model, color in zip(self.data, traj, gradient(self.data)):
            ax.plot(experiment.time, experiment, '-', c=color)
            ax.plot(model.time, model, '--', c=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Concentration [M]')
        ax.grid()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        self.hdisplay.update(HTML(f'''
        <div>
            <div>Iteration: {num_it}</div>
            <img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}">
            <div style="displaY: inline-block">{params._repr_html_()}</div>
        </div>
        '''))

class CRN(crn.CRN):
    """CRN class that visualizes fit progress"""
    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray,
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: float|xr.DataArray=1.,
            vary_t0: bool|None = None,
            *,
            iter_cb: Callable|None = None,
            **options) -> lmfit.minimizer.MinimizerResult:
        if iter_cb:
            return super().fit(data, initial, conversion, error, vary_t0,
                               iter_cb=iter_cb, **options)
        with FitProgress(self, data, initial, conversion, error) as progress:
            return super().fit(data, initial, conversion, error, vary_t0,
                               iter_cb=progress, **options)

class PartitionedCRN(crn.PartitionedCRN, CRN):
    """PartitionedCRN class that visualizes fit progress"""


# monkey patches
crn.CRN = CRN                        # type: ignore
crn.PartitionedCRN = PartitionedCRN  # type: ignore


####################################################################################################
#
# mars enhancements
#
####################################################################################################
class Assay(mars.Assay):
    """Assay class with graphical representation"""
    def _repr_mimebundle_(self, **kwargs) -> dict[str, Any]:
        return {'png': self._repr_png_(**kwargs)}

    def _repr_png_(self, **kwargs):
        fig = Figure()
        ax = fig.gca()
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("RFU")
        ax.set_title(self.plate.attrs["Test Name"])
        for sample, err, color in zip(self.mean, self.std, gradient(self.mean)):
            ax.fill_between(sample.minutes, sample-err, sample+err, color=color, alpha=0.25)
            ax.plot(sample.minutes, sample, c=color, label=str(sample.sample.values))
        ax.grid()
        ax.legend(ncols=4, loc='upper center', bbox_to_anchor=(0.5, 0),
                  bbox_transform=fig.transFigure)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches="tight")
        buf.seek(0)
        return buf.read()

    def _repr_html_(self, **kwargs) -> str:
        return f'<img src="data:image/png;base64,{base64.b64encode(self._repr_png_(**kwargs)).decode()}">'


# monkey patches
mars.Assay = Assay                   # type: ignore
