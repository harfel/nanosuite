"""Usability enhancements for running nanosuite in an interactive environment

This module requires IPython and matplotlib
"""
import base64
from copy import deepcopy
from itertools import cycle
import io
from typing import Any, Callable, Iterable, Sequence
from IPython.display import display, HTML  # type: ignore
import lmfit                               # type: ignore
from matplotlib import colormaps           # type: ignore
from matplotlib.figure import Figure       # type: ignore
import numpy as np
import xarray as xr                        # type: ignore
from . import crn, mars


def gradient(dataset: Sequence[Any], cmap: str = 'rainbow') -> Iterable[tuple]:
    """Color gradient for a given sequence"""
    size = len(dataset)
    for idx, _ in enumerate(dataset):
        yield colormaps[cmap](idx/size)


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
        gap = len(traj.time)//15
        for experiment, model, color in zip(self.data, traj, gradient(self.data)):
            ax.plot(experiment.time, experiment, '-', c=color)
            ax.plot(model.time[::gap], model[::gap], 'o', c=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Concentration [M]')
        ax.set_ylim(None, None)
        ax.grid()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # TODO: improve display
        # for long parameter lists the output is too long.
        # I could use style="overflow-y: auto; height: 30em" to constrain the
        # outer dimension. But for this to work I cannot update (recreate) the
        # display HTML. Instead, I have to update the model of a persistent
        # data view.
        self.hdisplay.update(HTML(f'''
        <div>
            <div>Iteration: {num_it}</div>
            <img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}">
            <div style="display: inline-block">{params._repr_html_()}</div>
        </div>
        '''))

class CRN(crn.CRN):
    """CRN class that visualizes fit progress"""
    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray,
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: float|xr.DataArray=1.,
            *,
            iter_cb: Callable|None = None,
            **options) -> lmfit.minimizer.MinimizerResult:
        if iter_cb:
            return super().fit(data, initial, conversion, error,
                               iter_cb=iter_cb, **options)
        with FitProgress(self, data, initial, conversion, error) as progress:
            return super().fit(data, initial, conversion, error,
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
    __doc__ = mars.Assay.__doc__

    def __init__(self, *args, **opts):
        super().__init__(*args, **opts)
        self.palette = xr.DataArray(np.zeros((len(self.setup), 4)),
                                    {'content': self.setup.content,
                                     'channel': ['R', 'G', 'B', 'A']})
        self.set_default_palette()

    def _repr_mimebundle_(self, **kwargs) -> dict[str, Any]:
        return {'html': self._repr_html_(**kwargs),
                'png': self._repr_png_(**kwargs)}

    def _repr_png_(self, **_):
        fig = self.plot()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches="tight")
        buf.seek(0)
        return buf.read()

    def _repr_html_(self, **kwargs) -> str:
        assay_img = base64.b64encode(self._repr_png_(**kwargs)).decode()
        return f"""
            <script>
                function openTab(evt, id) {{
                  let tab_group = evt.currentTarget.parentNode.parentNode;
                  tab_group.querySelectorAll('.tabcontent').forEach(
                    tab => tab.style.display = 'none'
                  );
                  tab_group.querySelectorAll('.tab button').forEach(
                    link => link.classList.remove('active')
                  );
                  tab_group.querySelector('.'+id).style.display = 'block';
                  evt.currentTarget.classList.add('active');
                }}
            </script>
            
            <div>
                <div class="tab">
                    <button onclick="openTab(event, 'rfu')" style="border: 1px solid grey">RFU</button>
                    <button onclick="openTab(event, 'setup')" style="border: 1px solid grey">Setup</button>
                    <button onclick="openTab(event, 'samplemap')" style="border: 1px solid grey">Sample Map</button>
                </div>
            
                <div id="rfu" class="rfu tabcontent">
                  <img src="data:image/png;base64,{assay_img}">
                </div>
                <div id="setup" class="setup tabcontent" style="display: none">
                  {self.setup._repr_html_() if self.setup is not None else 'No setup provided'}
                </div>
                <div id="samplemap" class="samplemap tabcontent" style="display: none">
                  {self.sample_map._repr_html_() if self.sample_map is not None else 'No sample map provided'}
                </div>
            </div>
        """

    def set_default_palette(self):
        positive = np.unique(self.setup.positive)
        negative = np.unique(self.setup.negative)
        controls = np.concatenate([positive, negative])
        for sample in controls:
            self.palette.loc[self.setup[self.setup.sample==sample].content] = np.array([0, 0, 0, 1])
        #groups = self.setup.groupby('group')
        groups = self.setup[~self.setup.sample.isin(controls)].groupby('group')
        cmaps = [colormaps[name] for name in ('Reds', 'Greens', 'Blues', 'Oranges', 'Purples')]
        for (name, group), gradient in zip(groups, cycle(cmaps)):
            samples = len(group)+len(group)//4
            for idx, sample in enumerate(group, start=len(group)//4):
                self.palette.loc[sample.content, :] = np.array(gradient(idx/samples))

    def set_palette(self, color_by, colormap: str = 'brg', portion: tuple[float, float] = (0,1)):
        start, end = portion
        group_colors = colormaps[colormap]
        groups = self.sample_map.groupby(by=color_by, sort=False)
        for group_idx, (_, group) in enumerate(groups):
            f = start + group_idx/len(groups)*(end-start)
            primary = np.array(group_colors(f))
            base = np.array([1, 1, 1, 1])
            for count, (content, _) in enumerate(group.iterrows(), start=1):
                f = count/len(group)
                color = f*primary + (1-f)*base
                self.palette.loc[{'content': content}] = color

    def plot(self) -> Figure:
        fig = Figure()
        ax = fig.gca()
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("RFU")
        ax.set_title(self.rfu.attrs["Test Name"])
        for sample, err in zip(self.mean, self.std):
            color = self.palette.sel(content=sample.content).data
            ax.fill_between(sample.minutes, sample-err, sample+err, color=color, alpha=0.25)
            ax.plot(sample.minutes, sample, c=color, label=str(sample.sample.values))
        ax.grid()
        ax.legend(ncols=4, loc='upper center', bbox_to_anchor=(0.5, 0),
                  bbox_transform=fig.transFigure)
        return fig


# monkey patches
mars.Assay = Assay                   # type: ignore
