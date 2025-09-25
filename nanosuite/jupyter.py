"""Usability enhancements for running nanosuite in an interactive environment

This module requires IPython and matplotlib
"""
import base64
from contextlib import ExitStack
from copy import deepcopy
from itertools import cycle
import io
from typing import Any, Callable, Iterable, Sequence
from IPython import display           # type: ignore
import lmfit                          # type: ignore
from matplotlib import colormaps      # type: ignore
from matplotlib.figure import Figure  # type: ignore
import numpy as np
import xarray as xr                   # type: ignore
from . import crn, mars, numerics

####################################################################################################
#
# control interactivity
#
####################################################################################################
interactive: bool|str = True

def ion(mode: bool|str = True) -> ExitStack:
    """Turn interactive mode on

    Allows to temporarily turn on interactive mode. Can be used as a context manager:
    >>> with ns.jupyter.ion():
    >>>     perform_fit()

    Parameters
    ----------
    mode: bool or 'temporary' (default False)
        False:       no interactive output
        True:        permanent interactive output
        'temporary': interactive output deleted after operation
    """
    global interactive  # pylint: disable=global-statement

    if mode not in [False, True, 'temporary']:
        raise ValueError("Mode must be one of True, False or 'temporary'.")

    stack = ExitStack()
    stack.callback(ion if interactive else ioff)  # type: ignore
    interactive = mode
    return stack

def ioff() -> ExitStack:
    """
    Turn interactive mode off

    Allows to temporarily turn off interactive mode. Can be used as a context manager:
    >>> with ns.jupyter.ioff():
    >>>     perform_fit()
    """
    global interactive  # pylint: disable=global-statement
    stack = ExitStack()
    stack.callback(ion if interactive else ioff)  # type: ignore
    interactive = False
    return stack


####################################################################################################
#
# numerics enhancements
#
####################################################################################################
class Trajectory(numerics.Trajectory):
    """Trajectory class that visualizes fit progress"""
    last_result: xr.DataArray = xr.DataArray()

    def eval(self, *args, **opts):
        self.last_result = super().eval(*args, **opts)
        return self.last_result

    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray|dict,
            observe: str|Callable[[xr.DataArray], xr.DataArray]|None = None,
            conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
            error: float|xr.DataArray = 1.,
            *, iter_cb: Callable|None = None,
            **options) -> lmfit.minimizer.MinimizerResult:
        if iter_cb or not interactive:
            return super().fit(data, initial, observe, conversion, error, iter_cb=iter_cb, **options)
        with TrajectoryFitProgress(self, data, observe, conversion, error) as progress:
            return super().fit(data, initial, observe, conversion, error, iter_cb=progress, **options)


class Equilibrium(numerics.Equilibrium):
    """Equilibrium class that visualizes fit progress"""
    last_result: xr.DataArray = xr.DataArray()

    def eval(self, *args, **opts):
        self.last_result = super().eval(*args, **opts)
        return self.last_result

    def fit(self,
            data: xr.DataArray,
            initial: xr.DataArray|dict,
            conversion: Callable[[xr.DataArray], xr.DataArray],
            *, iter_cb: Callable|None = None,
            **options) -> lmfit.minimizer.MinimizerResult:
        if iter_cb or not interactive:
            return super().fit(data, initial, conversion, iter_cb=iter_cb, **options)

        with EquilibriumFitProgress(self, data, conversion) as progress:
            return super().fit(data, initial, conversion, iter_cb=progress, **options)


class TrajectoryFitProgress:
    """Live visualization for Trajectory.fit"""
    def __init__(self,
                 trajectory: Trajectory,
                 data: xr.DataArray,
                 observe: str|Callable[[xr.DataArray], xr.DataArray]|None = None,
                 conversion: Callable[[xr.DataArray], xr.DataArray]|None = None,
                 error: float|xr.DataArray = 1):
        self.trajectory = trajectory
        self.data = data
        self.observe = observe
        self.conversion = conversion
        self.error = error
        self.hdisplay = display.display(display.HTML('<div/>'), display_id=True)

    def __enter__(self):
        # self(self.trajectory.crn.params, 0, [])
        return self

    def __exit__(self, typ, value, traceback):
        if interactive == 'temporary':
            display.clear_output()

    def __call__(self, params: crn.ParameterMap, num_it: int, residuals: Sequence,
                 *args, **kwargs) -> None:
        def gradient(dataset: Sequence|xr.DataArray, cmap: str = 'rainbow') -> Iterable[tuple]:
            size = len(dataset)
            for idx, _ in enumerate(dataset):
                yield colormaps[cmap](idx/size)

        observe = self.observe or self.conversion or self.data.species
        convert = lambda conc: conc.sel(species=observe) if isinstance(observe, str) else observe

        traj = convert(self.trajectory.last_result)

        fig = Figure()
        ax = fig.gca()
        gap = len(traj.time)//15 or 1
        for experiment, model, color in zip(self.data, traj, gradient(self.data)):
            # TODO: it would be nice if this could use Assay colors
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
        self.hdisplay.update(display.HTML(f'''
        <div style="display: flex; flex-wrap: wrap; align-items: flex-start">
            <div style="width: 100%">Iteration: {num_it}</div>
            <img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}">
            <div style="display: inline-block">{params._repr_html_()}</div>
        </div>
        '''))


class EquilibriumFitProgress:
    """Live visualization of Equilibrium.fit"""
    def __init__(self,
                 equilibrium: Equilibrium,
                 data: xr.DataArray,
                 conversion: Callable[[xr.DataArray], xr.DataArray]):
        self.equilibrium = equilibrium
        self.data = data
        self.conversion = conversion
        self.hdisplay = display.display(display.HTML('<div/>'), display_id=True)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        if interactive == 'temporary':
            display.clear_output()

    def __call__(self, params: crn.ParameterMap, num_it: int, residuals: Sequence,
                 *args, **kwargs) -> None:
        original = deepcopy(self.equilibrium.crn.params)
        self.equilibrium.crn.params = params
        eq = self.conversion(self.equilibrium.last_result)
        self.equilibrium.crn.params = original

        fig = Figure()
        ax = fig.gca()

        x = np.arange(0, len(eq.sample) if 'sample' in eq.coords else 1)
        ax.set_xticks(x, eq.sample.data if 'sample' in eq.coords else ['Sample'], rotation=90)
        ax.bar(x-0.2, self.data, label="experiment", width=0.4)
        ax.bar(x+0.2, eq, label="model", width=0.4)
        ax.legend()
        ax.grid(axis='y')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        self.hdisplay.update(display.HTML(f'''
        <div style="display: flex; flex-wrap: wrap; align-items: flex-start">
            <div style="width: 100%">Iteration: {num_it}</div>
            <img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}">
            <div style="display: inline-block">{params._repr_html_()}</div>
        </div>
        '''))


# monkey patches
numerics.Trajectory = Trajectory          # type: ignore
numerics.Equilibrium = Equilibrium        # type: ignore


####################################################################################################
#
# mars enhancements
#
####################################################################################################
class Assay(mars.Assay):
    """Assay class with graphical representation"""
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
        # pylint: disable=protected-access
        assay_img = base64.b64encode(self._repr_png_(**kwargs)).decode()
        setup = self.setup._repr_html_() if self.setup is not None else 'No setup provided'
        sample_map = (self.sample_map._repr_html_()  # type: ignore
                      if self.sample_map is not None
                      else 'No sample map provided')

        return f"""
            <div>
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

                <div class="tab">
                    <button onclick="openTab(event, 'rfu')" style="border: 1px solid grey">RFU</button>
                    <button onclick="openTab(event, 'setup')" style="border: 1px solid grey">Setup</button>
                    <button onclick="openTab(event, 'samplemap')" style="border: 1px solid grey">Sample Map</button>
                </div>

                <div class="rfu tabcontent">
                  <img src="data:image/png;base64,{assay_img}">
                </div>
                <div class="setup tabcontent" style="display: none">
                  {setup}
                </div>
                <div class="samplemap tabcontent" style="display: none; font-size: 0.75rem">
                  {sample_map}
                </div>
            </div>
        """  # type: ignore

    def set_default_palette(self):
        """Set distinct gradients for each sample group"""
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
        """Set distinct gradient for each sample group."""
        if self.sample_map is None:
            raise ValueError("Assay.set_palette requires Assay.sample_map to be defined.")

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
        """Visualize assay as matplotlib figure"""
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
