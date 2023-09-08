import math
from functools import partial, cmp_to_key

import holoviews as hv
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, CheckboxGroup, ColumnDataSource,
                          Div, LabelSet, NumeralTickFormatter, Select, Slider,
                          Spacer, Span, Spinner, TextInput, Whisker)
from holoviews.streams import Stream

from geminidr.interactive import server
from geminidr.interactive.controls import Controller
from geminidr.interactive.fit.help import PLOT_TOOLS_HELP_SUBTEXT
from geminidr.interactive.interactive import PrimitiveVisualizer, build_text_slider
from geminidr.interactive.interactive_config import interactive_conf
from geminidr.interactive.interactive_config import show_add_aperture_button
from geminidr.interactive.server import interactive_fitter
from gempy.library.tracing import (find_apertures, find_wavelet_peaks,
                                   get_limits, pinpoint_peaks)
from gempy.utils import logutils

hv.extension('bokeh')

renderer = hv.renderer('bokeh')

__all__ = ["interactive_find_source_apertures", ]

log = logutils.get_logger(__name__)


DETAILED_HELP = """

<h2>Help</h2>

<p>Finds sources in 2D spectral images and store them in an APERTURE table for
each extension. Each table will, then, be used in later primitives to perform
aperture extraction.</p>
<p>The primitive operates by first collapsing the 2D spectral image in the
spatial direction to identify sky lines as regions of high pixel-to-pixel
variance, and the regions between the sky lines which consist of at least
`min_sky_region` pixels are selected. These are then collapsed in the
dispersion direction to produce a 1D spatial profile, from which sources are
identified using a peak-finding algorithm.</p>
<p>The widths of the apertures are determined by calculating a threshold level
relative to the peak, or an integrated flux relative to the total between the
minima on either side and determining where a smoothed version of the source
profile reaches this threshold.</p>

<h3>Profile parameters</h3>
<p>Those parameters applies to the computation of the 1D profile.</p>
<dl>
<dt>Percentile</dt>
<dd>
    Percentile to determine signal for each spatial pixel. Uses when
    collapsing along the dispersion direction to obtain a slit profile.
    If None, the mean is used instead.
</dd>
<dt>Min sky region</dt>
<dd>
    Minimum number of contiguous pixels between sky lines for a region
    to be added to the spectrum before collapsing to 1D.
</dd>
<dt>Use S/N ratio</dt>
<dd>
    Convert data to SNR per pixel before collapsing and peak-finding?
</dd>
<dt>Section</dt>
<dd>
    Comma-separated list of colon-separated pixel coordinate pairs
    indicating the region(s) over which the spectral signal should be
    used. The first and last values can be blank, indicating to
    continue to the end of the data
</dd>

<h3>Peak finding parameters</h3>
<p>Those parameters applies to the detection of peaks in the 1D profile.</p>
<dt>Max Apertures</dt>
<dd>
    Maximum number of apertures expected to be found. By default it is
    None so all apertures are returned.
</dd>
<dt>Threshold</dt>
<dd>
    Parameter describing either the height above background (relative
    to peak) or the integral under the spectrum (relative to the
    integral to the next minimum) at which to define the edges of
    the aperture.
</dd>
<dt>Sizing method</dt>
<dd>
    Method for automatic width determination: <i>peak</i> for the height
    relative to peak, or <i>integral</i> for the integrated flux.
</dd>
</dl>
""" + \
PLOT_TOOLS_HELP_SUBTEXT


def avoid_multiple_update(func):
    """Decorator to prevent handlers to update multiple times."""
    def wrapper(self, attr, old, new):
        if self.in_update:
            return

        self.in_update = True
        func(self, attr, old, new)
        self.in_update = False
    return wrapper


class ApertureModel:
    def __init__(self, aperture_id, location, start, end, parent):
        self.source = ColumnDataSource({
            'id': [aperture_id],
            'location': [location],
            'start': [start],
            'end': [end],
        })
        self.parent = parent

    def delete(self):
        self.parent.delete_aperture(self.source.data['id'][0])

    def update_values(self, notify=True, **kwargs):
        data = {field: [(0, value)] for field, value in kwargs.items()}
        self.source.patch(data)
        if notify:
            self.parent.adjust_aperture(self.source.data['id'][0])

    def result(self):
        data = self.source.data
        return data['location'][0], (data['start'][0], data['end'][0])


class FindSourceAperturesModel:
    def __init__(self, ext, **aper_params):
        """
        Create an aperture model with the given initial set of inputs.

        This creates an aperture model that we can use to do the aperture
        fitting.  This model allows us to tweak the various inputs using the
        UI and then recalculate the fit as often as desired.

        """
        self.listeners = []
        self.ext = ext
        self.profile = None
        self.profile_shape = self.ext.shape[0]
        self.aperture_models = {}
        self.selected = None

        # keep the initial parameters
        self._aper_params = aper_params.copy()

        # parameters used to compute the current profile
        self.profile_params = None
        self.profile_source = ColumnDataSource({
            'x': np.arange(self.profile_shape),
            'y': np.zeros(self.profile_shape),
        })

        # target_location is the row from the target coords
        # max_width is the largest distance (in arcsec) from there to the edge of the slit
        # Note: although the ext may have been transposed to ensure that
        # the slit is vertical, the WCS has not been modified
        target_location = ext.wcs.invert(
            ext.central_wavelength(asNanometers=True), ext.target_ra(),
            ext.target_dec())[2 - ext.dispersion_axis()]
        # gWCS will return NaN coords if sent Nones, so assume target is in center
        if np.isnan(target_location):
            target_location = (self.profile_shape - 1) / 2
            self.max_width = target_location
        else:
            self.max_width = max(target_location, self.profile_shape - 1 - target_location)
        self.max_width = int(np.ceil(self.max_width * ext.pixel_scale()))

        # initial parameters are set as attributes
        self.reset()
        del self._aper_params['direction']  # no longer passed to find_apertures()

    @property
    def aper_params(self):
        """Return the actual param dict from the instance attributes."""
        return {name: getattr(self, name) for name in self._aper_params}

    def add_listener(self, listener):
        """Add a listener for update to the apertures."""
        self.listeners.append(listener)

    def call_listeners(self, method, *args, update_view=False, **kwargs):
        for listener in self.listeners:
            func = getattr(listener, method, None)
            if func:
                func(*args, **kwargs)
            if update_view and hasattr(listener, 'update_view'):
                listener.update_view()

    def reset(self):
        """Reset model to its initial values."""
        for name, value in self._aper_params.items():
            setattr(self, name, value)
        if self.max_separation is None:
            self.max_separation = self.max_width

    def find_closest(self, x, x_start, x_end, prefer_selected=True):
        """
        Find the closest aperture.

        If `prefer_selected` is True and we have an existing
        selection, that aperture is returned.  This is the logic
        we want when, for example, deleting.  Then we delete the
        selected or closest aperture.  If the flag is false, we drop
        to more complex logic as follows:

        We want to select the closest visible aperture.  In case of a
        tie with the current selection, we want to select an unselected
        one with the next higher id.  If there are none with higher ids,
        we want to select the lowest id.  If the closest aperture is
        only the currently selected one, we want to return the currently
        selected one.

        Parameters
        ----------
        x : float
            x coordinate of mouse in data space
        x_start : float
            x coordinate of start of visible range
        x_end : float
            x coordinate of end of visible range
        prefer_selected : bool
            If True, return the selected aperture if there is one

        Returns
        -------
        int aperture id of closest or selected
        """
        if prefer_selected and self.selected:
            return self.selected

        def aperture_comparator(a, b):
            if a[0] < b[0]:
                return -1
            if a[0] > b[0]:
                return 1
            if a[1] == self.selected:
                return 1
            if b[1] == self.selected:
                return -1
            if self.selected and a[1] < self.selected:
                if b[1] > self.selected:
                    return 1
            if self.selected and b[1] < self.selected:
                if a[1] > self.selected:
                    return -1
            return 1 if b[1] < a[1] else -1

        model = min([(abs(ap.source.data['location'][0] - x), ap.source.data['id'][0], ap.source.data['location'][0])
                     for ap in self.aperture_models.values() if x_start <= ap.source.data['location'][0] < x_end],
                    key=cmp_to_key(aperture_comparator), default=(None, None, None))
        return model[1]

    def find_peak(self, x):
        # Find local maximum to help pinpoint_peaks
        data = np.ma.array(self.profile, mask=self.prof_mask)
        #initx = np.ma.argmax(data[int(x) - 20:int(x) + 21]) + int(x) - 20
        peaks = find_wavelet_peaks(self.profile, [2], reject_bad=False)[0]
        if peaks.size:
            initx = peaks[np.argmin(abs(peaks - x))]
            if abs(initx - x) <= 20:
                peaks = pinpoint_peaks(data=self.profile, mask=self.prof_mask, peaks=[initx])[0]
                limits = get_limits(np.nan_to_num(self.profile),
                                    self.prof_mask,
                                    peaks=peaks,
                                    threshold=self.threshold,
                                    min_snr=self.min_snr)
                log.stdinfo(f"Found source at {self.direction}: {peaks[0]:.1f}")
                self.add_aperture(peaks[0], *limits[0])

    def update(self, extras):
        for k in extras.keys():
            setattr(self, k, extras[k])

    def recalc_apertures(self):
        """
        Recalculate the apertures based on the current set of fitter inputs.

        This will redo the aperture detection.  Then it calls to each
        registered listener function and calls it with a list of N locations
        and N limits.

        """
        self.clear_apertures()

        if self.profile_params is None:
            self.profile_params = self.aper_params.copy()
            recompute_profile = True
        else:
            # Find if parameters that would change the profile have
            # been modified
            recompute_profile = False
            for name in ('min_sky_region', 'percentile', 'section',
                         'min_snr', 'use_snr'):
                if self.profile_params[name] != self.aper_params[name]:
                    recompute_profile = True
                    break
            self.profile_params = self.aper_params.copy()

        if recompute_profile:
            # if any of those parameters changed we must recompute the profile
            locations, all_limits, self.profile, self.prof_mask = \
                find_apertures(self.ext, **self.aper_params)
            self.profile_source.patch({'y': [(slice(None), self.profile)]})
        else:
            # otherwise pass the existing profile for speed
            locations, all_limits, _, _ = \
                find_apertures(self.ext, **self.aper_params,
                               profile=np.ma.masked_array(self.profile, self.prof_mask))

        self.aperture_models.clear()

        for aperture_id, (location, limits) in enumerate(
                zip(locations, all_limits), start=1):

            model = ApertureModel(aperture_id, location, limits[0],
                                  limits[1], self)
            self.aperture_models[aperture_id] = model
            self.call_listeners('add_aperture', aperture_id, model)

        self.call_listeners('update_view')

    def select_aperture(self, aperture_id):
        self.selected = aperture_id
        self.call_listeners('update_view')

    def add_aperture(self, location, start, end):
        aperture_id = max(self.aperture_models, default=0) + 1
        model = ApertureModel(aperture_id, location, start, end, self)
        self.aperture_models[aperture_id] = model
        self.call_listeners('add_aperture', aperture_id, model,
                            update_view=True)
        return aperture_id

    def adjust_aperture(self, aperture_id):
        """Adjust an existing aperture by ID to a new range."""
        self.call_listeners('update_aperture', aperture_id, update_view=True)

    def delete_aperture(self, aperture_id):
        """Delete an aperture by ID."""
        if self.selected == aperture_id:
            self.selected = None
        self.call_listeners('delete_aperture', aperture_id)
        del self.aperture_models[aperture_id]
        self.call_listeners('update_view')

    def clear_apertures(self):
        """Remove all apertures, calling delete on the listeners for each."""
        self.selected = None
        for aperture_id in self.aperture_models:
            self.call_listeners('delete_aperture', aperture_id)
        self.aperture_models.clear()
        self.call_listeners('update_view')

    def renumber_apertures(self):
        self.selected = None
        new_models = {}
        for new_id, model in enumerate(self.aperture_models.values(), start=1):
            model.update_values(id=new_id, notify=False)
            new_models[new_id] = model
        self.aperture_models.clear()
        self.aperture_models.update(new_models)

        self.call_listeners('renumber_apertures')
        for aperture_id in self.aperture_models:
            self.call_listeners('update_aperture', aperture_id)
        self.call_listeners('update_view')


class AperturePlotView:
    """
    Create a visible glyph-set to show the existance
    of an aperture on the given figure.  This display
    will update as needed in response to panning/zooming.
    """
    def __init__(self, fig, model):
        self.model = model
        self.fig = fig

        if fig.document:
            fig.document.add_next_tick_callback(self.build_ui)
        else:
            self.build_ui()

    def build_ui(self):
        """
        Build the view in the figure.

        This call creates the UI elements for this aperture in the
        parent figure.  It also wires up event listeners to adjust
        the displayed glyphs as needed when the view changes.

        Parameters
        ----------
        fig : :class:`~bokeh.plotting.figure`
            bokeh figure to attach glyphs to

        """
        fig = self.fig
        source = self.model.source

        self.location = Span(location=source.data['location'][0],
                             dimension='height', line_color='green',
                             line_dash='dashed', line_width=1)
        fig.add_layout(self.location)

    def update(self):
        """
        Alter the coordinate range for this aperture. This will adjust the
        shaded area and the arrows/label for this aperture as displayed on
        the figure.
        """
        source = self.model.source
        self.location.location = source.data['location'][0]

    def delete(self):
        """Delete this aperture from it's view."""
        # TODO removing causes problems, because bokeh, sigh
        # TODO could create a list of disabled labels/boxes to reuse instead
        # of making new ones (if we have one to recycle)
        self.location.visible = False


class SelectedApertureLineView:
    def __init__(self, model):
        """ Create text inputs for the start, location and end of the selected aperture.

        Parameters
        ----------
        model : :class:`ApertureModel`
            The model that tracks the apertures and their ranges

        """
        self.apertures_model = model
        model.add_listener(self)

        self.model = None

        options = self._build_select_options()
        self.select = Select(options=options, width=64)
        self.select.on_change('value', self._handle_select_change)

        self.button = Button(label="Del", width=48)

        def _del():
            if self.model:
                self.model.delete()
        self.button.on_click(_del)

        # source = model.source
        fmt = NumeralTickFormatter(format='0.00')
        self.start_input = Spinner(width=80, low=0, format=fmt,
                                   value=0)
        self.location_input = Spinner(width=80, low=0, format=fmt,
                                      value=0)
        self.end_input = Spinner(width=80, low=0, format=fmt,
                                 value=0)

        self.in_update = False

        self.start_input.on_change("value", self._start_handler)
        self.location_input.on_change("value", self._location_handler)
        self.end_input.on_change("value", self._end_handler)

        self.component = column(
            row(
                Div(text="<b>Aperture</b>", width=64),
                Div(text="<b>Lower</b>", width=80),
                Div(text="<b>Location</b>", width=80),
                Div(text="<b>Upper</b>", width=80),
            ),
            row(
                [self.select,
                 self.start_input,
                 self.location_input,
                 self.end_input,
                 self.button]),
        )

    def _build_select_options(self):
        options = list()
        options.append("None")
        aperture_ids = list()
        aperture_ids.extend(self.apertures_model.aperture_models.keys())
        aperture_ids.sort()
        for m in aperture_ids:
            sid = "%s" % m
            options.append(sid)
        return options

    def _add_select_option(self, id):
        sid = "%s" % id
        if sid not in self.select.options:
            self.select.options.append(sid)

    def set_model(self, model):
        self.model = None
        if model:
            sid = "%s" % model.source.data['id'][0]
            self._add_select_option(model.source.data['id'][0])
            self.select.value = sid
            self.location_input.value = model.source.data['location'][0]
            self.start_input.value = self.location_input.value - model.source.data['start'][0]
            self.end_input.value = model.source.data['end'][0] - self.location_input.value
            self.start_input.disabled = False
            self.location_input.disabled = False
            self.end_input.disabled = False
            self.button.disabled = False
            self.model = model
            self.apertures_model.selected = model.source.data['id'][0]
        else:
            self.select.value = "None"
            self.start_input.value = None
            self.start_input.disabled = True
            self.location_input.value = None
            self.location_input.disabled = True
            self.end_input.value = None
            self.end_input.disabled = True
            self.button.disabled = True

    def _handle_select_change(self, attr, old, new):
        if old != new:
            if new == 'None':
                self.apertures_model.select_aperture(None)
            else:
                id = int(new)
                self.apertures_model.select_aperture(id)

    @avoid_multiple_update
    def _start_handler(self, attr, old, new):
        if self.model and old != new:
            self.model.update_values(start=self.location_input.value - self.start_input.value)

    @avoid_multiple_update
    def _end_handler(self, attr, old, new):
        if self.model and old != new:
            self.model.update_values(end=self.location_input.value + self.end_input.value)

    @avoid_multiple_update
    def _location_handler(self, attr, old, new):
        if self.model and old != new:
            self.model.update_values(location=self.location_input.value,
                                     start=self.location_input.value-self.start_input.value,
                                     end=self.location_input.value+self.end_input.value)

    @avoid_multiple_update
    def update(self, attr, old, new):
        # Because Bokeh checks the handler signatures we need the same
        # argument names for the decorator...
        if self.model and old != new:
            source = self.model.source
            self._add_select_option(source.data['id'][0])
            self.start_input.value = source.data['start'][0]
            self.location_input.value = source.data['location'][0]
            self.end_input.value = source.data['end'][0]
            self.aperture_name.text = f"# {source.data['id'][0]}"

    def update_viewport(self, start, end):
        """
        Respond to a viewport update.

        This checks the visible range of the viewport against the current range
        of this aperture.  If the aperture is not fully contained within the
        new visible area, all UI elements are disabled.  If the aperture is in
        range, the start and stop values for the slider are capped to the
        visible range.

        Parameters
        ----------
        start : float
            Visible start of x axis
        end : float
            Visible end of x axis
        """
        if self.model:
            disabled = (self.model.source.data['start'][0] < start or
                        self.model.source.data['end'][0] > end)
            for child in self.component.children:
                child.disabled = disabled

    def update_view(self):
        self.select.options = self._build_select_options()
        if self.apertures_model.selected:
            if self.apertures_model.selected in self.apertures_model.aperture_models:
                ap_model = self.apertures_model.aperture_models[self.apertures_model.selected]
                self.set_model(ap_model)
                return
        self.set_model(None)


class SelectedApertureEditor:
    def __init__(self, model):
        self.model = model
        model.add_listener(self)
        self.div = Div(text="<b>Selected Aperture</b>")
        self.salv = SelectedApertureLineView(model)
        self.component = column(self.div, self.salv.component)

    def select_aperture(self, aperture_id):
        pass

    def update_aperture(self, aperture_id):
        pass

    def delete_aperture(self, aperture_id):
        pass

    def add_aperture(self, aperture_id, model):
        pass

    def update_view(self):
        pass
        if self.model.selected:
            self.div.text = '<b>Selected Aperture</b>'
        else:
            self.div.text = '<b>Selected Aperture</b> - press S with the mouse near an aperture'


class ApertureView:
    """
    UI elements for displaying the current set of apertures.

    This class manages a set of colored regions on a figure to
    show where the defined apertures are, along with a numeric
    ID for each.

    Parameters
    ----------
    model : :class:`ApertureModel`
        Model for tracking the apertures, may be shared across multiple views
    fig : :class:`~bokeh.plotting.figure`
        bokeh plot for displaying the regions

    """
    def __init__(self, model, x_max, y_max):
        # The widgets (AperturePlotView, ApertureLineView) for each aperture
        self.widgets = {}

        self._pending_update_viewport = False

        # save profile max height when managing view.  We want
        # to resize the height if this changes, such as recalculated
        # input data, but not if it hasn't
        self._old_ymax = None

        self.model = model
        model.add_listener(self)

        self.fig = self._make_holoviews_quadmeshed(model, x_max, y_max)

        self.selected_aperture_editor = SelectedApertureEditor(model)
        self.controls = self.selected_aperture_editor.component

        # listen here because ap sliders can come and go, and we don't have to
        # convince the figure to release those references since it just ties to
        # this top-level container
        self.fig.x_range.on_change('start', lambda attr, old, new:
                                   self.update_viewport(start=new))
        self.fig.x_range.on_change('end', lambda attr, old, new:
                                   self.update_viewport(end=new))

        self.annotations_source = ColumnDataSource(data=dict(id=[], text=[], start=[], end=[]))

        self.labels = LabelSet(
            x='end',
            y=380,
            y_units='screen',
            text='text',
            y_offset=2,
            source=self.annotations_source,
        )

        self.fig.add_layout(self.labels)

        self.whisker = Whisker(
            source=self.annotations_source,
            base=380,
            lower="start",
            upper="end",
            dimension='width',
            base_units="screen",
            line_color="purple"
        )

        self.fig.add_layout(self.whisker)

    def update_viewport(self, start=None, end=None):
        """Handle a change in the view to enable/disable aperture lines."""
        if not self._pending_update_viewport:
            self._pending_update_viewport = True
            start = start or self.fig.x_range.start
            end = end or self.fig.x_range.end
            callback = partial(self.update_viewport_callback, start, end)
            curdoc().add_timeout_callback(callback, 100)

    def update_viewport_callback(self, start, end):
        self._pending_update_viewport = False
        self._reload_holoviews()

    def update_aperture(self, aperture_id):
        """Handle an updated or added aperture."""
        plot = self.widgets[aperture_id]
        plot.update()
        for i in range(len(self.annotations_source.data['id'])):
            if self.annotations_source.data['id'][i] == aperture_id:
                self.annotations_source.patch({'end': [(i, plot.model.source.data['end'][0])],
                                               'start': [(i, plot.model.source.data['start'][0])]})

    def add_aperture(self, aperture_id, model):
        plot = AperturePlotView(self.fig, model)
        self.widgets[aperture_id] = plot
        self.annotations_source.stream({'id': [model.source.data['id'][0], ],
                                        'text': [str(model.source.data['id'][0]), ],
                                        'start': [model.source.data['start'][0], ],
                                        'end': [model.source.data['end'][0], ]})

    def delete_aperture(self, aperture_id):
        """Remove an aperture by ID."""
        if aperture_id in self.widgets:
            plot = self.widgets[aperture_id]
            plot.delete()
            del self.widgets[aperture_id]
            newdata = {'id': [], 'text': [], 'start': [], 'end': []}
            for i in range(len(self.annotations_source.data['text'])):
                if self.annotations_source.data['id'][i] != aperture_id:
                    newdata['id'].append(self.annotations_source.data['id'][i])
                    newdata['text'].append(self.annotations_source.data['text'][i])
                    newdata['start'].append(self.annotations_source.data['start'][i])
                    newdata['end'].append(self.annotations_source.data['end'][i])
            self.annotations_source.data = newdata

    def renumber_apertures(self):
        new_widgets = {}
        newlabeldata = {'id': [], 'text': [], 'start': [], 'end': []}
        for plot in self.widgets.values():
            aperture_id = plot.model.source.data['id'][0]
            new_widgets[aperture_id] = plot
            newlabeldata['id'].append(aperture_id)
            newlabeldata['text'].append(str(aperture_id))
            newlabeldata['start'].append(plot.model.source.data['start'])
            newlabeldata['end'].append(plot.model.source.data['end'])
        self.annotations_source.data = newlabeldata
        self.widgets.clear()
        self.widgets.update(new_widgets)

    def update_view(self):
        self._reload_holoviews()
        ymax = np.nanmax(self.model.profile)
        # don't reset plot Y axis if the profile max height hasn't changed
        if ymax and self._old_ymax is None or self._old_ymax != ymax:
            self._old_ymax = ymax
            self.fig.y_range.end = np.nanmax(self.model.profile) * 1.1
            self.fig.y_range.start = np.nanmin(self.model.profile) * 0.9
            self.fig.y_range.reset_end = self.fig.y_range.end
            self.fig.y_range.reset_start = self.fig.y_range.start

    def _prepare_data_for_holoviews(self, aperture_model, x_max, y_max):
        if hasattr(self, 'fig'):
            y = [min(0, self.fig.y_range.start),
                 max(y_max, self.fig.y_range.end)]
        else:
            y = [0, y_max]
        x = [0]
        datarr = [0]
        ranges = [[am.source.data['start'][0], am.source.data['end'][0],
                   True if am.source.data['id'][0] == aperture_model.selected else False]
                  for am in aperture_model.aperture_models.values()]
        ranges.sort(key=lambda x: x[0])

        for range_ in ranges:
            x += range_[0:2]
            datarr += [2 if range_[2] else 1, 0]
        x.append(x_max)
        return x, y, [datarr]

    def _make_holoviews_quadmeshed(self, aperture_model, x_max, y_max):
        da = self._prepare_data_for_holoviews(aperture_model, x_max, y_max)
        cmap = ['#ffffff00', '#d1efd1', '#ff8888']
        xyz = Stream.define('XYZ', data=da)
        self.qm_dmap = hv.DynamicMap(hv.QuadMesh, streams=[xyz()])
        self.qm_dmap.opts(cmap=cmap,
                          height=500,
                          alpha=0.5,
                          responsive=True,
                          show_grid=True,
                          clipping_colors={'NaN': (0, 0, 0, 0)},
                          clim=(0, 2))
        return hv.render(self.qm_dmap)

    def _reload_holoviews(self):
        if self.model.profile is not None:
            x_max = self.model.profile.shape[0]
            y_max = math.ceil(np.nanmax(self.model.profile) * 10)  # arbitrarily large, TBD infinity in QM?
            da = self._prepare_data_for_holoviews(self.model, x_max, y_max)
            self.qm_dmap.event(data=da)


class FindSourceAperturesVisualizer(PrimitiveVisualizer):
    def __init__(self, model, filename_info='', ui_params=None):
        """
        Create a view for finding apertures with the given
        :class:`FindSourceAperturesModel`

        Parameters
        ----------
        model : :class:`FindSourceAperturesModel`
            Model to use for tracking the input parameters and recalculating
            fresh sets as needed
        """
        super().__init__(title='Find Source Apertures',
                         primitive_name='findApertures',
                         filename_info=filename_info,
                         ui_params=ui_params)
        self.model = model
        self.fig = None
        self.help_text = DETAILED_HELP

        # Customize the max_separation behavior away from the defaults.  In particular,
        # we depend on extracting some information from the model which was not readily
        # available in the primitive.
        self.ui_params = ui_params
        self.ui_params.fields["max_separation"].min = 5
        self.ui_params.fields["max_separation"].max = self.model.max_width
        if self.ui_params.fields["max_separation"].default is None:
            self.ui_params.fields["max_separation"].default = self.model.max_separation
        if self.ui_params.values['max_separation'] is None:
            self.ui_params.values['max_separation'] = self.model.max_separation
        if self._reinit_params['max_separation'] is None:
            self._reinit_params['max_separation'] = self.model.max_separation
        # Not necessary since the TextBox is disabled and so the user cannot set to None
        self.ui_params.fields["max_separation"].optional = False

    def add_aperture(self):
        """
        Add a new aperture in the middle of the current display area.

        This is used when the user adds an aperture.  We don't know where
        they want it yet so we just place one in the screen center.

        """
        x = (self.fig.x_range.start + self.fig.x_range.end) / 2
        self.model.add_aperture(x, x, x)

    def parameters_view(self):
        model = self.model

        reset_button = Button(
            label="Reset",
            button_type='warning',
            width=200
        )

        def _reset_handler(result):
            if result:
                reset_button.disabled = True
                def fn():
                    model.reset()
                    self.reset_reinit_panel()
                    self.model.recalc_apertures()
                    reset_button.disabled = False
                self.do_later(fn)

        find_button = Button(
            label="Find apertures",
            button_type='primary',
            width=200
        )

        def _find_handler(result):
            if result:
                find_button.disabled = True
                def fn():
                    self.model.update(self.extras)
                    self.model.recalc_apertures()
                    find_button.disabled = False
                self.do_later(fn)

        widgets = self.make_widgets_from_parameters(self.ui_params, reinit_live=False, slider_width=256,
                                                    add_spacer=True, hide_textbox=['max_separation'])

        self.make_ok_cancel_dialog(reset_button,
                                   'Reset will change all inputs for this tab '
                                   'back to their original values.  Proceed?',
                                   _reset_handler)

        self.make_ok_cancel_dialog(find_button,
                                   'All apertures will be recomputed and '
                                   'changes will be lost. Proceed?',
                                   _find_handler)

        self.make_modal(find_button, 'Recalculating Apertures...')
        self.make_modal(reset_button, 'Recalculating Apertures...')

        return column(
            Div(text="Parameters to compute the profile:",
                css_classes=['param_section']),
            *widgets[0:5],
            Div(text="Parameters to find peaks:",
                css_classes=['param_section']),
            *widgets[5:],
            row([reset_button, find_button]),
            width_policy="min",
        )

    def visualize(self, doc):
        """
        Build the visualization in bokeh in the given browser document.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            Bokeh provided document to add visual elements to
        """
        super().visualize(doc)

        bokeh_data_color = interactive_conf().bokeh_data_color

        params = self.parameters_view()

        ymax = 100  # we will update this when we have a profile
        aperture_view = ApertureView(self.model, self.model.profile_shape, ymax)
        aperture_view.fig.step(x='x', y='y', source=self.model.profile_source,
                               color=bokeh_data_color, mode="center")
        self.fig = aperture_view.fig  # figure now comes from holoviews, need to pull it out here

        # making button configurable so we can add it conditionally for notebooks in future
        renumber_label = "Renumber Apertures"
        clear_label = "Clear Apertures"
        if show_add_aperture_button:
            add_button = Button(
                label="Add",
                button_type='primary',
                width=200
            )

            add_button.on_click(self.add_aperture)
            # need shorter labels
            renumber_label = "Renumber"
            clear_label = "Clear"

        renumber_button = Button(
            label=renumber_label,
            button_type='primary',
            width=200
        )

        renumber_button.on_click(self.model.renumber_apertures)

        clear_button = Button(
            label=clear_label,
            button_type='warning',
            width=200
        )

        def do_clear_apertures():
            def handle_clear(okc):
                if okc:
                    self.model.clear_apertures()
            self.show_ok_cancel('Clear All Apertures?', handle_clear)
        clear_button.on_click(do_clear_apertures)

        helptext = Div(margin=(20, 0, 0, 35), sizing_mode='scale_width')
        controls = column(children=[
            params,
            aperture_view.controls,
            row(clear_button, renumber_button, add_button) if show_add_aperture_button
                else row(clear_button, renumber_button),
        ])

        self.model.recalc_apertures()

        col = column(children=[aperture_view.fig, helptext],
                     sizing_mode='scale_width')

        for btn in (self.submit_button, self.abort_button):
            btn.align = 'end'
            btn.height = 35
            btn.height_policy = "fixed"
            btn.margin = (0, 5, 5, 5)
            btn.width = 212
            btn.width_policy = "fixed"

        toolbar = row(Spacer(width=250),
                      column(self.get_filename_div(), row(self.abort_button, self.submit_button)),
                      Spacer(width=10),
                      align="end", css_classes=['top-row'])

        layout = column(toolbar, row(controls, col))
        layout.sizing_mode = 'scale_width'

        Controller(aperture_view.fig, self.model, None, helptext)

        doc.add_root(layout)

    def submit_button_handler(self):
        """
        Submit button handler.

        The parent version checks for bad/poor fits, but that's not an issue
        here, so we just exit by disabling the submit button, which triggers
        some callbacks.
        """
        self.submit_button.disabled = True

    def result(self):
        """
        Get the result of the find.

        Returns
        -------
        list of float, list of tuple :
            list of locations and list of the limits as tuples

        """
        models = self.model.aperture_models
        res = (models[id_].result() for id_ in sorted(models.keys()))
        try:
            locations, limits = zip(*res)
        except ValueError:
            # There were no results.  Can't check ahead because they are generators
            return [[], []]
        return np.array(locations), limits


def interactive_find_source_apertures(ext, ui_params=None, **kwargs):
    """
    Perform an interactive find of source apertures with the given initial
    parameters.

    This will do all the bokeh initialization and display a UI for
    interactively modifying parameters from their initial values.  The user can
    also interact directly with the found aperutres as desired.  When the user
    hits the `Submit` button, this method will return the results of the find
    to the caller.

    """
    model = FindSourceAperturesModel(ext, **kwargs)
    fsav = FindSourceAperturesVisualizer(model, ui_params=ui_params, filename_info=ext.filename)
    interactive_fitter(fsav)
    return fsav.result()
