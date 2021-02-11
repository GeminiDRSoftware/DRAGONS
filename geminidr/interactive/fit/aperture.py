import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (BoxAnnotation, Button, CheckboxGroup,
                          ColumnDataSource, CustomJS, Div, LabelSet,
                          NumeralTickFormatter, Select, Slider, Spacer, Span,
                          Spinner, TextInput, Whisker)
from bokeh.plotting import figure

from geminidr.interactive import server
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import (PrimitiveVisualizer,
                                              hamburger_helper)
from gempy.library.tracing import (find_apertures, find_apertures_peaks,
                                   get_limits, pinpoint_peaks)
from gempy.utils import logutils

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
"""


class CustomWidget:
    """Defines a default handler that set the value on the model."""

    def __init__(self, title, model, attr, handler=None, **kwargs):
        self.title = title
        self.attr = attr
        self.model = model
        self._handler = handler
        self.kwargs = kwargs

    @property
    def value(self):
        """The value from the model."""
        return getattr(self.model, self.attr)

    def handler(self, attr, old, new):
        if self._handler is not None:
            self._handler(new)
        else:
            setattr(self.model, self.attr, new)


class SpinnerInputLine(CustomWidget):
    def build(self):
        self.spinner = Spinner(value=self.value, width=64, **self.kwargs)
        self.spinner.on_change("value", self.handler)
        return row([Div(text=self.title, align='center'),
                    Spacer(width_policy='max'),
                    self.spinner])

    def reset(self):
        self.spinner.value = self.value


class TextInputLine(CustomWidget):
    def build(self):
        self.text_input = TextInput(value=self.value, width=256, **self.kwargs)
        self.text_input.on_change("value", self.handler)
        return row([Div(text=self.title, align='center'),
                    Spacer(width_policy='max'),
                    self.text_input])

    def reset(self):
        self.text_input.value = self.value


class TextSlider(CustomWidget):
    def build(self):
        self.in_update = False
        self.spinner = Spinner(value=self.value, width=64,
                               step=self.kwargs.get('step'),
                               low=self.kwargs.get('start'),
                               high=self.kwargs.get('end'))
        self.slider = Slider(start=self.kwargs.get('start'),
                             end=self.kwargs.get('end'),
                             step=self.kwargs.get('step'),
                             value=self.value, title=self.title, width=256)
        self.spinner.on_change("value", self.handler)
        self.slider.on_change("value", self.handler)

        return row([self.slider,
                    Spacer(width_policy='max'),
                    self.spinner])

    def reset(self):
        self.spinner.value = self.value
        self.slider.value = self.value

    def handler(self, attr, old, new):
        if self.in_update:
            # To avoid triggering the handler with both spinner and slider
            return
        self.in_update = True
        self.spinner.value = new
        self.slider.value = new
        super().handler(attr, old, new)
        self.in_update = False


class CheckboxLine(CustomWidget):
    def build(self):
        self.checkbox = CheckboxGroup(labels=[""],
                                      active=[0] if self.value else [],
                                      width=40, width_policy='fixed',
                                      align='center')
        self.checkbox.on_click(self.handler)
        return row([Div(text=self.title, align='center'),
                    Spacer(width_policy='max'),
                    self.checkbox])

    def reset(self):
        self.checkbox.active = [0] if self.value else []

    def handler(self, new):
        super().handler(None, None, new)


class SelectLine(CustomWidget):
    def build(self):
        self.select = Select(value=self.value, options=["peak", "integral"],
                             width=128)
        self.select.on_change("value", self.handler)
        return row([Div(text=self.title, align='center'),
                    Spacer(width_policy='max'),
                    self.select])

    def reset(self):
        self.select.value = self.value


class ApertureModel:
    def __init__(self, aperture_id, location, start, end, parent):
        self.aperture_id = aperture_id
        self.source = ColumnDataSource({
            'id': [aperture_id],
            'location': [location],
            'start': [start],
            'end': [end],
            'label_position': [0],
        })
        self.parent = parent

    def delete(self):
        self.parent.delete_aperture(self.aperture_id)

    def update_values(self, **kwargs):
        data = {field: [(0, value)] for field, value in kwargs.items()}
        self.source.patch(data)
        self.parent.adjust_aperture(self.aperture_id)

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
        self.listeners = list()
        self.ext = ext
        self.profile_shape = self.ext.shape[0]
        self.aperture_models = {}

        # keep the initial parameters
        self._aper_params = aper_params.copy()

        # parameters used to compute the current profile
        self.profile_params = None
        self.profile_source = ColumnDataSource({
            'x': np.arange(self.profile_shape),
            'y': np.zeros(self.profile_shape),
        })

        # initial parameters are set as attributes
        self.reset()

    @property
    def aper_params(self):
        """Return the actual param dict from the instance attributes."""
        return {name: getattr(self, name) for name in self._aper_params.keys()}

    def add_listener(self, listener):
        """Add a listener for update to the apertures."""
        self.listeners.append(listener)

    def reset(self):
        """Reset model to its initial values."""
        for name, value in self._aper_params.items():
            setattr(self, name, value)

    def find_closest(self, x):
        model = min(self.aperture_models.values(),
                    key=lambda m: abs(m.source.data['location'][0] - x))
        return model.source.data['id'][0]

    def find_peak(self, x):
        peaks = pinpoint_peaks(self.profile, self.prof_mask, [x],
                               halfwidth=20, threshold=0)
        if len(peaks) > 0:
            limits = get_limits(np.nan_to_num(self.profile),
                                self.prof_mask,
                                peaks=peaks,
                                threshold=self.threshold,
                                method=self.sizing_method)
            log.stdinfo(f"Found source at {self.direction}: {peaks[0]:.1f}")
            self.add_aperture(peaks[0], *limits[0])

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
            for name in ('min_sky_region', 'percentile', 'section', 'use_snr'):
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
            # otherwise we can redo only the peak detection
            locations, all_limits = find_apertures_peaks(
                self.profile, self.prof_mask, self.max_apertures,
                self.direction, self.threshold, self.sizing_method)

        self.aperture_models.clear()

        for aperture_id, (location, limits) in enumerate(
                zip(locations, all_limits), start=1):

            model = ApertureModel(aperture_id, location, limits[0],
                                  limits[1], self)
            self.aperture_models[aperture_id] = model

            for listener in self.listeners:
                listener.handle_aperture(aperture_id, model)

    def add_aperture(self, location, start, end):
        aperture_id = max(self.aperture_models, default=0) + 1
        model = ApertureModel(aperture_id, location, start, end, self)
        self.aperture_models[aperture_id] = model

        for listener in self.listeners:
            listener.handle_aperture(aperture_id, model)
        return aperture_id

    def adjust_aperture(self, aperture_id):
        """Adjust an existing aperture by ID to a new range."""
        for listener in self.listeners:
            listener.handle_aperture(aperture_id,
                                     self.aperture_models[aperture_id])

    def delete_aperture(self, aperture_id):
        """Delete an aperture by ID."""
        for listener in self.listeners:
            listener.delete_aperture(aperture_id)
        del self.aperture_models[aperture_id]

    def clear_apertures(self):
        """Remove all apertures, calling delete on the listeners for each."""
        for aperture_id in self.aperture_models.keys():
            for listener in self.listeners:
                listener.delete_aperture(aperture_id)
        self.aperture_models.clear()


class AperturePlotView:
    """
    Create a visible glyph-set to show the existance
    of an aperture on the given figure.  This display
    will update as needed in response to panning/zooming.

    Parameters
    ----------
    fig : :class:`~bokeh.plotting.Figure`
        Bokeh Figure to attach to
    aperture_id : int
        ID of the aperture (for displaying)
    start : float
        Start of the x-range for the aperture
    end : float
        End of the x-range for the aperture
    """
    def __init__(self, fig, aperture_id, model):
        self.model = model
        self.fig = fig

        if fig.document:
            fig.document.add_next_tick_callback(self.build_ui)
        else:
            self.build_ui()

    def compute_ymid(self):
        if (self.fig.y_range.start is not None and
                self.fig.y_range.end is not None):
            ymin = self.fig.y_range.start
            ymax = self.fig.y_range.end
            return (ymax - ymin)*.8 + ymin
        else:
            return 0

    def update_id(self, aperture_id):
        self.model.update_values(aperture_id=aperture_id)

    def build_ui(self):
        """
        Build the view in the figure.

        This call creates the UI elements for this aperture in the
        parent figure.  It also wires up event listeners to adjust
        the displayed glyphs as needed when the view changes.

        Parameters
        ----------
        fig : :class:`~bokeh.plotting.Figure`
            bokeh figure to attach glyphs to

        """
        fig = self.fig
        source = self.model.source

        self.box = BoxAnnotation(left=source.data['start'][0],
                                 right=source.data['end'][0],
                                 fill_alpha=0.1,
                                 fill_color='green')
        fig.add_layout(self.box)

        self.label = LabelSet(source=source, x="location", y="label_position",
                              y_offset=2, text="id")
        fig.add_layout(self.label)

        self.whisker = Whisker(source=source, base="label_position",
                               lower="start", upper="end", dimension='width',
                               line_color="purple")
        fig.add_layout(self.whisker)

        self.location = Span(location=source.data['location'][0],
                             dimension='height', line_color='green',
                             line_dash='dashed', line_width=1)
        fig.add_layout(self.location)

        fig.y_range.on_change(
            'start', lambda attr, old, new: self.update_viewport())
        fig.y_range.on_change(
            'end', lambda attr, old, new: self.update_viewport())

        # convince the aperture lines to update on zoom
        fig.y_range.js_on_change(
            'end', CustomJS(args=dict(plot=fig),
                            code="plot.properties.renderers.change.emit()"))

        self.update_viewport()

    def update_viewport(self):
        """
        Update the view in the figure whenever we detect a change in the
        display area of the view.  By redrawing, we ensure the lines and axis
        label are in view, at 80% of the way up the visible Y axis.
        """
        ymid = self.compute_ymid()
        if ymid:
            self.model.update_values(label_position=ymid)

    def update(self):
        """
        Alter the coordinate range for this aperture. This will adjust the
        shaded area and the arrows/label for this aperture as displayed on
        the figure.
        """
        source = self.model.source
        self.box.left = source.data['start'][0]
        self.box.right = source.data['end'][0]
        self.location.location = source.data['location'][0]

    def delete(self):
        """Delete this aperture from it's view."""
        # TODO removing causes problems, because bokeh, sigh
        # TODO could create a list of disabled labels/boxes to reuse instead
        # of making new ones (if we have one to recycle)
        self.label.visible = False
        self.box.visible = False
        self.location.visible = False
        self.whisker.visible = False


def avoid_multiple_update(func):
    def wrapper(self, attr, old, new):
        if self.in_update:
            return

        self.in_update = True
        func(self, attr, old, new)
        self.in_update = False
    return wrapper


class ApertureLineView:
    def __init__(self, aperture_id, model):
        """ Create text inputs for the start, location and end of an aperture.

        Parameters
        ----------
        aperture_id : int
            The ID of the aperture
        model : :class:`ApertureModel`
            The model that tracks the apertures and their ranges

        """
        self.model = model
        self.aperture_id = aperture_id

        button = Button(label="Del", width=48)
        button.on_click(self.model.delete)

        source = model.source
        fmt = NumeralTickFormatter(format='0.00')
        self.start_input = Spinner(width=96, low=0, format=fmt,
                                   value=source.data['start'][0])
        self.location_input = Spinner(width=96, low=0, format=fmt,
                                      value=source.data['location'][0])
        self.end_input = Spinner(width=96, low=0, format=fmt,
                                 value=source.data['end'][0])

        self.in_update = False

        self.aperture_name = Div(text=f"# {self.aperture_id}",
                                 align='center', width=24)
        self.start_input.on_change("value", self._start_handler)
        self.location_input.on_change("value", self._location_handler)
        self.end_input.on_change("value", self._end_handler)

        self.component = row([self.aperture_name,
                              self.start_input,
                              self.location_input,
                              self.end_input,
                              button])

    @avoid_multiple_update
    def _start_handler(self, attr, old, new):
        self.model.update_values(start=self.start_input.value)

    @avoid_multiple_update
    def _end_handler(self, attr, old, new):
        self.model.update_values(end=self.end_input.value)

    @avoid_multiple_update
    def _location_handler(self, attr, old, new):
        self.start_input.value += new - old
        self.end_input.value += new - old
        self.model.update_values(location=self.location_input.value,
                                 start=self.start_input.value,
                                 end=self.end_input.value)

    def update_title(self):
        self.aperture_name.text = f"Aperture {self.aperture_id}"

    @avoid_multiple_update
    def update(self, attr, old, new):
        # Because Bokeh checks the handler signatures we need the same
        # argument names for the decorator...
        source = self.model.source
        self.start_input.value = source.data['start'][0]
        self.location_input.value = source.data['location'][0]
        self.end_input.value = source.data['end'][0]

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
        disabled = (self.model.source.data['start'][0] < start or
                    self.model.source.data['end'][0] > end)
        for child in self.component.children:
            child.disabled = disabled


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
    fig : :class:`~bokeh.plotting.Figure`
        bokeh plot for displaying the regions

    """
    def __init__(self, model, fig):
        # The list of AperturePlotView widget (aperture plots)
        self.aperture_plots = {}
        # The list of ApertureLineView widget (text inputs)
        self.aperture_lines = {}

        self.fig = fig

        # The hamburger menu, which needs to have access to the aperture line
        # widgets (inner_controls)
        self.inner_controls = column(max_height=300, height_policy='auto',
                                     width=440, css_classes=['scrollable'])
        self.controls = hamburger_helper("Apertures", self.inner_controls)

        self.inner_controls.children.append(
            row([
                Div(text="", width=48),
                Div(text="<b>Start</b>", width=96, align="center"),
                Div(text="<b>Location</b>", width=96, align="center"),
                Div(text="<b>End</b>", width=96, align="center"),
            ])
        )

        self.model = model
        model.add_listener(self)

        self.view_start = fig.x_range.start
        self.view_end = fig.x_range.end

        # listen here because ap sliders can come and go, and we don't have to
        # convince the figure to release those references since it just ties to
        # this top-level container
        fig.x_range.on_change('start', lambda attr, old, new:
                              self.update_viewport(new, self.view_end))
        fig.x_range.on_change('end', lambda attr, old, new:
                              self.update_viewport(self.view_start, new))

    def update_viewport(self, start, end):
        """Handle a change in the view to enable/disable aperture lines."""
        # Bokeh docs provide no indication of the datatype or orientation of
        # the start/end tuples, so I have left the doc blank for now
        self.view_start = start
        self.view_end = end
        for apline in self.aperture_lines.values():
            apline.update_viewport(start, end)

    def handle_aperture(self, aperture_id, model):
        """Handle an updated or added aperture."""
        if aperture_id in self.aperture_plots:
            self.aperture_plots[aperture_id].update()
            self.aperture_lines[aperture_id].update(None, None, None)
        else:
            ap = AperturePlotView(self.fig, aperture_id, model)
            self.aperture_plots[aperture_id] = ap

            ap = ApertureLineView(aperture_id, model)
            self.aperture_lines[aperture_id] = ap
            self.inner_controls.children.append(ap.component)

    def delete_aperture(self, aperture_id):
        """Remove an aperture by ID. If the ID is not recognized, do nothing.
        """
        if aperture_id in self.aperture_plots:
            self.aperture_plots[aperture_id].delete()
            self.inner_controls.children.remove(
                self.aperture_lines[aperture_id].component)
            del self.aperture_plots[aperture_id]
            del self.aperture_lines[aperture_id]

        # for ap in self.aperture_plots[idx:]:
        #     ap.update_id(ap.aperture_id - 1)

        # for ap in self.aperture_lines[idx:]:
        #     ap.aperture_id -= 1
        #     ap.update_title()


class FindSourceAperturesVisualizer(PrimitiveVisualizer):
    def __init__(self, model, filename_info=''):
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
                         filename_info=filename_info)
        self.model = model
        self.fig = None
        self.help_text = DETAILED_HELP

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

        def _maxaper_handler(new):
            model.max_apertures = int(new) if new is not None else None

        def _use_snr_handler(new):
            model.use_snr = 0 in new

        def _reset_handler(result):
            if result:
                model.reset()
                for widget in (maxaper, minsky, use_snr,
                               threshold, percentile, sizing):
                    widget.reset()
                self.model.recalc_apertures()

        def _find_handler(result):
            if result:
                self.model.recalc_apertures()

        # Profile parameters
        percentile = TextSlider("Percentile (use mean if no value)", model,
                                attr="percentile", start=0, end=100, step=1)
        minsky = SpinnerInputLine("Min sky region", model,
                                  attr="min_sky_region", low=0)
        use_snr = CheckboxLine("Use S/N ratio ?", model, attr="use_snr",
                               handler=_use_snr_handler)
        sections = TextInputLine("Sections", model, attr="section",
                                 placeholder="e.g. 100:900,1500:2000")

        # Peak finding parameters
        maxaper = SpinnerInputLine("Max Apertures (empty means no limit)",
                                   model, attr="max_apertures",
                                   handler=_maxaper_handler, low=0)
        threshold = TextSlider("Threshold", model, attr="threshold",
                               start=0, end=1, step=0.01)
        sizing = SelectLine("Sizing method", model, attr="sizing_method")

        reset_button = Button(label="Reset", button_type='danger',
                              default_size=200)

        self.make_ok_cancel_dialog(reset_button,
                                   'Reset will change all inputs for this tab '
                                   'back to their original values.  Proceed?',
                                   _reset_handler)

        find_button = Button(label="Find apertures", button_type='primary',
                             default_size=200)

        self.make_ok_cancel_dialog(find_button,
                                   'All apertures will be recomputed and '
                                   'changes will be lost. Proceed?',
                                   _find_handler)

        return column(
            Div(text="Parameters to compute the profile:",
                css_classes=['param_section']),
            percentile.build(),
            minsky.build(),
            use_snr.build(),
            sections.build(),
            Div(text="Parameters to find peaks:",
                css_classes=['param_section']),
            maxaper.build(),
            threshold.build(),
            sizing.build(),
            row([reset_button, find_button]),
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

        params = self.parameters_view()

        # Create a blank figure with labels
        self.fig = fig = figure(
            # plot_width=600,
            plot_height=500,
            title='Source Apertures',
            tools="pan,wheel_zoom,box_zoom,reset",
            x_range=(0, self.model.profile_shape)
        )
        fig.height_policy = 'fixed'
        fig.width_policy = 'fit'

        aperture_view = ApertureView(self.model, fig)
        # self.model.add_listener(self)

        fig.step(x='x', y='y', source=self.model.profile_source,
                 color="black", mode="center")

        add_button = Button(label="Add Aperture", button_type='primary')
        add_button.on_click(self.add_aperture)

        helptext = Div(margin=(0, 0, 0, 20), sizing_mode='scale_width')
        controls = column(children=[
            params,
            aperture_view.controls,
            add_button,
        ])

        details_button = Button(label="Show detailed help",
                                button_type='primary')
        details_button.js_on_click(CustomJS(code="openHelpPopup();"))
        self.model.recalc_apertures()

        col = column(children=[fig, helptext], sizing_mode='scale_width')
        toolbar = row(
            children=[
                Div(text=f'<b>Filename:</b> {self.filename_info or ""}<br/>'),
                Spacer(sizing_mode='scale_width'),
                self.submit_button,
                Spacer(sizing_mode='scale_width'),
                details_button,
            ],
        )

        layout = column(toolbar, row(controls, col))
        layout.sizing_mode = 'scale_width'

        Controller(fig, self.model, None, helptext, showing_residuals=False)

        doc.add_root(layout)

    def result(self):
        """
        Get the result of the find.

        Returns
        -------
            list of float, list of tuple :
                list of locations and list of the limits as tuples

        """
        res = (model.result() for model in self.model.aperture_models.values())
        locations, limits = zip(*res)
        return np.array(locations), limits


def interactive_find_source_apertures(ext, **kwargs):
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
    fsav = FindSourceAperturesVisualizer(model, filename_info=ext.filename)
    server.set_visualizer(fsav)
    server.start_server()
    return fsav.result()
