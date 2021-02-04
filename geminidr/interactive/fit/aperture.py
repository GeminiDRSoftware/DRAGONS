import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (BoxAnnotation, Button, CheckboxGroup,
                          ColumnDataSource, CustomJS, Div, LabelSet, Select,
                          Slider, Spacer, Span, Spinner, TextInput, Whisker)
from bokeh.plotting import figure

from geminidr.interactive import server
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import (PrimitiveVisualizer,
                                              hamburger_helper)
from gempy.library.tracing import find_apertures, find_apertures_peaks

__all__ = ["interactive_find_source_apertures", ]


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
        print(f'Calling handler for {self.__class__.__name__}, {self.attr}, '
              f'{attr}, {new}')
        if self._handler is not None:
            self._handler(new)
        else:
            setattr(self.model, self.attr, new)


class TextInputLine(CustomWidget):
    def build(self):
        self.spinner = Spinner(value=self.value, width=64, **self.kwargs)
        self.spinner.on_change("value", self.handler)
        return row([Div(text=self.title, align='end'),
                    Spacer(width_policy='max'),
                    self.spinner])

    def reset(self):
        self.spinner.value = self.value


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
        return row([Div(text=self.title, align='end'),
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
        return row([Div(text=self.title, align='end'),
                    Spacer(width_policy='max'),
                    self.select])

    def reset(self):
        self.select.value = self.value


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
        aperture_id = None
        location = None
        delta = None
        for i, loc in enumerate(self.locations):
            new_delta = abs(loc-x)
            if delta is None or new_delta < delta:
                aperture_id = i+1
                location = loc
                delta = new_delta
        return (aperture_id, location,
                self.all_limits[aperture_id-1][0],
                self.all_limits[aperture_id-1][1])

    def recalc_apertures(self):
        """
        Recalculate the apertures based on the current set of fitter inputs.

        This will redo the aperture detection.  Then it calls to each
        registered listener function and calls it with a list of N locations
        and N limits.

        """
        if self.profile_params is None:
            self.profile_params = self.aper_params.copy()
            recompute_profile = True
        else:
            # Find if parameters that would change the profile have
            # been modified
            recompute_profile = False
            for name in ('min_sky_region', 'percentile', 'sec_regions',
                         'use_snr'):
                if self.profile_params[name] != self.aper_params[name]:
                    recompute_profile = True
                    break
            self.profile_params = self.aper_params.copy()

        if recompute_profile:
            # if any of those parameters changed we must recompute the profile
            locations, self.all_limits, self.profile, self.prof_mask = \
                find_apertures(self.ext, **self.aper_params)
            self.profile_source.patch({'y': [(slice(None), self.profile)]})
        else:
            # otherwise we can redo only the peak detection
            locations, self.all_limits = find_apertures_peaks(
                self.profile, self.prof_mask, self.max_apertures,
                self.direction, self.threshold, self.sizing_method)

        self.locations = list(locations)

        for listener in self.listeners:
            for i, (loc, limits) in enumerate(
                    zip(self.locations, self.all_limits), start=1):
                listener.handle_aperture(i, loc, limits[0], limits[1])

    def add_aperture(self, location, start, end):
        aperture_id = len(self.locations)+1
        self.locations.append(location)
        self.all_limits.append((start, end))

        for listener in self.listeners:
            listener.handle_aperture(aperture_id, location, start, end)
        return aperture_id

    def adjust_aperture(self, aperture_id, location, start, end):
        """Adjust an existing aperture by ID to a new range."""
        if location < start or location > end:
            raise ValueError("Location of aperture must be between start and end")
        self.locations[aperture_id-1] = location
        self.all_limits[aperture_id-1] = (start, end)
        for listener in self.listeners:
            listener.handle_aperture(aperture_id, location, start, end)

    def delete_aperture(self, aperture_id):
        """Delete an aperture by ID."""
        del self.locations[aperture_id-1]
        del self.all_limits[aperture_id-1]

        for listener in self.listeners:
            listener.delete_aperture(aperture_id)

    def clear_apertures(self):
        """Remove all apertures, calling delete on the listeners for each."""
        for iap in range(len(self.locations), 1, -1):
            for listener in self.listeners:
                listener.delete_aperture(iap)
        self.locations = []
        self.all_limits = []


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
    def __init__(self, fig, aperture_id, location, start, end):
        self.aperture_id = aperture_id

        self.source = ColumnDataSource({
            'id': [str(aperture_id)],
            'end': [end],
            'location': [location],
            'start': [start],
            'whisker_y': [0],
        })

        if fig.document:
            fig.document.add_next_tick_callback(
                lambda: self.build_ui(fig))
        else:
            self.build_ui(fig)

    def compute_ymid(self, fig):
        if fig.y_range.start is not None and fig.y_range.end is not None:
            ymin = fig.y_range.start
            ymax = fig.y_range.end
            return (ymax - ymin)*.8 + ymin
        else:
            return 0

    def update_id(self, aperture_id):
        self.aperture_id = aperture_id
        self.source.data['id'] = [aperture_id]

    def build_ui(self, fig):
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
        ymid = self.compute_ymid(fig)

        self.box = BoxAnnotation(left=self.source.data['start'][0],
                                 right=self.source.data['end'][0],
                                 fill_alpha=0.1,
                                 fill_color='green')
        fig.add_layout(self.box)

        self.label = LabelSet(source=self.source, x="location", y=ymid,
                              y_offset=2, text="id")
        fig.add_layout(self.label)

        self.whisker = Whisker(source=self.source, base="whisker_y",
                               lower="start", upper="end", dimension='width',
                               line_color="purple")
        fig.add_layout(self.whisker)

        self.location = Span(location=self.source.data['location'][0],
                             dimension='height', line_color='green',
                             line_dash='dashed', line_width=1)
        fig.add_layout(self.location)

        self.fig = fig

        fig.y_range.on_change(
            'start', lambda attr, old, new: self.update_viewport())
        fig.y_range.on_change(
            'end', lambda attr, old, new: self.update_viewport())

        # convince the aperture lines to update on zoom
        fig.y_range.js_on_change(
            'end', CustomJS(args=dict(plot=fig),
                            code="plot.properties.renderers.change.emit()"))

    def update_viewport(self):
        """
        Update the view in the figure whenever we detect a change in the
        display area of the view.  By redrawing, we ensure the lines and axis
        label are in view, at 80% of the way up the visible Y axis.
        """
        ymid = self.compute_ymid(self.fig)
        if ymid:
            self.label.y = ymid
            self.source.data['whisker_y'] = [ymid]

    def update(self, location, start, end):
        """
        Alter the coordinate range for this aperture. This will adjust the
        shaded area and the arrows/label for this aperture as displayed on
        the figure.

        Parameters
        ----------
        start : float
            new starting x coordinate
        end : float
            new ending x coordinate
        """
        self.box.left = start
        self.box.right = end
        self.source.data['start'] = [start]
        self.source.data['end'] = [end]
        self.source.data['location'] = [location]
        self.location.location = location
        # self.label.x = location+2

    def delete(self):
        """Delete this aperture from it's view."""
        # TODO removing causes problems, because bokeh, sigh
        # TODO could create a list of disabled labels/boxes to reuse instead
        # of making new ones (if we have one to recycle)
        self.label.visible = False
        self.box.visible = False
        self.location.visible = False
        self.whisker.visible = False


class ApertureLineView:
    def __init__(self, model, aperture_id, location, start, end):
        """ Create text inputs for the start, location and end of an aperture.

        Parameters
        ----------
        model : :class:`ApertureModel`
            The model that tracks the apertures and their ranges
        aperture_id : int
            The ID of the aperture
        start, location, end : float
            The start, location and end of the aperture

        """
        self.model = model
        self.aperture_id = aperture_id
        self.location = location
        self.start = start
        self.end = end

        def _del_button_handler():
            self.model.delete_aperture(self.aperture_id)

        button = Button(label="Del", width=48)
        button.on_click(_del_button_handler)

        start_input = Spinner(width=96, value=start)
        location_input = Spinner(width=96, value=location)
        end_input = Spinner(width=96, value=end)

        def _start_handler(attr, old, new):
            if new > self.location:
                print('start cannot be > location')
                start_input.value = self.start
            else:
                self.start = start_input.value
            print(f'adjust {attr} {self.start:.2f} {self.location:.2f} {self.end:.2f}')
            self.model.adjust_aperture(self.aperture_id, self.location,
                                       self.start, self.end)

        def _end_handler(attr, old, new):
            if new < self.location:
                print('end cannot be < location')
                end_input.value = self.end
            else:
                self.end = end_input.value
            print(f'adjust {attr} {self.start:.2f} {self.location:.2f} {self.end:.2f}')
            self.model.adjust_aperture(self.aperture_id, self.location,
                                       self.start, self.end)

        def _location_handler(attr, old, new):
            self.location = location_input.value
            self.start = start_input.value = start_input.value + new - old
            self.end = end_input.value = end_input.value + new - old
            print(f'adjust {attr} {self.start:.2f} {self.location:.2f} {self.end:.2f}')
            self.model.adjust_aperture(self.aperture_id, self.location,
                                       self.start, self.end)

        start_input.on_change("value", _start_handler)
        location_input.on_change("value", _location_handler)
        end_input.on_change("value", _end_handler)

        self.component = row([Div(align='end'),
                              start_input,
                              location_input,
                              end_input,
                              button])
        self.update_title()

    def update_title(self):
        self.component.children[0].text = f"Aperture {self.aperture_id}"

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
        disabled = self.start < start or self.end > end
        for child in self.component.children:
            child.disabled = disabled
            # self.slider.children[0].start = start
            # self.slider.children[0].end = end


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
        self.aperture_plots = []
        # The list of ApertureLineView widget (text inputs)
        self.aperture_lines = []

        self.fig = fig

        # The hamburger menu, which needs to have access to the aperture line
        # widgets (inner_controls)
        self.inner_controls = column()
        self.inner_controls.height_policy = "auto"
        self.controls = hamburger_helper("Apertures", self.inner_controls)

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
        for apline in self.aperture_lines:
            apline.update_viewport(start, end)

    def handle_aperture(self, aperture_id, location, start, end):
        """Handle an updated or added aperture.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to update or create in the view
        location, start, end : float
            Location, start and end of the aperture in x coordinates

        """
        if aperture_id <= len(self.aperture_plots):
            ap = self.aperture_plots[aperture_id-1]
            ap.update(location, start, end)
        else:
            ap = AperturePlotView(self.fig, aperture_id, location, start, end)
            self.aperture_plots.append(ap)

            ap = ApertureLineView(self.model, aperture_id, location, start,
                                  end)
            self.aperture_lines.append(ap)
            self.inner_controls.children.append(ap.component)

    def delete_aperture(self, aperture_id):
        """Remove an aperture by ID. If the ID is not recognized, do nothing.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to remove
        """
        idx = aperture_id-1
        if aperture_id <= len(self.aperture_plots):
            self.aperture_plots[idx].delete()
            self.inner_controls.children.remove(
                self.aperture_lines[idx].component)
            del self.aperture_plots[idx]
            del self.aperture_lines[idx]

        for ap in self.aperture_plots[idx:]:
            ap.update_id(ap.aperture_id - 1)

        for ap in self.aperture_lines[idx:]:
            ap.aperture_id -= 1
            ap.update_title()


def parameters_view(model, recalc_handler):

    def _maxaper_handler(new):
        model.max_apertures = int(new) if new is not None else None

    maxaper = TextInputLine("Max Apertures (empty means no limit)",
                            model, attr="max_apertures",
                            handler=_maxaper_handler, low=0)

    percentile = TextSlider("Percentile (use mean if no value)", model,
                            attr="percentile", start=0, end=100, step=1)

    minsky = TextInputLine("Min sky region", model, attr="min_sky_region",
                           low=0)

    def _use_snr_handler(new):
        model.use_snr = 0 in new

    use_snr = CheckboxLine("Use S/N ratio ?", model, attr="use_snr",
                           handler=_use_snr_handler)

    threshold = TextSlider("Threshold", model, attr="threshold",
                           start=0, end=1, step=0.01)

    sizing = SelectLine("Sizing method", model, attr="sizing_method")

    def _reset_handler():
        model.reset()
        for widget in (maxaper, minsky, use_snr,
                       threshold, percentile, sizing):
            widget.reset()
        recalc_handler()

    reset_button = Button(label="Reset", default_size=200)
    reset_button.on_click(_reset_handler)

    find_button = Button(label="Find apertures", button_type='success',
                         default_size=200)
    find_button.on_click(recalc_handler)

    return column([
        maxaper.build(),
        percentile.build(),
        minsky.build(),
        use_snr.build(),
        threshold.build(),
        sizing.build(),
        row([reset_button, find_button]),
    ])


class FindSourceAperturesVisualizer(PrimitiveVisualizer):
    def __init__(self, model):
        """
        Create a view for finding apertures with the given
        :class:`FindSourceAperturesModel`

        Parameters
        ----------
        model : :class:`FindSourceAperturesModel`
            Model to use for tracking the input parameters and recalculating
            fresh sets as needed
        """
        super().__init__(title='Find Source Apertures')
        self.model = model
        self.details = None
        self.fig = None

    def clear_and_recalc(self):
        """Clear apertures and recalculate a new set."""
        self.model.clear_apertures()
        self.model.recalc_apertures()

    def add_aperture(self):
        """
        Add a new aperture in the middle of the current display area.

        This is used when the user adds an aperture.  We don't know where
        they want it yet so we just place one in the screen center.

        """
        x = (self.fig.x_range.start + self.fig.x_range.end) / 2
        self.model.add_aperture(x, x, x)
        self.update_details()

    def visualize(self, doc):
        """
        Build the visualization in bokeh in the given browser document.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            Bokeh provided document to add visual elements to
        """
        super().visualize(doc)

        current_file = TextInput(title="Current file:",
                                 value=self.model.ext.filename,
                                 disabled=True,
                                 background='white')

        params = parameters_view(self.model, self.clear_and_recalc)

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
        self.model.add_listener(self)

        fig.step(x='x', y='y', source=self.model.profile_source,
                 color="black", mode="center")

        add_button = Button(label="Add Aperture")
        add_button.on_click(self.add_aperture)

        helptext = Div()
        controls = column(children=[
            current_file,
            params,
            aperture_view.controls,
            add_button,
            self.submit_button,
            helptext,
        ])

        self.details = Div(text="")
        self.model.recalc_apertures()
        self.update_details()

        col = column(fig, self.details)
        col.sizing_mode = 'scale_width'
        layout = row(controls, col)

        Controller(fig, self.model, None, helptext)

        doc.add_root(layout)

    def handle_aperture(self, aperture_id, location, start, end):
        self.update_details()

    def delete_aperture(self, aperture_id):
        self.update_details()

    def update_details(self):
        """Update the details text area with the latest aperture data."""
        text = ""
        for i, (loc, limits) in enumerate(
                zip(self.model.locations, self.model.all_limits), start=1):
            text += (f"Aperture #{i}\t <b>Location:</b> {loc:.2f}"
                     f" <b>Lower Limit:</b> {limits[0]:.2f}"
                     f" <b>Upper Limit:</b> {limits[1]:.2f}<br/>")
        self.details.text = text

    def result(self):
        """
        Get the result of the find.

        Returns
        -------
            list of float, list of tuple :
                list of locations and list of the limits as tuples

        """
        return np.array(self.model.locations), self.model.all_limits


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
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)
    server.start_server()
    return fsav.result()
