from abc import ABC, abstractmethod

from bokeh.layouts import row, column
from bokeh.models import Slider, TextInput, ColumnDataSource, BoxAnnotation, Button, CustomJS, Label, Column, Div, Dropdown

from geminidr.interactive import server

from gempy.library.config import FieldValidationError


class PrimitiveVisualizer(ABC):
    def __init__(self, log=None):
        """
        Initialize a visualizer.

        This base class creates a submit button suitable for any subclass
        to use and it also listens for the UI window to close, executing a
        submit if that happens.  The submit button will cause the `bokeh`
        event loop to exit and the code will resume executing in whatever
        top level call you are visualizing from.
        """
        self.log = log
        self.user_satisfied = False

        self.submit_button = Button(label="Submit")
        self.submit_button.on_click(self.submit_button_handler)
        callback = CustomJS(code="""
            window.close();
        """)
        self.submit_button.js_on_click(callback)
        self.doc = None

    def submit_button_handler(self, stuff):
        """
        Handle the submit button by stopping the bokeh server, which
        will resume python execution in the DRAGONS primitive.

        Parameters
        ----------
        stuff
            passed by bokeh, but we do not use it

        Returns
        -------
        none
        """
        self.user_satisfied = True
        server.stop_server()

    def visualize(self, doc):
        """
        Perform the visualization.

        This is called via bkapp by the bokeh server and happens
        when the bokeh server is spun up to interact with the user.

        Subclasses should implement this method with their particular
        UI needs, but also should call super().visualize(doc) to
        listen for session terminations.

        Returns
        -------
        none
        """
        self.doc = doc
        doc.on_session_destroyed(self.submit_button_handler)

    def do_later(self, fn):
        if self.doc is None:
            if self.log is not None:
                self.log.warn("Call to do_later, but no document is set.  Does this PrimitiveVisualizer call "
                              "super().visualize(doc)?")
        else:
            self.doc.add_next_tick_callback(lambda: fn())

    def make_modal(self, widget, message):
        callback = CustomJS(args=dict(source=widget), code="""
            console.log("checking button state");
            if (source.disabled) {
                openModal('%s');
            } else {
                closeModal();
            }
        """ % message)
        widget.js_on_change('disabled', callback)

    def make_widgets_from_config(self, params):
        """Makes appropriate widgets for all the parameters in params,
        using the config to determine the type. Also adds these widgets
        to a dict so they can be accessed from the calling primitive"""
        widgets = []
        for pname, value in self.config.items():
            if pname not in params:
                continue
            field = self.config._fields[pname]
            # Do some inspection of the config to determine what sort of widget we want
            doc = field.doc.split('\n')[0]
            if hasattr(field, 'min'):
                # RangeField => Slider
                start, end = field.min, field.max
                # TODO: Be smarter here!
                if start is None:
                    start = -20
                if end is None:
                    end = 50
                step = start
                widget = build_text_slider(doc, value, step, start, end, obj=self.config, attr=pname)
                self.widgets[pname] = widget.children[0]
            elif hasattr(field, 'allowed'):
                # ChoiceField => drop-down menu
                widget = Dropdown(label=doc, menu=list(self.config.allowed.keys()))
            else:
                # Anything else
                widget = TextInput(label=doc)

            widgets.append(widget)
            # Complex multi-widgets will already have been added
            if pname not in self.widgets:
                self.widgets[pname] = widget

        return widgets


def build_text_slider(title, value, step, min_value, max_value, obj=None, attr=None, handler=None,
                      throttled=False):
    """
    Make a slider widget to use in the bokeh interface.

    This method handles some extra boilerplate logic for inspecting
    our primitive field configurations and determining sensible values
    for minimum, maximum, etc.

    Parameters
    ----------
    title : str
        Title for the slider
    value : int
        Value to initially set
    step : float
        Step size
    min_value : int
        Minimum slider value, or None defaults to min(value,0)
    max_value : int
        Maximum slider value, or None defaults to value*2
    obj : object
        Instance to modify the attribute of when slider changes
    attr : str
        Name of attribute in obj to be set with the new value
    handler : method
        Function to call after setting the attribute
    throttled : bool
        Set to `True` to limit handler calls to when the slider is released (default False)

    Returns
    -------
        :class:`~Row` bokeh Row component with the interface inside
    """
    start = min(value, min_value) if min_value else min(value, 0)
    end = max(value, max_value) if max_value else max(10, value*2)
    slider = Slider(start=start, end=end, value=value, step=step, title=title)
    slider.width = 256

    text_input = TextInput()
    text_input.width = 64
    text_input.value = str(value)
    component = row(slider, text_input)

    slider = slider
    text_input = text_input

    def _float_check(val):
        if isinstance(val, int) or isinstance(val, float):
            return True
        try:
            chk = float(val)
            return True
        except ValueError:
            return False

    def update_slider(attrib, old, new):
        if not _float_check(new):
            if _float_check(old):
                text_input.value = str(old)
            return
        if old != new:
            ival = None
            try:
                ival = int(new)
            except ValueError:
                ival = float(new)
            if ival > slider.end and not max_value:
                slider.end = ival
            if 0 <= ival < slider.start and min_value is None:
                slider.start = ival
            if slider.start <= ival <= slider.end:
                slider.value = ival

    def update_text_input(attrib, old, new):
        if new != old:
            text_input.value = str(new)

    def handle_value(attrib, old, new):
        if obj and attr:
            try:
                value = int(new)
            except ValueError:
                value = float(new)
            try:
                obj.__setattr__(attr, value)
            except FieldValidationError:
                # reset textbox
                text_input.remove_on_change("value", handle_value)
                text_input.value = str(old)
                text_input.on_change("value", handle_value)
            else:
                update_slider(attrib, old, new)
        if handler:
            handler()

    if throttled:
        # Since here the text_input calls handle_value, we don't
        # have to call it from the slider as it will happen as
        # a side-effect of update_text_input
        slider.on_change("value_throttled", update_text_input)
        text_input.on_change("value", handle_value)
    else:
        slider.on_change("value", update_text_input)
        # since slider is listening to value, this next line will cause the slider
        # to call the handle_value method and we don't need to do so explicitly
        text_input.on_change("value", handle_value)
    return component


def connect_figure_extras(fig, aperture_model, band_model):
    # If we have bands or apertures to show, show them
    if band_model:
        bands = GIBandView(fig, band_model)
    if aperture_model:
        aperture_view = GIApertureView(aperture_model, fig)

    # This is a workaround for a bokeh bug.  Without this, things like the background shading used for
    # apertures and bands will not update properly after the figure is visible.
    fig.js_on_change('center', CustomJS(args=dict(plot=fig),
                                        code="plot.properties.renderers.change.emit()"))


class GIBandListener(ABC):
    """
    interface for classes that want to listen for updates to a set of bands.
    """

    @abstractmethod
    def adjust_band(self, band_id, start, stop):
        pass

    @abstractmethod
    def delete_band(self, band_id):
        pass


class GIBandModel(object):
    """
    Model for tracking a set of bands.
    """
    def __init__(self):
        # Right now, the band model is effectively stateless, other
        # than maintaining the set of registered listeners.  That is
        # because the bands are not used for anything, so there is
        # no need to remember where they all are.  This is likely to
        # change in future and that information should likely be
        # kept in here.
        self.band_id = 1
        self.listeners = list()

    def add_listener(self, listener):
        """
        Add a listener to this band model.

        The listener can either be a :class:`GIBandListener` or
        it can be a function,  The function should expect as
        arguments, the `band_id`, and `start`, and `stop` x
        range values.

        Parameters
        ----------
        listener : :class:`GIBandListener` or function

        """
        if not isinstance(listener, GIBandListener):
            raise ValueError("must be a BandListener")
        self.listeners.append(listener)

    def adjust_band(self, band_id, start, stop):
        """
        Adjusts the given band ID to the specified X range.

        The band ID may refer to a brand new band ID as well.
        This method will call into all registered listeners
        with the updated information.

        Parameters
        ----------
        band_id : int
            ID fo the band to modify
        start : float
            Starting coordinate of the x range
        stop : float
            Ending coordinate of the x range

        """
        for listener in self.listeners:
            listener.adjust_band(band_id, start, stop)


class GIBandView(GIBandListener):
    """
    View for the set of bands to show then in a figure.
    """
    def __init__(self, fig, model):
        """
        Create the view for the set of bands managed in the given model
        to display them in a figure.

        Parameters
        ----------
        fig : :class:`GIFigure`
            the figure to display the bands in
        model : :class:`GIBandModel`
            the model for the band information (may be shared by multiple :class:`GIBandView`s)
        """
        self.model = model
        model.add_listener(self)
        self.bands = dict()
        self.fig = fig

    def adjust_band(self, band_id, start, stop):
        """
        Adjust a band by it's ID.

        This may also be a new band, if it is an ID we haven't
        seen before.  This call will create or adjust the glyphs
        in the figure to reflect the new data.

        Parameters
        ----------
        band_id : int
            id of band to create or adjust
        start : float
            start of the x range of the band
        stop : float
            end of the x range of the band
        """
        if band_id in self.bands:
            band = self.bands[band_id]
            band.left = start
            band.right = stop
        else:
            band = BoxAnnotation(left=start, right=stop, fill_alpha=0.1, fill_color='navy')
            self.fig.add_layout(band)
            self.bands[band_id] = band

    def delete_band(self, band_id):
        """
        Delete a band by ID.

        If the view does not recognize the id, this is a no-op.
        Otherwise, all related glyphs are cleaned up from the figure.

        Parameters
        ----------
        band_id : int
            ID of band to remove

        """
        if band_id in self.bands:
            band = self.bands[band_id]
            # TODO remove it


class GIApertureModel(object):
    """
    Model for tracking the Apertures.

    This tracks the apertures and a list of subscribers
    to notify when there are any changes.
    """
    def __init__(self):
        """
        Create the apertures model
        """
        self.aperture_id = 1
        self.listeners = list()
        # spare_ids holds any IDs that were returned to
        # us via a delete, so we can re-use them for
        # new apertures
        self.spare_ids = list()

    def add_listener(self, listener):
        """
        Add a listener for update to the apertures.

        Parameters
        ----------
        listener : :class:`GIApertureListener` or function
            The listener to notify if there are any updates
        """
        self.listeners.append(listener)

    def add_aperture(self, start, end):
        """
        Add a new aperture, using the next available ID

        Parameters
        ----------
        start : float
            x coordinate the aperture starts at
        end : float
            x coordinate the aperture ends at

        Returns
        -------
            int id of the aperture
        """
        aperture_id = self.aperture_id
        self.aperture_id += 1
        self.adjust_aperture(aperture_id, start, end)
        return aperture_id

    def adjust_aperture(self, aperture_id, start, end):
        """
        Adjust an existing aperture by ID to a new range.
        This will alert all subscribed listeners.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to adjust
        start : float
            X coordinate of the new start of range
        end : float
            X coordiante of the new end of range

        """
        for l in self.listeners:
            l.handle_aperture(aperture_id, start, end)

    def delete_aperture(self, aperture_id):
        """
        Delete an aperture by ID.

        This will notify all subscribers of the removal
        of this aperture and return it's ID to the available
        pool.

        Parameters
        ----------
        aperture_id : int
            The ID of the aperture to delete

        Returns
        -------

        """
        for listener in self.listeners:
            listener.delete_aperture(aperture_id)
        self.aperture_id = self.aperture_id-1

    def clear_apertures(self):
        while self.aperture_id > 1:
            self.delete_aperture(self.aperture_id-1)


class GISingleApertureView(object):
    def __init__(self, fig, aperture_id, start, end):
        """
        Create a visible glyph-set to show the existance
        of an aperture on the given figure.  This display
        will update as needed in response to panning/zooming.

        Parameters
        ----------
        gifig : :class:`GIFigure`
            Figure to attach to
        aperture_id : int
            ID of the aperture (for displaying)
        start : float
            Start of the x-range for the aperture
        end : float
            End of the x-range for the aperture
        """
        self.aperture_id = aperture_id
        self.box = None
        self.label = None
        self.left_source = None
        self.left = None
        self.right_source = None
        self.right = None
        self.line_source = None
        self.line = None
        self.fig = None
        if fig.document:
            fig.document.add_next_tick_callback(lambda: self.build_ui(fig, aperture_id, start, end))
        else:
            self.build_ui(fig, aperture_id, start, end)

    def build_ui(self, fig, aperture_id, start, end):
        """
        Build the view in the figure.

        This call creates the UI elements for this aperture in the
        parent figure.  It also wires up event listeners to adjust
        the displayed glyphs as needed when the view changes.

        Parameters
        ----------
        fig : :class:`Figure`
            bokeh figure to attach glyphs to
        aperture_id : int
            ID of this aperture, displayed
        start : float
            Start of x-range of aperture
        end : float
            End of x-range of aperture

        """
        if fig.y_range.start is not None and fig.y_range.end is not None:
            ymin = fig.y_range.start
            ymax = fig.y_range.end
            ymid = (ymax-ymin)*.8+ymin
            ytop = ymid + 0.05*(ymax-ymin)
            ybottom = ymid - 0.05*(ymax-ymin)
        else:
            ymin=0
            ymax=0
            ymid=0
            ytop=0
            ybottom=0
        self.box = BoxAnnotation(left=start, right=end, fill_alpha=0.1, fill_color='green')
        fig.add_layout(self.box)
        self.label = Label(x=(start+end)/2-5, y=ymid, text="%s" % aperture_id)
        fig.add_layout(self.label)
        self.left_source = ColumnDataSource({'x': [start, start], 'y': [ybottom, ytop]})
        self.left = fig.line(x='x', y='y', source=self.left_source, color="purple")
        self.right_source = ColumnDataSource({'x': [end, end], 'y': [ybottom, ytop]})
        self.right = fig.line(x='x', y='y', source=self.right_source, color="purple")
        self.line_source = ColumnDataSource({'x': [start, end], 'y': [ymid, ymid]})
        self.line = fig.line(x='x', y='y', source=self.line_source, color="purple")

        self.fig = fig

        fig.y_range.on_change('start', lambda attr, old, new: self.update_viewport())
        fig.y_range.on_change('end', lambda attr, old, new: self.update_viewport())
        # feels like I need this to convince the aperture lines to update on zoom
        fig.y_range.js_on_change('end', CustomJS(args=dict(plot=fig),
                                                 code="plot.properties.renderers.change.emit()"))

    def update_viewport(self):
        """
        Update the view in the figure.

        This call is made whenever we detect a change in the display
        area of the view.  By redrawing, we ensure the lines and
        axis label are in view, at 80% of the way up the visible
        Y axis.

        """
        if self.fig.y_range.start is not None and self.fig.y_range.end is not None:
            ymin = self.fig.y_range.start
            ymax = self.fig.y_range.end
            ymid = (ymax-ymin)*.8+ymin
            ytop = ymid + 0.05*(ymax-ymin)
            ybottom = ymid - 0.05*(ymax-ymin)
            self.left_source.data = {'x': self.left_source.data['x'], 'y': [ybottom, ytop]}
            self.right_source.data = {'x': self.right_source.data['x'], 'y': [ybottom, ytop]}
            self.line_source.data = {'x':  self.line_source.data['x'], 'y': [ymid, ymid]}
            self.label.y = ymid

    def update(self, start, end):
        """
        Alter the coordinate range for this aperture.

        This will adjust the shaded area and the arrows/label for this aperture
        as displayed on the figure.

        Parameters
        ----------
        start : float
            new starting x coordinate
        end : float
            new ending x coordinate
        """
        self.box.left = start
        self.box.right = end
        self.left_source.data = {'x': [start, start], 'y': self.left_source.data['y']}
        self.right_source.data = {'x': [end, end], 'y': self.right_source.data['y']}
        self.line_source.data = {'x': [start, end], 'y': self.line_source.data['y']}
        self.label.x = (start+end)/2-5

    def delete(self):
        """
        Delete this aperture from it's view.
        """
        self.fig.renderers.remove(self.line)
        self.fig.renderers.remove(self.left)
        self.fig.renderers.remove(self.right)
        # TODO removing causes problems, because bokeh, sigh
        # TODO could create a list of disabled labels/boxes to reuse instead of making new ones
        #  (if we have one to recycle)
        self.label.text = ""
        self.box.fill_alpha = 0.0


class GIApertureSliders(object):
    def __init__(self, view, fig, model, aperture_id, start, end):
        self.view = view
        self.model = model
        self.aperture_id = aperture_id
        self.start = start
        self.end = end

        slider_start = fig.x_range.start
        slider_end = fig.x_range.end

        title = "<h3>Aperture %s</h3>" % aperture_id
        self.label = Div(text=title)
        self.lower_slider = build_text_slider("Start", start, 0.01, slider_start, slider_end,
                                              obj=self, attr="start", handler=self.do_update)
        self.upper_slider = build_text_slider("End", end, 0.01, slider_start, slider_end,
                                              obj=self, attr="end", handler=self.do_update)
        button = Button(label="Delete")
        button.on_click(self.delete_from_model)

        self.component = Column(self.label, self.lower_slider, self.upper_slider, button)

    def delete_from_model(self):
        self.model.delete_aperture(self.aperture_id)

    def update_viewport(self, start, end):
        if self.start < start or self.end > end:
            self.lower_slider.children[0].disabled = True
            self.lower_slider.children[1].disabled = True
        else:
            self.lower_slider.children[0].disabled = False
            self.lower_slider.children[1].disabled = False
            self.lower_slider.children[0].start = start
            self.lower_slider.children[0].end = end
        if self.upper_slider.children[0].value < start or self.upper_slider.children[0].value > end:
            self.upper_slider.children[0].disabled = True
            self.upper_slider.children[1].disabled = True
        else:
            self.upper_slider.children[0].disabled = False
            self.upper_slider.children[1].disabled = False
            self.upper_slider.children[0].start = start
            self.upper_slider.children[0].end = end

    def do_update(self):
        if self.start > self.end:
            self.model.adjust_aperture(self.aperture_id, self.end, self.start)
        else:
            self.model.adjust_aperture(self.aperture_id, self.start, self.end)


class GIApertureView(object):
    """
    UI elements for displaying the current set of apertures.

    This class manages a set of colored bands on a figure to
    show where the defined apertures are, along with a numeric
    ID for each.
    """
    def __init__(self, model, fig):
        """

        Parameters
        ----------
        model : :class:`GIApertureModel`
            Model for tracking the apertures, may be shared across multiple views
        fig : :class:`~Figure`
            bokeh plot for displaying the bands
        """
        self.aps = list()
        self.ap_sliders = list()

        self.fig = fig
        self.controls = column()
        self.model = model
        model.add_listener(self)

        self.view_start = fig.x_range.start
        self.view_end = fig.x_range.end

        # listen here because ap sliders can come and go, and we don't have to
        # convince the figure to release those references since it just ties to
        # this top-level container
        fig.x_range.on_change('start', lambda attr, old, new: self.update_viewport(new, self.view_end))
        fig.x_range.on_change('end', lambda attr, old, new: self.update_viewport(self.view_start, new))

    def update_viewport(self, start, end):
        """
        Handle a change in the view.

        We will adjust the slider ranges and/or disable them.

        Parameters
        ----------
        start
        end
        """
        self.view_start = start
        self.view_end = end
        for ap_slider in self.ap_sliders:
            ap_slider.update_viewport(start, end)

    def handle_aperture(self, aperture_id, start, end):
        """
        Handle an updated or added aperture.

        We either update an existing aperture if we recognize the `aperture_id`
        or we create a new one.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to update or create in the view
        start : float
            Start of the aperture in x coordinates
        end : float
            End of the aperture in x coordinates

        """
        if aperture_id <= len(self.aps):
            ap = self.aps[aperture_id-1]
            ap.update(start, end)
        else:
            ap = GISingleApertureView(self.fig, aperture_id, start, end)
            self.aps.append(ap)
            slider = GIApertureSliders(self, self.fig, self.model, aperture_id, start, end)
            self.ap_sliders.append(slider)
            self.controls.children.append(slider.component)

    def delete_aperture(self, aperture_id):
        """
        Remove an aperture by ID.  If the ID is not recognized, do nothing.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to remove

        Returns
        -------

        """
        if aperture_id <= len(self.aps):
            ap = self.aps[aperture_id-1]
            ap.delete()
            self.controls.children.remove(self.ap_sliders[aperture_id-1].component)
            del self.aps[aperture_id-1]
            del self.ap_sliders[aperture_id-1]
        for ap in self.aps[aperture_id-1:]:
            ap.aperture_id = ap.aperture_id-1
            ap.label.text = "%s" % ap.aperture_id
        for ap_slider in self.ap_sliders[aperture_id-1:]:
            ap_slider.aperture_id = ap_slider.aperture_id-1
            ap_slider.label.text = "<h3>Aperture %s</h3>" % ap_slider.aperture_id
