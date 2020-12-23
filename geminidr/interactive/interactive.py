from abc import ABC, abstractmethod
from copy import copy

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Slider, TextInput, ColumnDataSource, BoxAnnotation, Button, CustomJS, Label, Column, Div, \
    Dropdown, RangeSlider, Span, NumeralTickFormatter

from geminidr.interactive import server

from gempy.library.config import FieldValidationError


class PrimitiveVisualizer(ABC):
    def __init__(self, config=None, title=''):
        """
        Initialize a visualizer.

        This base class creates a submit button suitable for any subclass
        to use and it also listens for the UI window to close, executing a
        submit if that happens.  The submit button will cause the `bokeh`
        event loop to exit and the code will resume executing in whatever
        top level call you are visualizing from.
        """
        self.title = title
        self.extras = dict()
        if config is None:
            self.config = None
        else:
            self.config = copy(config)

        self.user_satisfied = False

        self.submit_button = Button(label="Submit", align='center', button_type='primary', width_policy='min')
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

        Parameters
        ----------
        doc : :class:`~bokeh.document.document.Document`
            Bokeh document, this is saved for later in :attr:`~geminidr.interactive.interactive.PrimitiveVisualizer.doc`
        """
        self.doc = doc
        doc.on_session_destroyed(self.submit_button_handler)

        # curdoc().template_variables["primitive_name"] = 'Cheeeeeze'  # self.title

        # callback = CustomJS(code="""
        #     setVersion('2.2.1');
        # """)
        # self.doc.add_next_tick_callback(callback)

    def do_later(self, fn):
        """
        Perform an operation later, on the bokeh event loop.

        This call lets you stage a function to execute within the bokeh event loop.
        This is necessary if you want to interact with the bokeh and you are not
        coming from code that is already executing in that context.  Basically,
        this happens when the code is executing because a key press in the browser
        came in through the tornado server via the `handle_key` URL.

        Parameters
        ----------
        fn : function
            Function to execute in the bokeh loop (should not take required arguments)
        """
        if self.doc is None:
            if self.log is not None:
                self.log.warn("Call to do_later, but no document is set.  Does this PrimitiveVisualizer call "
                              "super().visualize(doc)?")
        else:
            self.doc.add_next_tick_callback(lambda: fn())

    def make_modal(self, widget, message):
        """
        Make a modal dialog that activates whenever the widget is disabled.

        A bit of a hack, but this attaches a modal message that freezes
        the whole UI when a widget is disabled.  This is intended for long-running
        operations.  So, what you do is you set `widget.disabled=True` in your
        code and then use `do_later` to queue a long running bit of work.  When
        that work is finished, it should also do a `widget.disabled=False`.

        The reason the work has to be posted to the bokeh loop via `do_later`
        is to allow this modal dialog to execute first.

        Parameters
        ----------
        widget : :class:`~bokeh.models.widgets.widget.Widget`
            bokeh widget to watch for disable/enable
        message : str
            message to display in the popup modal
        """
        callback = CustomJS(args=dict(source=widget), code="""
            if (source.disabled) {
                openModal('%s');
            } else {
                closeModal();
            }
        """ % message)
        widget.js_on_change('disabled', callback)

    def make_widgets_from_config(self, params, extras, reinit_live):
        """
        Makes appropriate widgets for all the parameters in params,
        using the config to determine the type. Also adds these widgets
        to a dict so they can be accessed from the calling primitive.

        Parameters
        ----------
        params : list of str
            which DRAGONS configuration fields to make a UI for
        extras : dict
            Dictionary of additional field definitions for anything not included in the primitive configuration
        reinit_live : bool
            True if recalcuating points is cheap, in which case we don't need a button and do it on any change

        Returns
        -------
        list : Returns a list of widgets to display in the UI.
        """
        widgets = []
        if self.config is None:
            self.log.warn("No config, unable to make widgets")
            return widgets
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
        for pname, field in extras.items():
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

                def handler(val):
                    self.extras[pname] = val
                    if reinit_live:
                        self.reconstruct_points()
                widget = build_text_slider(doc, field.default, step, start, end, obj=self.extras, attr=pname,
                                           handler=handler, throttled=True)
                self.widgets[pname] = widget.children[0]
                self.extras[pname] = field.default
            else:
                # Anything else
                widget = TextInput(label=doc)
                self.extras[pname] = ''

            widgets.append(widget)
            # Complex multi-widgets will already have been added
            if pname not in self.widgets:
                self.widgets[pname] = widget

        return widgets


def build_text_slider(title, value, step, min_value, max_value, obj=None, attr=None, handler=None,
                      throttled=False):
    """
    Make a slider widget to use in the bokeh interface.

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
        :class:`~bokeh.models.layouts.Row` bokeh Row component with the interface inside
    """
    is_float = True
    if isinstance(value, int):
        is_float = False

    start = min(value, min_value) if min_value else min(value, 0)
    end = max(value, max_value) if max_value else max(10, value*2)

    # trying to convince int-based sliders to behave
    if not is_float:
        fmt = NumeralTickFormatter(format='0,0')
        slider = Slider(start=start, end=end, value=value, step=step, title=title, format=fmt)
    else:
        slider = Slider(start=start, end=end, value=value, step=step, title=title)
    slider.width = 256

    text_input = TextInput()
    text_input.width = 64
    text_input.value = str(value)
    component = row(slider, text_input)

    slider = slider
    text_input = text_input

    def _input_check(val):
        # Check if the value is viable as an int or float, according to our type
        if ((not is_float) and isinstance(val, int)) or (is_float and isinstance(val, float)):
            return True
        try:
            if is_float:
                float(val)
            else:
                int(val)
            return True
        except ValueError:
            return False

    def update_slider(attrib, old, new):
        # Update the slider with the new value from the text input
        if not _input_check(new):
            if _input_check(old):
                text_input.value = str(old)
            return
        if old != new:
            if is_float:
                ival = float(new)
            else:
                ival = int(new)
            if ival > slider.end and not max_value:
                slider.end = ival
            if 0 <= ival < slider.start and min_value is None:
                slider.start = ival
            if slider.start <= ival <= slider.end:
                slider.value = ival

    def update_text_input(attrib, old, new):
        # Update the text input
        if new != old:
            text_input.value = str(new)

    def handle_value(attrib, old, new):
        # Handle a new value and set the registered object/attribute accordingly
        # Also updates the slider and calls the registered handler function, if any
        numeric_value = None
        if is_float:
            numeric_value = float(new)
        else:
            numeric_value = int(new)
        if obj and attr:
            try:
                if not hasattr(obj, attr) and isinstance(obj, dict):
                    obj[attr] = numeric_value
                else:
                    obj.__setattr__(attr, numeric_value)
            except FieldValidationError:
                # reset textbox
                text_input.remove_on_change("value", handle_value)
                text_input.value = str(old)
                text_input.on_change("value", handle_value)
            else:
                update_slider(attrib, old, new)
        if handler:
            if numeric_value is not None:
                handler(numeric_value)
            else:
                handler(new)

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


def build_range_slider(title, location, start, end, step, min_value, max_value, obj=None, location_attr=None,
                       start_attr=None, end_attr=None, handler=None, throttled=False):
    """
    Make a range slider widget to use in the bokeh interface.

    Parameters
    ----------
    title : str
        Title for the slider
    location : int or float
        Value for the location
    start : int or float
        Value to initially set for start
    end : int or float
        Value to initially set for end
    step : int or float
        Step size
    min_value : int or float
        Minimum slider value, or None defaults to min(start,0)
    max_value : int or float
        Maximum slider value, or None defaults to end*2
    obj : object
        Instance to modify the attribute of when slider changes
    start_attr : str
        Name of attribute in obj to be set with the new start value
    end_attr : str
        Name of the attribute on obj to be set with the new end value
    handler : method
        Function to call after setting the attribute
    throttled : bool
        Set to `True` to limit handler calls to when the slider is released (default False)

    Returns
    -------
        :class:`~bokeh.models.layouts.Row` bokeh Row component with the interface inside
    """
    # We track of this entry is working on int values or float.  This affects the
    # behavior and type conversion throughout the rest of the slider logic
    is_float = True
    if isinstance(start, int) and isinstance(end, int):
        is_float = False

    slider_start = min(start, min_value) if min_value else min(start, 0)
    slider_end = max(end, max_value) if max_value else max(10, end*2)
    slider = RangeSlider(start=slider_start, end=slider_end, value=(start, end), step=step, title=title)
    slider.width = 192

    start_text_input = TextInput()
    start_text_input.width = 64
    start_text_input.value = str(start)
    location_text_input = TextInput()
    location_text_input.width = 64
    location_text_input.value = str(location)
    end_text_input = TextInput()
    end_text_input.width = 64
    end_text_input.value = str(end)
    component = row(slider, start_text_input, location_text_input, end_text_input)

    def _input_check(val):
        """
        Check the validity of the input value, or reject

        Parameters
        ----------
        val : float or int

        Returns
        -------
            bool : True of the input is valid, False if not.  This may also be the case if a float is passed
            where int is expected
        """
        if ((not is_float) and isinstance(val[0], int) and isinstance(val[1], int)) \
                or (is_float and isinstance(val[0], float) and isinstance(val[1], float)):
            return True
        try:
            if is_float:
                if float(val[0]) > float(val[1]):
                    return False
            else:
                if int(val[0]) > int(val[1]):
                    return False
            if (slider.start > float(val[0]) > slider.end) or (slider.start > float(val[1]) > slider.end):
                # out of view
                return False
            return True
        except ValueError:
            return False

    def update_slider(attrib, old, new):
        """
        This performs an :meth:`~geminidr.interactive.interactive.build_range_slider._input_check`
        on the new value.  If it passes, it is converted and accepted into the slider.  If it
        is a bad value, the change is rolled back and we use the `old` value.

        Parameters
        ----------
        attrib : ignored
        old : tuple of int or float
            old value pair from the range slider
        new : tuple of int or float or str
            new value pair from the range slider/text fields.  This may be passes as a tuple of str from the text inputs
        """
        # Update the slider with a new (start, end) value
        if not _input_check(new):
            if _input_check(old):
                start_text_input.value = str(old[0])
                end_text_input.value = str(old[1])
            return
        if old != new:
            if is_float:
                start_val = float(new[0])
                end_val = float(new[1])
            else:
                start_val = int(new[0])
                end_val = int(new[1])
            if start_val > end_val:
                start_val, end_val = end_val, start_val
            if end_val > slider.end and not max_value:
                slider.end = end_val
            if 0 <= start_val < slider.start and min_value is None:
                slider.start = start_val
            if slider.start <= start_val <= end_val <= slider.end:
                slider.value = (start_val, end_val)

    def update_text_input(attrib, old, new):
        # Update the text inputs with the new (start, end) value for the slider
        if new != old:
            start_text_input.value = str(new[0])
            end_text_input.value = str(new[1])

    def handle_start_value(attrib, old, new):
        if new == old:
            return
        # called by the start text input.  We pull the end value and delegate to handle_value
        try:
            if slider.start <= float(new) <= slider.end:
                if float(new) > float(location_text_input.value):
                    location_text_input.value = new
                handle_value(attrib, (old, location_text_input.value, end_text_input.value),
                             [new, location_text_input.value, end_text_input.value])
                return
        except ValueError as ve:
            pass
        start_text_input.value = old

    def handle_location_value(attrib, old, new):
        if new == old:
            return
        # called by the location text input.  We pull the end value and delegate to handle_value
        try:
            if slider.start <= float(new) <= slider.end:
                handle_value(attrib, (slider.value[0], old, slider.value[1]),
                             [slider.value[0], new, str(slider.value[1])])
                return
        except ValueError:
            pass
        location_text_input.value = old

    def handle_end_value(attrib, old, new):
        if new == old:
            return
        # called by the end text input.  We pull the start value and delegate to handle_value
        try:
            if slider.start <= float(new) <= slider.end:
                if float(new) < float(location_text_input.value):
                    location_text_input.value = new
                handle_value(attrib, (start_text_input.value, location_text_input.value, old),
                             [start_text_input.value, location_text_input.value, new])
                return
        except ValueError:
            pass
        end_text_input.value = old

    def handle_value(attrib, old, new):
        if new == old:
            return
        # Handle a change in value.  Since this has a value that is
        # (start, end) we always end up working on both values.  This
        # is even though typically the user will only be changing one
        # or the other.
        if obj and start_attr and end_attr:
            if is_float:
                start_numeric_value = float(new[0])
                location_numeric_value = float(new[1])
                end_numeric_value = float(new[2])
            else:
                start_numeric_value = int(new[0])
                location_numeric_value = int(new[1])
                end_numeric_value = int(new[2])
            try:
                if start_numeric_value > end_numeric_value:
                    start_numeric_value, end_numeric_value = end_numeric_value, start_numeric_value
                    new[2], new[0] = new[0], new[2]
                if location_numeric_value > end_numeric_value:
                    location_numeric_value = end_numeric_value
                    location_text_input.remove_on_change("value", handle_location_value)
                    location_text_input.value = str(location_numeric_value)
                    location_text_input.on_change("value", handle_location_value)
                if location_numeric_value < start_numeric_value:
                    location_numeric_value = start_numeric_value
                    location_text_input.remove_on_change("value", handle_location_value)
                    location_text_input.value = str(location_numeric_value)
                    location_text_input.on_change("value", handle_location_value)
                obj.__setattr__(start_attr, start_numeric_value)
                obj.__setattr__(location_attr, location_numeric_value)
                obj.__setattr__(end_attr, end_numeric_value)
            except FieldValidationError:
                # reset textbox
                start_text_input.remove_on_change("value", handle_start_value)
                start_text_input.value = str(old[0])
                start_text_input.on_change("value", handle_start_value)
                end_text_input.remove_on_change("value", handle_end_value)
                end_text_input.value = str(old[2])
                end_text_input.on_change("value", handle_end_value)
                location_text_input.remove_on_change("value", handle_location_value)
                location_text_input.value = str(old[1])
                location_text_input.on_change("value", handle_location_value)
            else:
                update_slider(attrib, (old[0], old[2]), (new[0], new[2]))
        if handler:
            handler()

    if throttled:
        # Since here the text_input calls handle_value, we don't
        # have to call it from the slider as it will happen as
        # a side-effect of update_text_input
        slider.on_change("value_throttled", update_text_input)
    else:
        slider.on_change("value", update_text_input)
        # since slider is listening to value, this next line will cause the slider
        # to call the handle_value method and we don't need to do so explicitly
    start_text_input.on_change("value", handle_start_value)
    location_text_input.on_change("value", handle_location_value)
    end_text_input.on_change("value", handle_end_value)

    return component


def connect_figure_extras(fig, aperture_model, band_model):
    """
    Connect a figure to an aperture and band model for rendering.

    This call will add extra visualizations to the bokeh figure to
    show the bands and apertures in the given models.  Either may
    be passed as None if not relevant.

    This call also does a fix to bokeh to work around a rendering bug.

    Parameters
    ----------
    fig : :class:`~bokeh.plotting.Figure`
        bokeh Figure to add visualizations too
    aperture_model : :class:`~geminidr.interactive.interactive.GIApertureModel`
        Aperture model to add view for
    band_model : :class:`~geminidr.interactive.interactive.GIBandModel`
        Band model to add view for
    """
    # If we have bands or apertures to show, show them
    if band_model:
        bands = GIBandView(fig, band_model)
    if aperture_model:
        aperture_view = GIApertureView(aperture_model, fig)

    # This is a workaround for a bokeh bug.  Without this, things like the background shading used for
    # apertures and bands will not update properly after the figure is visible.
    fig.js_on_change('center', CustomJS(args=dict(plot=fig),
                                        code="plot.properties.renderers.change.emit()"))


_hamburger_order_number = 1


def hamburger_helper(title, widget):
    """
    Create a bokeh layout with a top title and hamburger button
    to show/hide the given widget.

    This will make a wrapper column around whatever you pass in
    and give a control for showing and hiding it.  It is useful
    for potentially larger sets of controls such as a list of
    aperture controls.

    Parameters
    ----------
    title : str
        Text to put in the top area
    widget : :class:`~bokeh.models.layouts.LayoutDOM`
        Component to show/hide with the hamburger action

    Returns
    -------
    :class:`~bokeh.models.layouts.Column` : bokeh column to add into your document
    """
    global _hamburger_order_number
    if widget.css_classes:
        widget.css_classes.append('hamburger_helper_%s' % _hamburger_order_number)
    else:
        widget.css_classes = list('hamburger_helper_%s' % _hamburger_order_number)
    _hamburger_order_number = _hamburger_order_number+1

    # TODO Hamburger icon
    button = Button(label=title, css_classes=['hamburger_helper',])

    top = button

    def burger_action():
        if widget.visible:
            # button.label = "HamburgerHamburgerHamburger"
            widget.visible = False
        else:
            # button.label = "CheeseburgerCheeseburgerCheeseburger"
            widget.visible = True
            # try to force resizing, bokeh bug workaround
            if hasattr(widget, "children") and widget.children:
                last_child = widget.children[len(widget.children) - 1]
                widget.children.remove(last_child)
                widget.children.append(last_child)

    button.on_click(burger_action)
    return column(top, widget)


class GIBandListener(ABC):
    """
    interface for classes that want to listen for updates to a set of bands.
    """

    @abstractmethod
    def adjust_band(self, band_id, start, stop):
        """
        Called when the model adjusted a band's range.

        Parameters
        ----------
        band_id : int
            ID of the band that was adjusted
        start : float
            New start of the range
        stop : float
            New end of the range
        """
        pass

    @abstractmethod
    def delete_band(self, band_id):
        """
        Called when the model deletes a band.

        Parameters
        ----------
        band_id : int
            ID of the band that was deleted
        """
        pass

    @abstractmethod
    def finish_bands(self):
        """
        Called by the model when a band update completes and any resulting
        band merges have already been done.
        """
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
        self.bands = dict()

    def add_listener(self, listener):
        """
        Add a listener to this band model.

        The listener can either be a :class:`GIBandListener` or
        it can be a function,  The function should expect as
        arguments, the `band_id`, and `start`, and `stop` x
        range values.

        Parameters
        ----------
        listener : :class:`~geminidr.interactive.interactive.GIBandListener`
        """
        if not isinstance(listener, GIBandListener):
            raise ValueError("must be a BandListener")
        self.listeners.append(listener)

    def load_from_tuples(self, tuples):
        for band_id in self.bands.keys():
            self.delete_band(band_id)
        self.band_id=1
        for tup in tuples:
            self.adjust_band(self.band_id, tup.start, tup.stop)
            self.band_id = self.band_id+1

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
        if start is not None and stop is not None and start > stop:
            start, stop = stop, start
        if start is not None:
            start = int(start)
        if stop is not None:
            stop = int(stop)
        self.bands[band_id] = [start, stop]
        for listener in self.listeners:
            listener.adjust_band(band_id, start, stop)

    def delete_band(self, band_id):
        """
        Delete a band by id

        Parameters
        ----------
        band_id : int
            ID of the band to delete

        """
        del self.bands[band_id]
        for listener in self.listeners:
            listener.delete_band(band_id)

    def finish_bands(self):
        """
        Finish operating on the bands.

        This call is for when a band update is completed.  During normal
        mouse movement, we update the view to be interactive.  We do not
        do more expensive stuff like merging intersecting bands together
        or updating the mask and redoing the fit.  That is done here
        instead once the band is set.
        """
        # first we do a little consolidation, in case we have overlaps
        band_dump = list()
        for key, value in self.bands.items():
            band_dump.append([key, value])
        for i in range(len(band_dump)-1):
            for j in range(i+1, len(band_dump)):
                # check for overlap and delete/merge bands
                akey, aband = band_dump[i]
                bkey, bband = band_dump[j]
                if (aband[0] is None or bband[1] is None or aband[0] < bband[1]) \
                        and (aband[1] is None or bband[0] is None or aband[1] > bband[0]):
                    # full overlap?
                    if (aband[0] is None or (bband[0] is not None and aband[0] <= bband[0])) \
                            and (aband[1] is None or (bband is not None and aband[1] >= bband[1])):
                        # remove bband
                        self.delete_band(bkey)
                    elif (bband[0] is None or (aband[0] is not None and aband[0] >= bband[0])) \
                            and (bband[1] is None or (aband[1] is not None and aband[1] <= bband[1])):
                        # remove aband
                        self.delete_band(akey)
                    else:
                        aband[0] = None if None in (aband[0], bband[0]) else min(aband[0], bband[0])
                        aband[1] = None if None in (aband[1], bband[1]) else max(aband[1], bband[1])
                        self.adjust_band(akey, aband[0], aband[1])
                        self.delete_band(bkey)
        for listener in self.listeners:
            listener.finish_bands()

    def find_band(self, x):
        """
        Find the first band that contains x in it's range, or return a tuple of None

        Parameters
        ----------
        x : float
            Find the first band that contains x, if any

        Returns
        -------
            tuple : (band id, start, stop) or (None, None, None) if there are no matches
        """
        for band_id, band in self.bands.items():
            if (band[0] is None or band[0] <= x) and (band[1] is None or x <= band[1]):
                return band_id, band[0], band[1]
        return None, None, None

    def closest_band(self, x):
        """
        Fid the band with an edge closest to x.

        Parameters
        ----------
        x : float
            x position to find closest edge to

        Returns
        -------
        int, float : int band id and float position of other edge or None, None if no bands exist
        """
        ret_band_id = None
        ret_band = None
        closest = None
        for band_id, band in self.bands.items():
            distance = None if band[1] is None else abs(band[1]-x)
            if closest is None or (distance is not None and distance < closest):
                ret_band_id = band_id
                ret_band = band[0]
                closest = distance
            distance = None if band[0] is None else abs(band[0] - x)
            if closest is None or (distance is not None and distance < closest):
                ret_band_id = band_id
                ret_band = band[1]
                closest = distance
        return ret_band_id, ret_band

    def contains(self, x):
        """
        Check if any of the bands contains point x

        Parameters
        ----------
        x : float
            point to check for band inclusion

        Returns
        -------
        bool : True if there are no bands defined, or if any band contains x in it's range
        """
        if len(self.bands.values()) == 0:
            return True
        for b in self.bands.values():
            if (b[0] is None or b[0] < x) and (b[1] is None or x < b[1]):
                return True
        return False

    def build_regions(self):
        def deNone(val):
            return '' if val is None else val
        if self.bands is None or len(self.bands.values()) == 0:
            return None
        return ','.join(['{}:{}'.format(deNone(b[0]), deNone(b[1])) for b in self.bands.values()])


class BandHolder(object):
    """
    Used by `~geminidr.interactive.interactive.GIBandView` to track start/stop
    independently of the bokeh Annotation since we want to support `None`.

    We need to know if the start/stop values are a specific value or `None`
    which is open ended left/right.
    """
    def __init__(self, annotation, start, stop):
        self.annotation = annotation
        self.start = start
        self.stop = stop


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
        fig : :class:`~bokeh.plotting.Figure`
            the figure to display the bands in
        model : :class:`~geminidr.interactive.interactive.GIBandModel`
            the model for the band information (may be shared by multiple
            :class:`~geminidr.interactive.interactive.GIBandView` instances)
        """
        self.fig = fig
        self.model = model
        model.add_listener(self)
        self.bands = dict()
        fig.y_range.on_change('start', lambda attr, old, new: self.update_viewport())
        fig.y_range.on_change('end', lambda attr, old, new: self.update_viewport())

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
        def fn():
            draw_start = start
            draw_stop = stop
            if draw_start is None:
                draw_start = self.fig.x_range.start - ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
            if draw_stop is None:
                draw_stop = self.fig.x_range.end + ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
            if band_id in self.bands:
                band = self.bands[band_id]
                band.start = start
                band.stop = stop
                band.annotation.left = draw_start
                band.annotation.right = draw_stop
            else:
                band = BoxAnnotation(left=draw_start, right=draw_stop, fill_alpha=0.1, fill_color='navy')
                self.fig.add_layout(band)
                self.bands[band_id] = BandHolder(band, start, stop)
        if self.fig.document is not None:
            self.fig.document.add_next_tick_callback(lambda: fn())
        else:
            # do it now
            fn()

    def update_viewport(self):
        """
        Update the view in the figure.

        This call is made whenever we detect a change in the display
        area of the view.  By redrawing, we ensure the lines and
        axis label are in view, at 80% of the way up the visible
        Y axis.

        """
        if self.fig.y_range.start is not None and self.fig.y_range.end is not None:
            for band in self.bands.values():
                if band.start is None or band.stop is None:
                    draw_start = band.start
                    draw_stop = band.stop
                    if draw_start is None:
                        draw_start = self.fig.x_range.start - ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
                    if draw_stop is None:
                        draw_stop = self.fig.x_range.end + ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
                    band.annotation.left = draw_start
                    band.annotation.right = draw_stop

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
        def fn():
            if band_id in self.bands:
                band = self.bands[band_id]
                band.annotation.left = 0
                band.annotation.right = 0
                band.start = 0
                band.stop = 0
                # TODO remove it (impossible?)
        # We have to defer this as the delete may come via the keypress URL
        # But we aren't in the PrimitiveVisualizaer so we reference the
        # document and queue it directly
        self.fig.document.add_next_tick_callback(lambda: fn())

    def finish_bands(self):
        pass


class GIApertureModel(ABC):
    """
    Model for tracking the Apertures.

    This tracks the apertures and a list of subscribers
    to notify when there are any changes.
    """
    def __init__(self):
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
        listener : :class:`~geminidr.interactive.interactive.GIApertureView`
            The listener to notify if there are any updates
        """
        self.listeners.append(listener)

    @abstractmethod
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
        pass

    @abstractmethod
    def adjust_aperture(self, aperture_id, location, start, end):
        """
        Adjust an existing aperture by ID to a new range.
        This will alert all subscribed listeners.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to adjust
        location : float
            X coordinate of the location of the aperture
        start : float
            X coordinate of the new start of range
        end : float
            X coordinate of the new end of range
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def clear_apertures(self):
        """
        Remove all apertures, calling delete on the listeners for each.
        """
        pass

    @abstractmethod
    def get_profile(self):
        pass

    @abstractmethod
    def find_closest(self, x):
        pass


class GISingleApertureView(object):
    def __init__(self, fig, aperture_id, location, start, end):
        """
        Create a visible glyph-set to show the existance
        of an aperture on the given figure.  This display
        will update as needed in response to panning/zooming.

        Parameters
        ----------
        gifig : :class:`~bokeh.plotting.Figure`
            Bokeh Figure to attach to
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
        self.location = None
        self.fig = None
        if fig.document:
            fig.document.add_next_tick_callback(lambda: self.build_ui(fig, aperture_id, location, start, end))
        else:
            self.build_ui(fig, aperture_id, location, start, end)

    def build_ui(self, fig, aperture_id, location, start, end):
        """
        Build the view in the figure.

        This call creates the UI elements for this aperture in the
        parent figure.  It also wires up event listeners to adjust
        the displayed glyphs as needed when the view changes.

        Parameters
        ----------
        fig : :class:`~bokeh.plotting.Figure`
            bokeh figure to attach glyphs to
        aperture_id : int
            ID of this aperture, displayed
        location : float
            Location of the aperture
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
        self.label = Label(x=location+2, y=ymid, text="%s" % aperture_id)
        fig.add_layout(self.label)
        self.left_source = ColumnDataSource({'x': [start, start], 'y': [ybottom, ytop]})
        self.left = fig.line(x='x', y='y', source=self.left_source, color="purple")
        self.right_source = ColumnDataSource({'x': [end, end], 'y': [ybottom, ytop]})
        self.right = fig.line(x='x', y='y', source=self.right_source, color="purple")
        self.line_source = ColumnDataSource({'x': [start, end], 'y': [ymid, ymid]})
        self.line = fig.line(x='x', y='y', source=self.line_source, color="purple")
        self.location = Span(location=location, dimension='height', line_color='green', line_dash='dashed',
                             line_width=1)
        fig.add_layout(self.location)

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

    def update(self, location, start, end):
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
        self.location.location = location
        self.label.x = location+2

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
        self.location.visible=False


class GIApertureSliders(object):
    def __init__(self, fig, model, aperture_id, location, start, end):
        """
        Create range sliders for an aperture.

        This creates a range slider and a pair of linked text
        entry boxes for the start and end of an aperture.

        Parameters
        ----------
        fig : :class:`~bokeh.plotting.Figure`
            The bokeh figure being plotted in, for handling zoom in/out
        model : :class:`~geminidr.interactive.interactive.GIApertureModel`
            The model that tracks the apertures and their ranges
        aperture_id : int
            The ID of the aperture
        start : float
            The start of the aperture
        end : float
            The end of the aperture
        """
        # self.view = view
        self.model = model
        self.aperture_id = aperture_id
        self.location = location
        self.start = start
        self.end = end

        slider_start = fig.x_range.start
        slider_end = fig.x_range.end

        self.slider = build_range_slider("%s" % aperture_id, location, start, end, 0.01, slider_start, slider_end,
                                         obj=self, location_attr="location", start_attr="start", end_attr="end",
                                         handler=self.do_update)
        button = Button(label="Del")
        button.on_click(self.delete_from_model)
        button.width = 48

        self.component = row(self.slider, button)

    def set_title(self, title):
        self.slider.children[0].title = title

    def delete_from_model(self):
        """
        Delete this aperture from the model
        """
        self.model.delete_aperture(self.aperture_id)

    def update_viewport(self, start, end):
        """
        Respond to a viewport update.

        This checks the visible range of the viewport against
        the current range of this aperture.  If the aperture
        is not fully contained within the new visible area,
        all UI elements are disabled.  If the aperture is in
        range, the start and stop values for the slider are
        capped to the visible range.

        Parameters
        ----------
        start : float
            Visible start of x axis
        end : float
            Visible end of x axis
        """
        if self.start < start or self.end > end:
            self.slider.children[0].disabled = True
            self.slider.children[1].disabled = True
            self.slider.children[2].disabled = True
            self.slider.children[3].disabled = True
        else:
            self.slider.children[0].disabled = False
            self.slider.children[1].disabled = False
            self.slider.children[2].disabled = False
            self.slider.children[3].disabled = False
            self.slider.children[0].start = start
            self.slider.children[0].end = end

    def do_update(self):
        if self.start > self.end:
            self.model.adjust_aperture(self.aperture_id, self.location, self.end, self.start)
        else:
            self.model.adjust_aperture(self.aperture_id, self.location, self.start, self.end)


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
        model : :class:`~geminidr.interactive.interactive.GIApertureModel`
            Model for tracking the apertures, may be shared across multiple views
        fig : :class:`~bokeh.plotting.Figure`
            bokeh plot for displaying the bands
        """
        self.aps = list()
        self.ap_sliders = list()

        self.fig = fig
        self.controls = column()
        self.controls.height_policy = "auto"
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
        # Bokeh docs provide no indication of the datatype or orientation of the start/end
        # tuples, so I have left the doc blank for now
        self.view_start = start
        self.view_end = end
        for ap_slider in self.ap_sliders:
            ap_slider.update_viewport(start, end)

    def handle_aperture(self, aperture_id, location, start, end):
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
            ap.update(location, start, end)
        else:
            ap = GISingleApertureView(self.fig, aperture_id, location, start, end)
            self.aps.append(ap)
            slider = GIApertureSliders(self.fig, self.model, aperture_id, location, start, end)
            self.ap_sliders.append(slider)
            self.inner_controls.children.append(slider.component)

    def delete_aperture(self, aperture_id):
        """
        Remove an aperture by ID.  If the ID is not recognized, do nothing.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to remove
        """
        if aperture_id <= len(self.aps):
            ap = self.aps[aperture_id-1]
            ap.delete()
            self.inner_controls.children.remove(self.ap_sliders[aperture_id-1].component)
            del self.aps[aperture_id-1]
            del self.ap_sliders[aperture_id-1]
        for ap in self.aps[aperture_id-1:]:
            ap.aperture_id = ap.aperture_id-1
            ap.label.text = "%s" % ap.aperture_id
        for ap_slider in self.ap_sliders[aperture_id-1:]:
            ap_slider.aperture_id = ap_slider.aperture_id-1
            ap_slider.set_title("%s" % ap_slider.aperture_id)
            # ap_slider.label.text = "<h3>Aperture %s</h3>" % ap_slider.aperture_id
