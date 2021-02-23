from abc import ABC, abstractmethod
from copy import copy

from bokeh.layouts import column, row
from bokeh.models import (BoxAnnotation, Button, CustomJS, Dropdown,
                          NumeralTickFormatter, RangeSlider, Slider, TextInput, Div)

from geminidr.interactive import server
from geminidr.interactive.server import register_callback
from gempy.library.astrotools import cartesian_regions_to_slices
from gempy.library.config import FieldValidationError


class PrimitiveVisualizer(ABC):
    def __init__(self, config=None, title='', primitive_name='', filename_info='', template=None):
        """
        Initialize a visualizer.

        This base class creates a submit button suitable for any subclass
        to use and it also listens for the UI window to close, executing a
        submit if that happens.  The submit button will cause the `bokeh`
        event loop to exit and the code will resume executing in whatever
        top level call you are visualizing from.
        """
        self.title = title
        self.filename_info = filename_info if filename_info else ''
        self.primitive_name = primitive_name if primitive_name else ''
        self.template = template
        self.extras = dict()
        if config is None:
            self.config = None
        else:
            self.config = copy(config)

        self.user_satisfied = False

        self.bokeh_legend = Div(text='Plot Tools<br/><img src="dragons/static/bokehlegend.png" />')
        self.submit_button = Button(label="Accept", align='center', button_type='success', width_policy='min')
        self.submit_button.on_click(self.submit_button_handler)
        callback = CustomJS(code="""
            window.close();
        """)
        self.submit_button.js_on_click(callback)
        self.doc = None

    def make_ok_cancel_dialog(self, btn, message, callback):
        # This is a bit hacky, but bokeh makes it very difficult to bridge the python-js gap.
        def _internal_handler(args):
            if callback:
                if args['result'] == [b'confirmed']:
                    result = True
                else:
                    result = False
                self.do_later(lambda: callback(result))

        callback_name = register_callback(_internal_handler)

        js_confirm_callback = CustomJS(code="""
            console.log('in confirm callback');
            console.log('element disabled, showing ok/cancel');
            debugger;
            cb_obj.name = '';
            var confirmed = confirm('%s');
            var cbid = '%s';
            console.log('OK/Cancel Result: ' + confirmed);
            if (confirmed) {
                $.ajax('/handle_callback?callback=' + cbid + '&result=confirmed');
            } else {
                $.ajax('/handle_callback?callback=' + cbid + '&result=rejected');
            }
            """ % (message, callback_name))
        btn.js_on_click(js_confirm_callback)

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

        # doc.add_root(self._ok_cancel_dlg.layout)
        # Add an OK/Cancel dialog we can tap into later

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
            True if recalcuating points is cheap, in which case we don't need a button and do it on any change.
            Currently only viable for text-slider style inputs

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


def build_text_slider(title, value, step, min_value, max_value, obj=None,
                      attr=None, handler=None, throttled=False):
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
    is_float = not isinstance(value, int)

    start = min(value, min_value) if min_value else min(value, 0)
    end = max(value, max_value) if max_value else max(10, value*2)

    # trying to convince int-based sliders to behave
    if not is_float:
        fmt = NumeralTickFormatter(format='0,0')
        slider = Slider(start=start, end=end, value=value, step=step, title=title, format=fmt)
    else:
        slider = Slider(start=start, end=end, value=value, step=step, title=title)
    slider.width = 256

    text_input = TextInput(width=64, value=str(value))
    component = row(slider, text_input)

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


def connect_figure_extras(fig, aperture_model, region_model):
    """
    Connect a figure to an aperture and region model for rendering.

    This call will add extra visualizations to the bokeh figure to
    show the regions and apertures in the given models.  Either may
    be passed as None if not relevant.

    This call also does a fix to bokeh to work around a rendering bug.

    Parameters
    ----------
    fig : :class:`~bokeh.plotting.Figure`
        bokeh Figure to add visualizations too
    aperture_model : :class:`~geminidr.interactive.interactive.GIApertureModel`
        Aperture model to add view for
    region_model : :class:`~geminidr.interactive.interactive.GIRegionModel`
        Band model to add view for
    """
    # If we have regions or apertures to show, show them
    if region_model:
        regions = GIRegionView(fig, region_model)
    if aperture_model:
        aperture_view = GIApertureView(aperture_model, fig)

    # This is a workaround for a bokeh bug.  Without this, things like the background shading used for
    # apertures and regions will not update properly after the figure is visible.
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
        widget.css_classes = ['hamburger_helper_%s' % _hamburger_order_number]

    _hamburger_order_number += 1

    # TODO Hamburger icon
    button = Button(label=title, css_classes=['hamburger_helper',])

    def burger_action():
        if widget.visible:
            widget.visible = False
        else:
            widget.visible = True
            # try to force resizing, bokeh bug workaround
            if hasattr(widget, "children") and widget.children:
                last_child = widget.children[len(widget.children) - 1]
                widget.children.remove(last_child)
                widget.children.append(last_child)

    button.on_click(burger_action)
    return column(button, widget)


class GIRegionListener(ABC):
    """
    interface for classes that want to listen for updates to a set of regions.
    """

    @abstractmethod
    def adjust_region(self, region_id, start, stop):
        """
        Called when the model adjusted a region's range.

        Parameters
        ----------
        region_id : int
            ID of the region that was adjusted
        start : float
            New start of the range
        stop : float
            New end of the range
        """
        pass

    @abstractmethod
    def delete_region(self, region_id):
        """
        Called when the model deletes a region.

        Parameters
        ----------
        region_id : int
            ID of the region that was deleted
        """
        pass

    @abstractmethod
    def finish_regions(self):
        """
        Called by the model when a region update completes and any resulting
        region merges have already been done.
        """
        pass


class GIRegionModel:
    """
    Model for tracking a set of regions.
    """
    def __init__(self):
        # Right now, the region model is effectively stateless, other
        # than maintaining the set of registered listeners.  That is
        # because the regions are not used for anything, so there is
        # no need to remember where they all are.  This is likely to
        # change in future and that information should likely be
        # kept in here.
        self.region_id = 1
        self.listeners = list()
        self.regions = dict()

    def add_listener(self, listener):
        """
        Add a listener to this region model.

        The listener can either be a :class:`GIRegionListener` or
        it can be a function,  The function should expect as
        arguments, the `region_id`, and `start`, and `stop` x
        range values.

        Parameters
        ----------
        listener : :class:`~geminidr.interactive.interactive.GIRegionListener`
        """
        if not isinstance(listener, GIRegionListener):
            raise ValueError("must be a BandListener")
        self.listeners.append(listener)

    def load_from_tuples(self, tuples):
        region_ids = list(self.regions.keys())
        for region_id in region_ids:
            self.delete_region(region_id)
        self.region_id=1
        for tup in tuples:
            self.adjust_region(self.region_id, tup.start, tup.stop)
            self.region_id = self.region_id + 1

    def load_from_string(self, region_string):
        self.load_from_tuples(cartesian_regions_to_slices(region_string))

    def adjust_region(self, region_id, start, stop):
        """
        Adjusts the given region ID to the specified X range.

        The region ID may refer to a brand new region ID as well.
        This method will call into all registered listeners
        with the updated information.

        Parameters
        ----------
        region_id : int
            ID fo the region to modify
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
        self.regions[region_id] = [start, stop]
        for listener in self.listeners:
            listener.adjust_region(region_id, start, stop)

    def delete_region(self, region_id):
        """
        Delete a region by id

        Parameters
        ----------
        region_id : int
            ID of the region to delete

        """
        del self.regions[region_id]
        for listener in self.listeners:
            listener.delete_region(region_id)

    def finish_regions(self):
        """
        Finish operating on the regions.

        This call is for when a region update is completed.  During normal
        mouse movement, we update the view to be interactive.  We do not
        do more expensive stuff like merging intersecting regions together
        or updating the mask and redoing the fit.  That is done here
        instead once the region is set.
        """
        # first we do a little consolidation, in case we have overlaps
        region_dump = list()
        for key, value in self.regions.items():
            region_dump.append([key, value])
        for i in range(len(region_dump)-1):
            for j in range(i+1, len(region_dump)):
                # check for overlap and delete/merge regions
                akey, aregion = region_dump[i]
                bkey, bregion = region_dump[j]
                if (aregion[0] is None or bregion[1] is None or aregion[0] < bregion[1]) \
                        and (aregion[1] is None or bregion[0] is None or aregion[1] > bregion[0]):
                    # full overlap?
                    if (aregion[0] is None or (bregion[0] is not None and aregion[0] <= bregion[0])) \
                            and (aregion[1] is None or (bregion is not None and aregion[1] >= bregion[1])):
                        # remove bregion
                        self.delete_region(bkey)
                    elif (bregion[0] is None or (aregion[0] is not None and aregion[0] >= bregion[0])) \
                            and (bregion[1] is None or (aregion[1] is not None and aregion[1] <= bregion[1])):
                        # remove aregion
                        self.delete_region(akey)
                    else:
                        aregion[0] = None if None in (aregion[0], bregion[0]) else min(aregion[0], bregion[0])
                        aregion[1] = None if None in (aregion[1], bregion[1]) else max(aregion[1], bregion[1])
                        self.adjust_region(akey, aregion[0], aregion[1])
                        self.delete_region(bkey)
        for listener in self.listeners:
            listener.finish_regions()

    def find_region(self, x):
        """
        Find the first region that contains x in it's range, or return a tuple of None

        Parameters
        ----------
        x : float
            Find the first region that contains x, if any

        Returns
        -------
            tuple : (region id, start, stop) or (None, None, None) if there are no matches
        """
        for region_id, region in self.regions.items():
            if (region[0] is None or region[0] <= x) and (region[1] is None or x <= region[1]):
                return region_id, region[0], region[1]
        return None, None, None

    def closest_region(self, x):
        """
        Fid the region with an edge closest to x.

        Parameters
        ----------
        x : float
            x position to find closest edge to

        Returns
        -------
        int, float : int region id and float position of other edge or None, None if no regions exist
        """
        ret_region_id = None
        ret_region = None
        closest = None
        for region_id, region in self.regions.items():
            distance = None if region[1] is None else abs(region[1]-x)
            if closest is None or (distance is not None and distance < closest):
                ret_region_id = region_id
                ret_region = region[0]
                closest = distance
            distance = None if region[0] is None else abs(region[0] - x)
            if closest is None or (distance is not None and distance < closest):
                ret_region_id = region_id
                ret_region = region[1]
                closest = distance
        return ret_region_id, ret_region

    def contains(self, x):
        """
        Check if any of the regions contains point x

        Parameters
        ----------
        x : float
            point to check for region inclusion

        Returns
        -------
        bool : True if there are no regions defined, or if any region contains x in it's range
        """
        if len(self.regions.values()) == 0:
            return True
        for b in self.regions.values():
            if (b[0] is None or b[0] < x) and (b[1] is None or x < b[1]):
                return True
        return False

    def build_regions(self):
        def deNone(val, offset=0):
            return '' if val is None else val + offset
        if self.regions is None or len(self.regions.values()) == 0:
            return None
        return ','.join(['{}:{}'.format(deNone(b[0],offset=1), deNone(b[1])) for b in self.regions.values()])


class RegionHolder:
    """
    Used by `~geminidr.interactive.interactive.GIRegionView` to track start/stop
    independently of the bokeh Annotation since we want to support `None`.

    We need to know if the start/stop values are a specific value or `None`
    which is open ended left/right.
    """
    def __init__(self, annotation, start, stop):
        self.annotation = annotation
        self.start = start
        self.stop = stop


class GIRegionView(GIRegionListener):
    """
    View for the set of regions to show then in a figure.
    """
    def __init__(self, fig, model):
        """
        Create the view for the set of regions managed in the given model
        to display them in a figure.

        Parameters
        ----------
        fig : :class:`~bokeh.plotting.Figure`
            the figure to display the regions in
        model : :class:`~geminidr.interactive.interactive.GIRegionModel`
            the model for the region information (may be shared by multiple
            :class:`~geminidr.interactive.interactive.GIRegionView` instances)
        """
        self.fig = fig
        self.model = model
        model.add_listener(self)
        self.regions = dict()
        fig.y_range.on_change('start', lambda attr, old, new: self.update_viewport())
        fig.y_range.on_change('end', lambda attr, old, new: self.update_viewport())

    def adjust_region(self, region_id, start, stop):
        """
        Adjust a region by it's ID.

        This may also be a new region, if it is an ID we haven't
        seen before.  This call will create or adjust the glyphs
        in the figure to reflect the new data.

        Parameters
        ----------
        region_id : int
            id of region to create or adjust
        start : float
            start of the x range of the region
        stop : float
            end of the x range of the region
        """
        def fn():
            draw_start = start
            draw_stop = stop
            if draw_start is None:
                draw_start = self.fig.x_range.start - ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
            if draw_stop is None:
                draw_stop = self.fig.x_range.end + ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
            if region_id in self.regions:
                region = self.regions[region_id]
                region.start = start
                region.stop = stop
                region.annotation.left = draw_start
                region.annotation.right = draw_stop
            else:
                region = BoxAnnotation(left=draw_start, right=draw_stop, fill_alpha=0.1, fill_color='navy')
                self.fig.add_layout(region)
                self.regions[region_id] = RegionHolder(region, start, stop)
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
            for region in self.regions.values():
                if region.start is None or region.stop is None:
                    draw_start = region.start
                    draw_stop = region.stop
                    if draw_start is None:
                        draw_start = self.fig.x_range.start - ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
                    if draw_stop is None:
                        draw_stop = self.fig.x_range.end + ((self.fig.x_range.end - self.fig.x_range.start) / 10.0)
                    region.annotation.left = draw_start
                    region.annotation.right = draw_stop

    def delete_region(self, region_id):
        """
        Delete a region by ID.

        If the view does not recognize the id, this is a no-op.
        Otherwise, all related glyphs are cleaned up from the figure.

        Parameters
        ----------
        region_id : int
            ID of region to remove

        """
        def fn():
            if region_id in self.regions:
                region = self.regions[region_id]
                region.annotation.left = 0
                region.annotation.right = 0
                region.start = 0
                region.stop = 0
                # TODO remove it (impossible?)
        # We have to defer this as the delete may come via the keypress URL
        # But we aren't in the PrimitiveVisualizaer so we reference the
        # document and queue it directly
        self.fig.document.add_next_tick_callback(lambda: fn())

    def finish_regions(self):
        pass


class RegionEditor(GIRegionListener):
    def __init__(self, region_model):
        self.text_input = TextInput(title="Regions (i.e. 100:500,510:900,950:   press 'Enter' to apply")
        self.text_input.value = region_model.build_regions()
        self.region_model = region_model
        self.region_model.add_listener(self)
        self.text_input.on_change("value", self.handle_text_value)

    def adjust_region(self, region_id, start, stop):
        pass

    def delete_region(self, region_id):
        self.text_input.value = self.region_model.build_regions()

    def finish_regions(self):
        self.text_input.value = self.region_model.build_regions()

    def handle_text_value(self, attr, old, new):
        current = self.region_model.build_regions()
        if current != new:
            self.region_model.load_from_string(new)

    def get_widget(self):
        return self.text_input
