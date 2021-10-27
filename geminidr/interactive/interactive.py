import re
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import cmp_to_key

from bokeh.core.property.instance import Instance
from bokeh.layouts import column, row
from bokeh.models import (BoxAnnotation, Button, CustomJS, Dropdown,
                          NumeralTickFormatter, Slider, TextInput, Div, NumericInput, PreText)
from bokeh import models as bm

from geminidr.interactive import server
from geminidr.interactive.fit.help import DEFAULT_HELP
from geminidr.interactive.server import register_callback
from gempy.library.astrotools import cartesian_regions_to_slices, parse_user_regions
from gempy.library.config import FieldValidationError, Config

# Singleton instance, there is only ever one of these
from gempy.utils import logutils


__all__ = ["FitQuality", "PrimitiveVisualizer", "build_text_slider", "connect_region_model",
           "GIRegionListener", "GIRegionModel", "RegionEditor", "TabsTurboInjector", "UIParameters",
           "do_later"]


_visualizer = None


_log = logutils.get_logger(__name__)


class FitQuality(Enum):
    GOOD = auto()
    POOR = auto()
    BAD = auto()


class PrimitiveVisualizer(ABC):
    def __init__(self, title='', primitive_name='',
                 filename_info='', template=None, help_text=None, ui_params=None):
        """
        Initialize a visualizer.

        This base class creates a submit button suitable for any subclass
        to use and it also listens for the UI window to close, executing a
        submit if that happens.  The submit button will cause the `bokeh`
        event loop to exit and the code will resume executing in whatever
        top level call you are visualizing from.

        Parameters
        ----------
        title : str
            Title fo the primitive for display, currently not used
        primitive_name : str
            Name of the primitive function related to this UI, used in the title bar
        filename_info : str
            Information about the file being operated on
        template : str
            Optional path to an html template to render against, if customization is desired
        """
        global _visualizer
        _visualizer = self

        # set help to default, subclasses should override this with something specific to them
        self.help_text = help_text if help_text else DEFAULT_HELP

        self.exited = False
        self.title = title
        self.filename_info = filename_info if filename_info else ''
        self.primitive_name = primitive_name if primitive_name else ''
        self.template = template
        self.extras = dict()
        self.ui_params = ui_params

        self.user_satisfied = False

        self.bokeh_legend = Div(text='Plot Tools<br/><img src="dragons/static/bokehlegend.png" />')
        self.submit_button = Button(align='center',
                                    button_type='success',
                                    css_classes=["submit_btn"],
                                    id="_submit_btn",
                                    label="Accept",
                                    name="submit_btn",
                                    )
        self.abort_button = Button(align='center',
                                   button_type='warning',
                                   css_classes=["submit_btn"],
                                   id="_warning_btn",
                                   label="Abort",
                                   name="abort_btn",
                                   )
        # The submit_button_handler is only needed to flip the user_accepted flag to True before
        # the bokeh event loop terminates
        # self.submit_button.on_click(self.submit_button_handler)
        # This window closing will end the session.  That is what
        # causes the bokeh event look to terminate via session_ended().
        callback = CustomJS(code="""
            $.ajax('/shutdown').done(function()
                {
                    window.close();
                });
        """)

        # Listen to the disabled state and tweak that inside submit_button_handler
        # This allows us to execute a python callback to the submit_button on click.  Then,
        # if we decide to execute the /shutdown via javascript, we can just 'disable' the
        # submit button to trigger that logic (DOM events are the only way to trigger
        # JS callbacks in bokeh)
        self.submit_button.on_click(self.submit_button_handler)
        self.submit_button.js_on_change('disabled', callback)
        # self.submit_button.js_on_click(callback)

        abort_callback = CustomJS(code="""
            $.ajax('/shutdown?user_satisfied=false').done(function()
                {
                    window.close();
                });
        """)
        self.abort_button.on_click(self.abort_button_handler)
        self.abort_button.js_on_change('disabled', abort_callback)

        self.doc = None
        self._message_holder = None
        # callback for the new (buttonless) ok/cancel dialog.
        # This gets set just before the dialog is triggered
        self._ok_cancel_callback = None
        # Text widget for triggering ok/cancel via DOM text change event
        self._ok_cancel_holder = None

        self.fits = []

    def make_ok_cancel_dialog(self, btn, message, callback):
        """
        Make an OK/Cancel dialog that will trigger when the given button `btn` is set disabled

        Note this method is superceded by :meth:`show_ok_cancel` which is not dependant on
        disabling buttons.  Avoid using this as it will be refactored out at some point.

        Parameters
        ----------
        btn : :class:`~bokeh.models.Button`
            bokeh Button to listen for disabled
        message : str
            String message to show in the ok/cancel dialog
        callback : function
            Function to call with True/False for the user selection of OK/Cancel
        """
        # This is an older version of ok/cancel that requires a button.  The button is
        # disabled to activate the dialog.  More recently, we needed ok/cancel without
        # disabling a button so that is also available.  See `show_ok_cancel`
        # TODO refactor this out in favor of the other exclusively

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
            cb_obj.name = '';
            var confirmed = confirm('%s');
            var cbid = '%s';
            if (confirmed) {
                $.ajax('/handle_callback?callback=' + cbid + '&result=confirmed');
            } else {
                $.ajax('/handle_callback?callback=' + cbid + '&result=rejected');
            }
            """ % (message, callback_name))
        btn.js_on_click(js_confirm_callback)

    def submit_button_handler(self):
        """
        Submit button handler.

        This handler checks the sanity of the fit(s) and considers three possibilities.

        1) The fit is good, we proceed ahead as normal
        2) The fit is bad, we pop up a message dialog for the user and they hit 'OK' to return to the UI
        3) The fit is poor, we pop up an ok/cancel dialog for the user and continue or return to the UI as directed.
        """
        bad_fits = ", ".join(tab.title for fit, tab in zip(self.fits, self.tabs.tabs)
                             if fit.quality == FitQuality.BAD)
        poor_fits = ", ".join(tab.title for fit, tab in zip(self.fits, self.tabs.tabs)
                              if fit.quality == FitQuality.POOR)
        if bad_fits:
            # popup message
            self.show_user_message(f"Failed fit(s) on {bad_fits}. Please "
                                   "modify the parameters and try again.")
        elif poor_fits:
            def cb(accepted):
                if accepted:
                    # Trigger the exit/fit, otherwise we do nothing
                    self.submit_button.disabled = True
            self.show_ok_cancel(f"Poor quality fit(s)s on {poor_fits}. Click "
                                "OK to proceed anyway, or Cancel to return to "
                                "the fitter.", cb)
        else:
            # Fit is good, we can exit
            # Trigger the submit callback via disabling the submit button
            self.submit_button.disabled = True

    def abort_button_handler(self):
        """
        Used by the abort button to provide a last ok/cancel dialog to the
        user before killing reduce.
        """
        def cb(accepted):
            if accepted:
                # Trigger the exit/fit, otherwise we do nothing
                _log.warn("Aborting reduction on user request")
                self.abort_button.disabled = True

        self.show_ok_cancel(f"Are you sure you want to abort?  DRAGONS reduce will exit completely.", cb)

    def session_ended(self, sess_context, user_satisfied):
        """
        Handle the end of the session by stopping the bokeh server, which
        will resume python execution in the DRAGONS primitive.

        Parameters
        ----------
        sess_context : Any
            passed by bokeh, but we do not use it

        user_satisfied : bool
            True if the user was satisfied (i.e. we are responding to the submit button)

        Returns
        -------
        none
        """
        self.user_satisfied = user_satisfied
        if not self.exited:
            self.exited = True
            server.stop_server()

    def get_filename_div(self):
        """
        Returns a Div element that displays the current filename.
        """
        div = Div(text=f"<b>Current&nbsp;filename:&nbsp;</b>&nbsp;{self.filename_info}",
                  style={
                         "color": "dimgray",
                         "font-size": "16px",
                         "float": "right",
                  },
                  align="end",
                  )
        return div

    @abstractmethod
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
        # This is now called via show() to make the code cleaner
        # with respect to some before/after boilerplate.  It also
        # reduces the chances someone will forget to super() call this

    def show(self, doc):
        """
        Show the interactive fitter.

        This is called via bkapp by the bokeh server and happens
        when the bokeh server is spun up to interact with the user.

        This method also detects if it is running in 'test' mode
        and will build the UI and automatically submit it with the
        input parameters.

        Parameters
        ----------
        doc : :class:`~bokeh.document.document.Document`
            Bokeh document, this is saved for later in :attr:`~geminidr.interactive.interactive.PrimitiveVisualizer.doc`
        """
        self.doc = doc
        doc.on_session_destroyed(lambda stuff: self.session_ended(stuff, False))

        self.visualize(doc)

        if server.test_mode:
            # Simulate a click of the accept button
            self.do_later(lambda: self.submit_button_handler(None))

        # Add a widget we can use for triggering a message
        # This is a workaround, since CustomJS calls can only
        # respond to DOM events.  We'll be able to trigger
        # a Custom JS callback by modifying this widget
        self._message_holder = PreText(text='', css_classes=['hidden'])
        callback = CustomJS(args={}, code='alert(cb_obj.text);')
        self._message_holder.js_on_change('text', callback)

        # Add the invisible PreText element to drive message dialogs off
        # of.  We do this with a do_later so that it will happen after the
        # subclass implementation does all of it's document setup.  So,
        # this widget will be added at the end.
        self.do_later(lambda: doc.add_root(row(self._message_holder, )))

        # and we have to hide it, the css class isn't enough
        def _hide_message_holder():
            self._message_holder.visible = False
        self.do_later(_hide_message_holder)

        #################
        # OK/Cancel Setup
        #################
        # This is a workaround for bokeh so we can drive an ok/cancel dialog box
        # and have the response sent back down via a Tornado web endpoint.  This
        # is not dependent on being tied to a button like the earlier version.
        # It does, therefore, need it's own widget which we supply as hidden
        # and also double as the means for passing the text message to the js
        def _internal_ok_cancel_handler(args):
            if args['result'] == [b'confirmed']:
                result = True
            else:
                result = False
            self.do_later(lambda: self._ok_cancel_callback(result))

        # callback_name is the unique ID that will be passed back in to the /handle_callback endpoint
        # so it will execute the python method _internal_ok_cancel_handler
        callback_name = register_callback(_internal_ok_cancel_handler)

        # This JS callback will execute when the ok/cancel exits
        ok_cancel_callback = CustomJS(code="""
            cb_obj.name = '';
            var confirmed = confirm(cb_obj.text);
            var cbid = '%s';
            if (confirmed) {
                $.ajax('/handle_callback?callback=' + cbid + '&result=confirmed');
            } else {
                $.ajax('/handle_callback?callback=' + cbid + '&result=rejected');
            }
            """ % (callback_name,))

        # Add a widget we can use for triggering an ok/cancel
        # This is a workaround, since CustomJS calls can only
        # respond to DOM events.  We'll be able to trigger
        # a Custom JS callback by modifying this widget
        self._ok_cancel_holder = PreText(text='', css_classes=['hidden'])
        self._ok_cancel_holder.js_on_change('text', ok_cancel_callback)

        # Add the invisible PreText element to drive message dialogs off
        # of.  We do this with a do_later so that it will happen after the
        # subclass implementation does all of it's document setup.  So,
        # this widget will be added at the end.
        self.do_later(lambda: doc.add_root(row(self._ok_cancel_holder, )))

    def show_ok_cancel(self, message, callback):
        """
        New OK/Cancel dialog helper method.

        Use this in preference to the :meth:`make_ok_cancel_dialog` method
        which will be deprecated in future.

        Parameters
        ----------
        message : str
            Text message to show in the ok/cancel dialog
        callback : function
            Function to call with a bool of True if the user hit OK, or False for Cancel
        """
        # This saves the `callback` in the `_ok_cancel_callback` field.  When
        # the callback comes back into the bokeh server, that is the function
        # that will be called with True (OK) or False (Cancel).
        # We wrap it in a do_later so it will execute on the UI thread.
        self._ok_cancel_callback = lambda x: self.do_later(lambda: callback(x))

        # modifying the text of this hidden widget will trigger the ok/cancel dialog
        # which will use the text value as it's message.  Then the dialog will
        # make an AJAX back in and call the `callback` method.
        if self._ok_cancel_holder.text == message:
            # needs to be different to trigger the javascript
            self._ok_cancel_holder.text = f"{message} "
        else:
            self._ok_cancel_holder.text = message

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
            if hasattr(self, 'log') and self.log is not None:
                self.log.warn("Call to do_later, but no document is set.  Does this PrimitiveVisualizer call "
                              "super().visualize(doc)?")
            # no doc, probably ok to just execute
            fn()
        else:
            self.doc.add_next_tick_callback(fn)

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

    def show_user_message(self, message):
        """
        Make a modal message dialog to display to the user.

        Parameters
        ----------
        message : str
            message to display in the popup modal
        """
        # In the constructor, we setup this throwaway widget
        # so we could listen for changes in it's text field
        # and display those via an alert.  It's a workaround
        # so that here we can send messages to the user from
        # the bokeh server-side python.
        if self._message_holder.text == message:
            # need to trigger a change...
            self._message_holder.text = f"{message} "
        else:
            self._message_holder.text = message

    def make_widgets_from_parameters(self, params, reinit_live: bool = True,
                                     slider_width: int = 256):
        """
        Makes appropriate widgets for all the parameters in params,
        using the config to determine the type. Also adds these widgets
        to a dict so they can be accessed from the calling primitive.

        Parameters
        ----------
        params : :class:`UIParameters`
            Parameters to make widgets for
        reinit_live : bool
            True if recalcuating points is cheap, in which case we don't need a button and do it on any change.
            Currently only viable for text-slider style inputs
        slider_width : int
            Width of the sliders

        Returns
        -------
        list : Returns a list of widgets to display in the UI.
        """
        widgets = []
        if params.reinit_params:
            for key in params.reinit_params:
                field = params.fields[key]
                if hasattr(field, 'min'):
                    is_float = field.dtype is not int
                    if is_float:
                        step = 0.1
                    else:
                        step = 1
                    widget = build_text_slider(
                        params.titles[key], params.values[key], step, field.min, field.max, obj=params.values,
                        attr=key, slider_width=slider_width, allow_none=field.optional, throttled=True,
                        is_float=is_float,
                        handler=self.slider_handler_factory(key, reinit_live=reinit_live))

                    self.widgets[key] = widget.children[0]
                elif hasattr(field, 'allowed'):
                    # ChoiceField => drop-down menu
                    widget = Dropdown(label=field.title, menu=list(field.allowed.keys()))
                    self.widgets[key] = widget
                elif field.dtype is bool:
                    widget = bm.CheckboxGroup(labels=[params.titles[key]],
                                              active=[0] if params.values[key] else [])
                    self.widgets[key] = widget
                else:
                    # Anything else
                    widget = TextInput(title=field.title)
                    self.widgets[key] = widget

                widgets.append(widget)
        return widgets

    def slider_handler_factory(self, key, reinit_live=False):
        """
        Returns a function that updates the `extras` attribute.

        Parameters
        ----------
        key : str
            The parameter name to be updated.
        reinit_live : bool, optional
            Update the reconstructed points on "real time".

        Returns
        -------
        function : callback called when we change the slider value.
        """

        def handler(val):
            self.extras[key] = val
            if reinit_live:
                self.reconstruct_points()

        return handler


def build_text_slider(title, value, step, min_value, max_value, obj=None,
                      attr=None, handler=None, throttled=False,
                      slider_width=256, config=None, allow_none=False,
                      is_float=None):
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
    allow_none : bool
        Set to `True` to allow an empty text entry to specify a `None` value
    is_float : bool
        nature of parameter (None => try to figure it out)

    Returns
    -------
        :class:`~bokeh.models.layouts.Row` bokeh Row component with the interface inside

    """
    if min_value is None and config is not None:
        field = config._fields.get(attr, None)
        if field is not None:
            if hasattr(field, 'min'):
                min_value = field.min
    if max_value is None and config is not None:
        field = config._fields.get(attr, None)
        if field is not None:
            if hasattr(field, 'max'):
                max_value = field.max

    if value is None:
        start = min_value if min_value is not None else 0
        end = max_value if max_value is not None else 10
        slider_kwargs = {"value": start, "show_value": False}
    else:
        start = min(value, min_value) if min_value is not None else min(value, 0)
        end = max(value, max_value) if max_value is not None else max(10, value*2)
        slider_kwargs = {"value": value, "show_value": True}

    # trying to convince int-based sliders to behave
    if is_float is None:
        is_float = ((value is not None and not isinstance(value, int)) or
                    (min_value is not None and not isinstance(min_value, int)) or
                    (max_value is not None and not isinstance(max_value, int)))
    if step is None:
        if is_float:
            step = 0.1
        else:
            step = 1
    fmt = None
    if not is_float:
        fmt = NumeralTickFormatter(format='0,0')
        slider = Slider(start=start, end=end, step=step, title=title,
                        format=fmt, **slider_kwargs)
    else:
        slider = Slider(start=start, end=end, step=step, title=title,
                        **slider_kwargs)

    slider.width = slider_width

    # NOTE: although NumericInput can handle a high/low limit, it
    # offers no feedback to the user when it does.  Since some of our
    # inputs are capped and others open-ended, we use the js callbacks
    # below to enforce the range limits, if any.
    text_input = NumericInput(width=64, value=value,
                              format=fmt,
                              mode='float' if is_float else 'int')

    # Custom range enforcement with alert messages
    if max_value is not None:
        text_input.js_on_change('value', CustomJS(
            args=dict(inp=text_input),
            code="""
                if (%s inp.value > %s) {
                    alert('Maximum is %s');
                    inp.value = %s;
                }
            """ % ("inp.value != null && " if allow_none else "", max_value, max_value, max_value)))
    if min_value is not None:
        text_input.js_on_change('value', CustomJS(
            args=dict(inp=text_input),
            code="""
                if (%s inp.value < %s) {
                    alert('Minimum is %s');
                    inp.value = %s;
                }
            """ % ("inp.value != null && " if allow_none else "", min_value, min_value, min_value)))

    component = row(slider, text_input, css_classes=["text_slider_%s" % attr, ])

    def _input_check(val):
        # Check if the value is viable as an int or float, according to our type
        if ((not is_float) and isinstance(val, int)) or (is_float and isinstance(val, float)):
            return True
        if val is None and not allow_none:
            return False
        if val is None and allow_none:
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
                text_input.value = old
            return
        if new is not None and old != new:
            if is_float:
                ival = float(new)
            else:
                ival = int(new)
            if ival > slider.end and not max_value:
                slider.end = ival
            if ival < slider.end and end < slider.end:
                slider.end = max(end, ival)
            if 0 <= ival < slider.start and min_value is None:
                slider.start = ival
            if ival > slider.start and start > slider.start:
                slider.start = min(ival, start)
            if slider.start <= ival <= slider.end:
                slider.value = ival
            slider.show_value = True
        elif new is None:
            slider.show_value = False

    def update_text_input(attrib, old, new):
        # Update the text input
        if new != old:
            text_input.value = new

    def handle_value(attrib, old, new):
        # Handle a new value and set the registered object/attribute accordingly
        # Also updates the slider and calls the registered handler function, if any
        if obj and attr:
            try:
                if not hasattr(obj, attr) and isinstance(obj, dict):
                    obj[attr] = new
                else:
                    obj.__setattr__(attr, new)
            except FieldValidationError:
                # reset textbox
                text_input.remove_on_change("value", handle_value)
                text_input.value = old
                text_input.on_change("value", handle_value)
            else:
                update_slider(attrib, old, new)
        if handler:
            if new is not None:
                handler(new)
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


def connect_region_model(fig, region_model):
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
    region_model : :class:`~geminidr.interactive.interactive.GIRegionModel`
        Band model to add view for
    """
    # If we have regions or apertures to show, show them
    if region_model:
        GIRegionView(fig, region_model)

    # This is a workaround for a bokeh bug.  Without this, things like the background shading used for
    # apertures and regions will not update properly after the figure is visible.
    fig.js_on_change('center', CustomJS(args=dict(plot=fig),
                                        code="plot.properties.renderers.change.emit()"))


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
    def __init__(self, domain=None):
        # Right now, the region model is effectively stateless, other
        # than maintaining the set of registered listeners.  That is
        # because the regions are not used for anything, so there is
        # no need to remember where they all are.  This is likely to
        # change in future and that information should likely be
        # kept in here.
        self.region_id = 1
        self.listeners = list()
        self.regions = dict()
        if domain:
            self.min_x = domain[0]
            self.max_x = domain[1]
        else:
            self.min_x = None
            self.max_x = None

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

    def clear_regions(self):
        """
        Deletes all regions.
        """
        for region_id in self.regions.keys():
            for listener in self.listeners:
                listener.delete_region(region_id)
        self.regions = dict()

    def load_from_tuples(self, tuples):
        self.clear_regions()
        self.region_id = 1

        def constrain_min(val, min):
            if val is None:
                return min
            if min is None:
                return val
            return max(val, min)
        def constrain_max(val, max):
            if val is None:
                return max
            if max is None:
                return val
            return min(val, max)
        for tup in tuples:
            start = tup.start
            stop = tup.stop
            start = constrain_min(start, self.min_x)
            stop = constrain_max(stop, self.max_x)
            start = constrain_max(start, self.max_x)
            stop = constrain_min(stop, self.min_x)
            self.adjust_region(self.region_id, start, stop)
            self.region_id = self.region_id + 1
        self.finish_regions()

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
        if self.regions[region_id]:
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
            if (b[0] is None or b[0] <= x) and (b[1] is None or x <= b[1]):
                return True
        return False

    def build_regions(self):

        def none_cmp(x, y):
            if x is None and y is None:
                return 0
            if x is None:
                return -1
            if y is None:
                return 1
            return x - y

        def region_sorter(a, b):
            retval = none_cmp(a[0], b[0])
            if retval == 0:
                retval = none_cmp(a[1], b[1])
            return retval

        def deNone(val, offset=0):
            return '' if val is None else val + offset

        if self.regions is None or len(self.regions.values()) == 0:
            return ''

        sorted_regions = list()
        sorted_regions.extend(self.regions.values())
        sorted_regions.sort(key=cmp_to_key(region_sorter))
        return ', '.join(['{}:{}'.format(deNone(b[0], offset=1), deNone(b[1]))
                          for b in sorted_regions])


class RegionHolder:
    """
    Used by `~geminidr.interactive.interactive.GIRegionView` to track start/stop
    independently of the bokeh Annotation since we want to support `None`.

    We need to know if the start/stop values are a specific value or `None`
    which is open ended left/right.

    Not used outside the `interactive` module.
    """
    def __init__(self, annotation, start, stop):
        self.annotation = annotation
        self.start = start
        self.stop = stop


class GIRegionView(GIRegionListener):
    """
    View for the set of regions to show then in a figure.

    Not used outside the `interactive` module.
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
    """
    Widget used to create/edit fitting Regions.

    Parameters
    ----------
    region_model : GIRegionModel
        Class that connects this element to the regions on plots.
    """
    def __init__(self, region_model):
        self.text_input = TextInput(
            title="Regions (i.e. 101:500,511:900,951: Press 'Enter' to apply):",
            max_width=600,
            sizing_mode="stretch_width",
            width_policy="max",
        )
        self.text_input.value = region_model.build_regions()
        self.region_model = region_model
        self.region_model.add_listener(self)
        self.text_input.on_change("value", self.handle_text_value)

        self.error_message = Div(text="<b> <span style='color:red'> "
                                      "  Please use comma separated : delimited "
                                      "  values (i.e. 101:500,511:900,951:). "
                                      " Negative values are not allowed."
                                      "</span></b>")

        self.error_message.visible = False
        self.widget = column(self.text_input, self.error_message)
        self.handling = False

    def adjust_region(self, region_id, start, stop):
        pass

    def delete_region(self, region_id):
        self.text_input.value = self.region_model.build_regions()

    def finish_regions(self):
        self.text_input.value = self.region_model.build_regions()

    @staticmethod
    def standardize_region_text(region_text):
        """
        Remove spaces near commas separating values, and removes
        leading/trailing commas in string.

        Parameters
        ----------
        region_text : str
            Region text to clean up.

        Returns
        -------
        str : Clean region text.
        """
        # Handles when deleting last region
        region_text = '' if region_text is None else region_text

        # Replace " ," or ", " with ","
        region_text = re.sub(r'[ ,]+', ',', region_text)

        # Remove comma if region_text starts with ","
        region_text = re.sub(r'^,', '', region_text)

        # Remove comma if region_text ends with ","
        region_text = re.sub(r',$', '', region_text)

        return region_text

    def handle_text_value(self, attr, old, new):
        """
        Handles the new text value inside the Text Input field. The
        (attr, old, new) callback signature is a requirement from Bokeh.

        Parameters
        ----------
        attr :
            Attribute that would be changed. Not used here.
        old : str
            Old attribute value. Not used here.
        new : str
            New attribute value. Used to update regions.
        """
        if not self.handling:
            self.handling = True
            region_text = self.standardize_region_text(new)
            current = self.region_model.build_regions()

            # Clean up regions if text input is empty
            if not region_text or region_text.isspace():
                self.region_model.clear_regions()
                self.text_input.value = ""
                current = None
                region_text = None
                self.region_model.finish_regions()

            if current != region_text:
                unparseable = False
                try:
                    parse_user_regions(region_text)
                except ValueError:
                    unparseable = True
                if not unparseable and re.match(r'^((\d+:|\d+:\d+|:\d+)(,\d+:|,\d+:\d+|,:\d+)*)$|^ *$', region_text):
                    self.region_model.load_from_string(region_text)
                    self.text_input.value = self.region_model.build_regions()
                    self.error_message.visible = False
                else:
                    self.text_input.value = current
                    if 'region_input_error' not in self.text_input.css_classes:
                        self.error_message.visible = True
            else:
                self.error_message.visible = False

            self.handling = False

    def get_widget(self):
        return self.widget


class TabsTurboInjector:
    """
    This helper class wraps a bokeh Tabs widget
    and improves performance by dynamically
    adding and removing children from the DOM
    as their tabs are un/selected.

    There is a moment when the new tab is visually
    blank before the contents pop in, but I think
    the tradeoff is worth it if you're finding your
    tabs unresponsive at scale.
    """
    def __init__(self, tabs: bm.layouts.Tabs):
        """
        Create a tabs turbo helper for the given bokeh Tabs.

        :param tabs: :class:`~bokeh.model.layout.Tabs`
            bokeh Tabs to manage
        """
        if tabs.tabs:
            raise ValueError("TabsTurboInjector expects an empty Tabs object")

        self.tabs = tabs
        self.tab_children = list()
        self.tab_dummy_children = list()

        for i, tab in enumerate(tabs.tabs):
            self.tabs.append(tab)
            self.tab_children.append(tab.child)
            self.tab_dummy_children.append(row())

            if i != tabs.active:
                tab.child = self.tab_dummy_children[i]

        tabs.on_change('active', self.tabs_callback)

    def add_tab(self, child: Instance(bm.layouts.LayoutDOM), title: str):
        """
        Add the given bokeh widget as a tab with the given title.

        This child widget will be tracked by the Turbo helper.  When the tab
        becomes active, the child will be placed into the tab.  When the
        tab is inactive, it will be cleared to improve performance.

        :param child: :class:~bokeh.core.properties.Instance(bokeh.models.layouts.LayoutDOM)
            Widget to add as a panel's contents
        :param title: str
            Title for the new tab
        """
        tab_dummy = row(Div(),)
        tab_child = child

        self.tab_children.append(child)
        self.tab_dummy_children.append(tab_dummy)

        if self.tabs.tabs:
            self.tabs.tabs.append(bm.Panel(child=row(tab_dummy,), title=title))
        else:
            self.tabs.tabs.append(bm.Panel(child=row(tab_child,), title=title))

    def tabs_callback(self, attr, old, new):
        """
        This callback will be used when the tab selection changes.  It will clear the DOM
        of the contents of the inactive tab and add back the new tab to the DOM.

        :param attr: str
            unused, will be ``active``
        :param old: int
            The old selection
        :param new: int
            The new selection
        """
        if old != new:
            def clear_old_tab():
                self.tabs.tabs[old].child.children[0] = self.tab_dummy_children[old]
            # clear the old tab via an event on the UI loop
            # we don't want to do it right now - wait until the tab change has happened
            do_later(clear_old_tab)
            self.tabs.tabs[new].child.children[0] = self.tab_children[new]


class UIParameters:
    """
    Holder class for the set of UI-adjustable parameters
    """
    def __init__(self, config: Config = None, extras: dict = None, reinit_params: list = None,
                 title_overrides: dict = None):
        """
        Create a UIParameters set of parameters for the UI.

        This object holds a collection of parameters.  These are used for the visualizer to
        provide user interactivity of the inputs.  Although we track a `value` here, note that
        in some cases the UI may provide multiple tabs with distinct inputs.  In that case,
        it is up to the visualizer to track changes to these inputs on a per tab basis itself.

        Parameters
        ----------
        :config: :class:`~gempy.library.config.Config`
            DRAGONS primitive configuration
        :extras: dict
            Dictionary of names to new Fields to track
        :reinit_params: list
            List of names of configuration fields to show in the reinit panel
        :titles_overrides: dict
            Dictionary of overrides for labeling the fields in the UI
        """
        self.fields = dict()
        self.values = dict()
        self.reinit_params = reinit_params
        self.titles = dict()

        if config:
            for fname, field in config._fields.items():
                self.fields[fname] = field
                self.values[fname] = getattr(config, fname)
        if extras:
            for fname, field in extras.items():
                self.fields[fname] = field
                self.values[fname] = field.default

        # Parse the titles once and be done, grab initial values
        for fname, field in self.fields.items():
            if title_overrides and fname in title_overrides:
                title = title_overrides[fname]
            else:
                title = field.doc.split('\n')[0]
            self.titles[fname] = title

    def update_values(self, **kwargs):
        """
        Update values in the config from the supplied arguments
        """
        for k, v in kwargs.items():
            self.values[k] = v

    def __getattr__(self, attr):
        """
        Provides the same interface to get parameter values as Config

        Parameters
        ----------
        attr : str
            Name of config parameter to return the value of
        """
        try:
            return self.values[attr]
        except:
            return object.__getattribute__(self, attr)


def do_later(fn):
    """
    Helper method to queue work to be done on the bokeh UI thread.

    When actiona happen as a result of a key press, for instance, this comes in
    on a different thread than bokeh UI is operating on.  Performing any UI
    impacting changes on this other thread can cause issues.  Instead, wrap
    the desired changes in a function and pass it in here to be run on the UI
    thread.

    This works by referencing the active singleton Visualizer, so that
    code doesn't have to all track the visualizer.

    :param fn:
    :return:
    """
    _visualizer.do_later(fn)
