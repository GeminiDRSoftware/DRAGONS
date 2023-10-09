"""
Support for keyboard controls.

This is where we bolt on key controls onto Bokeh, which requires
some hackery since they are so unresponsive on this issue.

The controller has a concept of the current "state" of interaction.
It also receives a continuous update of x/y coordinates and
mouse clicks.  In it's default, inactive state, it is waiting for
the user to select a Task by it's key code.  Once a Task is selected,
it becomes the active Task and is handed further keys, mouse moves
and clicks.  Tasks may in turn spawn sub-Tasks to refine their
behavior or may manage some state on their own.  When the task is
finished (or canceled), the controller is returned to the top-level
inactive state.
"""
from abc import ABC, abstractmethod

from bokeh.events import (
    PointEvent,
    MouseEnter,
    MouseLeave,
)

__all__ = ["CONTROLLER", "Controller", "Handler"]

from bokeh.models import CustomJS

# This is the active controller.  It is activated when it's attached figure
# sees the mouse enter it's view.

# Controller instances will set this to listen to key presses.  The bokeh
# server will use this to send keys it recieves from the clients.  Everyone
# else should leave it alone!
CONTROLLER = None


# This is used to track if we already have a pending action to update the mouse
# position.  This is used to effect throttling.  When there is a mouse move, an
# action to respond to the mouse position is triggered in the future and this
# is set True.  Subsequent mouse moves will see we already have the action
# pending and do nothing.  Then when the action triggers it takes the current
# (future) mouse position and sets this back to False.
_PENDING_HANDLE_MOUSE = False


class Controller(object):
    """
    Controller is an interaction manager for our custom interfaces.

    We want rich client-server driven interactions and bokeh does a poor job
    of this by itself.  These new interactions are instead routed through here.
    In the case of key presses, this is routed in via a custom URL endpoint in
    the bokeh server from a javascript key listener in index.html.

    This routing can mean that some operations have to happen within a call to
    `add_next_tick_callback`.  That is because our key press is coming in via
    a different path than the normal bokeh interactivity.

    The controller maintains a set of :class:`~Task` instances.
    One :class:`~Task` is operating at a time. An active task receives all the
    key presses and when it is done, it returns control to the
    :class:`~Controller`.  The :class:`~Tasks` are also able to update the help
    text to give contextual help.
    """

    def __init__(
        self,
        fig,
        aperture_model,
        region_model,
        help_text,
        mask_handlers=None,
        domain=None,
        handlers=None,
        helpintrotext=None,
    ):
        """
        Create a controller to manage the given aperture and region models on
        the given GIFigure.

        Parameters
        ----------
        fig : :class:`~Figure`
            plot to attach controls to.
        aperture_model : :class:`GIApertureModel`
            model for apertures for this plot/dataset, or None.
        region_model : :class:`GIRegionModel`
            model for regions for this plot/dataset, or None.
        help_text : :class:`Div`
            div to update text in to provide help to the user.
        mask_handlers : None or tuple of {2,3} functions
            The first two functions handle mask/unmask commands, the third
            (optional) function handles 'P' point mask requests.
        domain : tuple of 2 numbers, or None
            The domain of the data being handled, used for region editing to
            constrain the values.
        handlers : list of `~geminidr.interactive.controls.Handler`
            List of handlers for key presses that are always active, not tied
            to a task.
        helpintrotext : str
            HTML to show at the top of the controller help (gray text block),
            or None for a default message.
        """

        fig.js_on_event(
            MouseEnter, CustomJS(code="window.controller_keys_enabled = true;")
        )
        fig.js_on_event(
            MouseLeave,
            CustomJS(code="window.controller_keys_enabled = false;"),
        )

        # set the class for the help_text div so we can have a common style
        help_text.css_classes.append("controller_div")

        self.aperture_model = aperture_model

        self.helpmaskingtext = ""

        if helpintrotext is not None:
            self.helpintrotext = f"{helpintrotext}<br/><br/>\n"

        else:
            self.helpintrotext = (
                "While the mouse is over the plot, choose from the "
                "following commands:<br/><br/>\n"
            )

        self.helptooltext = ""
        self.helptext = help_text
        self.enable_user_masking = True if mask_handlers else False

        self.handlers = dict()
        if mask_handlers:
            if len(mask_handlers) != 2:
                raise ValueError(
                    "Must pass tuple (mask_fn, unmask_fn) to mask_handlers "
                    "argument of Controller"
                )

            # pylint: disable=unused-argument
            def _mask(key, x, y):
                handler = mask_handlers[0]
                dx = (
                    (self.fig.x_range.end - self.fig.x_range.start)
                    / float(self.fig.inner_width)
                )

                dy = (
                    (self.fig.y_range.end - self.fig.y_range.start)
                    / float(self.fig.inner_height)
                )

                return handler(x, y, dx / dy)

            self.register_handler(
                Handler("m", "Mask selected/closest", _mask)
            )

            # pylint: disable=unused-argument
            def _unmask(key, x, y):
                handler = mask_handlers[1]
                dx = (
                    (self.fig.x_range.end - self.fig.x_range.start)
                    / float(self.fig.inner_width)
                )

                dy = (
                    (self.fig.y_range.end - self.fig.y_range.start)
                    / float(self.fig.inner_height)
                )

                return handler(x, y, dx / dy)

            self.register_handler(
                Handler("u", "Unmask selected/closest", _unmask)
            )

        if handlers:
            for handler in handlers:
                self.register_handler(handler)

        self.update_helpmaskingtext()

        self.tasks = dict()
        if aperture_model:
            self.tasks["a"] = ApertureTask(aperture_model, help_text, fig)
        if region_model:
            self.tasks["r"] = RegionTask(
                region_model, help_text, domain=domain
            )
        self.task = None
        self.x = None
        self.y = None
        # we need to always know where the mouse is in case someone
        # starts an Aperture or Band
        if aperture_model or region_model or mask_handlers or handlers:
            fig.on_event("mousemove", self.on_mouse_move)
            fig.on_event("mouseenter", self.on_mouse_enter)
            fig.on_event("mouseleave", self.on_mouse_leave)
        self.fig = fig
        self.set_help_text(None)

    def register_handler(self, handler):
        """
        Add the passed handler to the controller's list of known handlers
        for always on key commands.  This is also the mechanism that
        handles mask/unmask events.

        Only one handler can be registered for each key.  Please
        be careful not to register a handler for a key needed by one
        of the tasks.

        Parameters
        ----------
        handler : :class:`~geminidr.interactive.controls.Handler`
            Handler to add to the controller
        """
        if handler.key in self.handlers:
            raise ValueError(f"Key {handler.key} already registered")

        self.handlers[handler.key] = handler

    def update_helpmaskingtext(self):
        """
        Update the help preamble text for keys that are
        always available, starting with the masking.
        """
        if "m" in self.handlers:
            self.helpmaskingtext = (
                "<b>Masking</b> <br/> "
                'To mark or unmark one point at a time, select "Tap" on the '
                "toolbar on the right side of the plot.  To mark/unmark a "
                'group of point, activate "Box Select" on the toolbar.<br/>'
                "<b>M</b> - Mask selected/closest<br/>"
                "<b>U</b> - Unmask selected/closest<br/></br>"
            )

        if self.handlers - ["m", "u"]:
            if "m" in self.handlers:
                self.helpmaskingtext += "<b>Other Commands</b><br/>"

            else:
                self.helpmaskingtext += "<b>Commands</b><br/>"

            for k, v in self.handlers.items():
                if k not in ["u", "m"]:
                    self.helpmaskingtext += (
                        f"<b>{k.upper()}</b> - {v.description}<br/>"
                    )

            self.helpmaskingtext += "<br/>"

    def set_help_text(self, text=None):
        """
        Set the text in the help area.

        This updates the text in the help div for the user to
        know the most commonly used commands for the current
        context.

        Parameters
        ----------
        text : str
            html to display in the div
        """
        if text is not None:
            ht = self.helpintrotext + self.helpmaskingtext + text
        else:
            if self.tasks or self.handlers or self.enable_user_masking:
                # TODO somewhat editor-inheritance vs on enter function below,
                # refactor accordingly
                ht = self.helpintrotext + self.helpmaskingtext
                if len(self.tasks) == 1:
                    task = next(iter(self.tasks.values()))
                    ht = ht + task.helptext()
                else:
                    for key, task in sorted(self.tasks.items()):
                        ht = ht + "<b>%s</b> - %s<br/>\n" % (
                            key,
                            task.description(),
                        )
            else:
                ht = ""

        # This has to be done via a callback.  During the key press, we are
        # outside the context of the widget's bokeh document
        if self.helptext.document:
            # we now have an associated document, need to do this inside that
            # context
            self.helptext.document.add_next_tick_callback(
                lambda: self.helptext.update(text=ht)
            )
        else:
            # no document, we can update directly
            self.helptext.update(text=ht)

    # pylint: disable=unused-argument
    def on_mouse_enter(self, event):
        """
        Handle the mouse entering our related figure by activating this
        controller.

        Parameters
        ----------
        event
            the mouse event from bokeh, unused
        """
        global CONTROLLER
        CONTROLLER = self
        if len(self.tasks) == 1:
            # for k, v in self.tasks.items():
            #     self.task = v
            self.task = next(iter(self.tasks.values()))
            ht = self.task.helptext()
            self.set_help_text(ht)
            self.task.start(self.x, self.y)
        else:
            # show selection of available tasks
            self.set_help_text(None)

    def on_mouse_leave(self, event):
        """
        Response to the mouse leaving the connected figure.

        When the mouse leaves, we tell any active task to finish itself
        and we de-register ourselves as the active controller.  We will
        not receive further events until a subsequent mouse enter event.

        Parameters
        ----------
        event
            the mouse event from bokeh, unused
        """
        global CONTROLLER
        if self == CONTROLLER:
            self.set_help_text(None)
            if self.task:
                self.task.stop()
                self.task = None
            CONTROLLER = None

    def on_mouse_move(self, event: PointEvent):
        """
        Respond to mouse movement within the figure.

        This is a convenience method where we extract the
        x and y coordinates and call our more generic
        `handle_mouse` method.

        Parameters
        ----------
        event : :class:`~bokeh.events.PointEvent`
            the event from bokeh
        """
        self.x = event.x
        self.y = event.y
        self.handle_mouse(self.x, self.y)

    def handle_key(self, key):
        """
        Handle a key press.

        We respond to key presses by passing them to
        the active task.  If there is no active task,
        we check if this key is associated with a
        particular task and, if so, activate that task,

        Parameters
        ----------
        key : char
            Key that was pressed, such as 'a'
        """

        def _ui_loop_handle_key(_key):
            if _key in self.handlers:
                self.handlers[_key].handle(_key, self.x, self.y)

            elif self.task:
                if self.task.handle_key(_key):
                    if len(self.tasks) > 1:
                        # only if we have multiple tasks, otherwise no point in
                        # offering 1 task option
                        self.task = None
                        self.set_help_text()

            elif _key in self.tasks:
                self.task = self.tasks[_key]
                self.set_help_text(self.task.helptext())
                self.task.start(self.x, self.y)

        if self.fig.document:
            # we now have an associated document, need to do this inside that
            # context
            self.fig.document.add_next_tick_callback(
                lambda: _ui_loop_handle_key(_key=key)
            )

    def handle_mouse(self, x, y):
        """
        Handle mouse movement

        If there is an active task, we let it know aobut the
        mouse move event.  Tasks will handle this in their
        own custom way, perhaps responding to a drag or just
        knowing the mouse position in the event the user
        creates an aperture.

        Parameters
        ----------
        x : float
            x coordinate in data space
        y : float
            y coordinate in data space
        """
        self.x = x
        self.y = y
        global _PENDING_HANDLE_MOUSE
        if not _PENDING_HANDLE_MOUSE:
            _PENDING_HANDLE_MOUSE = True
            if self.fig.document is not None:
                self.fig.document.add_timeout_callback(
                    self.handle_mouse_callback, 100
                )
            else:
                self.handle_mouse_callback()

    def handle_mouse_callback(self):
        global _PENDING_HANDLE_MOUSE
        _PENDING_HANDLE_MOUSE = False
        if self.task:
            if self.task.handle_mouse(self.x, self.y):
                self.task = None
                self.set_help_text()


class Task(ABC):
    """A Task is a general concept of some interactive behavior where we
    also want keyboard support.

    A task may be connected to a top-level Controller by a key command.
    Once a task is active, the controller will send mouse and key events
    to it.  Tasks may share key presses with eachother as only one is
    active at a time.  Tasks are also convenient for holding state.
    For simple single-key actions that you want to always have active,
    see `~geminidr.interactive.controls.Handler`
    """

    @abstractmethod
    def handle_key(self, key):
        """Called when the task is active and we have a key press."""

    @abstractmethod
    def handle_mouse(self, x, y):
        """Called when we have a mouse move and the task is active.

        Parameters
        ----------
        x : float
            x coordinate of mouse in data space
        y : float
            y coordinate of mouse in data space

        """

    def helptext(self):
        """Override to provide HTML help text to display when this task is
        active.

        Returns
        -------
            The html to display
        """
        return ""


class ApertureTask(Task):
    """
    Task for controlling apertures.
    """

    def __init__(self, aperture_model, helptext, fig):
        """
        Create aperture task for the given model.

        Parameters
        ----------
        aperture_model : :class:`GIApertureModel`
            The aperture model to operate on
        """
        # location, width, left, right for placing location, width (both)
        # or left/right side
        self.mode = ""
        self.aperture_model = aperture_model
        self.aperture_id = None
        self.last_x = None
        self.last_y = None
        self.fig = fig
        self.helptext_area = helptext
        self.helptext_area.text = self.helptext()

    def start(self, x, y):
        """
        Start defining an aperture from the given coordinate.

        This method starts an aperture definition with the current coordinates.
        The x value will define the center of the aperture.  After that, as the
        mouse is moved, it will define the edge of the aperture with an equally
        wide edge on the other side.

        Parameters
        ----------
        x : float
            Current x coordinate, used to set the center of the aperture
        y : float
            Current y coordinate, not used
        """
        self.last_x = x
        self.last_y = y
        self.aperture_id = None

    def stop(self):
        """
        Stop the task, which stops the aperture if we're in the middle
        of workingon one.
        """
        self.stop_aperture()

    # pylint: disable=unused-argument
    def start_aperture(self, x, y):
        """
        Create a new aperture at this x coordinate.

        The aperture is centered at x with a width of 0.  Further
        mouse movement will set the width to the mouse location.

        Parameters
        ----------
        x, y : float
            x, y coordinate in data space
        """
        self.aperture_id = self.aperture_model.add_aperture(x, x, x)
        self.mode = "width"

    def stop_aperture(self):
        """Stop updating the current aperture. This causes the interactivity
        to end.
        """
        self.aperture_id = None
        self.mode = ""

    def handle_key(self, key):
        """
        Handle a key press.

        This will listen for a press of 'a' to tell the task to stop updating
        the aperture.

        Parameters
        ----------
        key : char
            key that was pressed

        Returns
        -------
            True if the task is finished and the controller should take over,
            False if we are not done with the Task
        """
        keymodes = {"[": "left", "]": "right", "l": "location"}

        def _handle_key():
            if self.aperture_id is None or self.mode == "":
                # get closest one
                self.aperture_id = self.aperture_model.find_closest(
                    self.last_x, self.fig.x_range.start, self.fig.x_range.end
                )
                if self.aperture_id is None:
                    return False
                self.mode = keymodes[key]
                return False
            else:
                self.stop_aperture()
                return True

        if key in "[l]":
            return _handle_key()
        elif key == "s":
            self.aperture_id = self.aperture_model.find_closest(
                self.last_x,
                self.fig.x_range.start,
                self.fig.x_range.end,
                prefer_selected=False,
            )
            self.aperture_model.select_aperture(self.aperture_id)
            return False
        elif key == "c":
            self.aperture_id = None
            self.aperture_model.select_aperture(None)
            return False
        elif key == "a":
            if self.aperture_id is None:
                self.start_aperture(self.last_x, self.last_y)
                return False
            else:
                self.stop_aperture()
                return True
        elif key == "f":
            if self.aperture_id is None:
                self.aperture_model.find_peak(self.last_x)
                return True
        elif key == "d":
            if self.aperture_id is None:
                # get closest one
                self.aperture_id = self.aperture_model.find_closest(
                    self.last_x, self.fig.x_range.start, self.fig.x_range.end
                )
                if self.aperture_id is None:
                    return False
            self.aperture_model.delete_aperture(self.aperture_id)
            self.stop_aperture()
            return True
        return False

    def handle_mouse(self, x, y):
        """
        Handle a mouse movement.  We respond to the mouse by continuously
        updating the active aperture around it's center point to a width to
        match the mouse position.

        Parameters
        ----------
        x, y : float
            mouse x, y coordinate in data space
        """
        # we are in aperture mode
        if self.aperture_id:
            if (
                self.aperture_id
                not in self.aperture_model.aperture_models.keys()
            ):
                pass
            model = self.aperture_model.aperture_models[self.aperture_id]
            location = model.source.data["location"][0]

            if self.mode == "width":
                width = abs(location - x)
                model.update_values(
                    start=location - width, end=location + width
                )
            elif self.mode == "left":
                if x < location:
                    model.update_values(start=x)
            elif self.mode == "right":
                if x > location:
                    model.update_values(end=x)
            elif self.mode == "location":
                diff = x - location
                model.update_values(
                    location=x,
                    start=model.source.data["start"][0] + diff,
                    end=model.source.data["end"][0] + diff,
                )

        self.last_x = x
        self.last_y = y
        return False

    def description(self):
        """
        Get the description for this task, to use in the help Div.

        Returns
        -------
        str : help html text for display
        """
        return "Edit <b>apertures</b> interactively"

    def helptext(self):
        """
        Get the detailed help of key commands available while this
        task is active.  This is shown in the help Div when the task
        is active.

        Returns
        -------
        str : help html for available commands when task is active.
        """
        return """
        <b>A</b> to start the aperture or set the value<br/>
        <b>S</b> to select an existing aperture<br/>
        <b>C</b> to clear the selection<br/>
        <b>F</b> to find a peak close to the cursor<br/>
        <b>[</b> to edit the left edge of selected or closest<br/>
        <b>]</b> to edit the right edge of selected or closest<br/>
        <b>L</b> to edit the location of selected or closest<br/>
        <b>D</b> to delete the selected or closest aperture
        """


class RegionTask(Task):
    """
    Task for operating on the regions.
    """

    def __init__(self, region_model, helptext, domain=None):
        """
        Create a region task for the given :class:`GIRegionModel`

        Parameters
        ----------
        region_model : :class:`GIRegionModel`
            The region model to operate on with this task
        """
        self.region_model = region_model
        self.region_edge = None
        self.region_id = None
        self.helptext_area = helptext
        self.last_x = None
        self.last_y = None
        if domain:
            self.min_x = domain[0]
            self.max_x = domain[1]
        else:
            self.min_x = None
            self.max_x = None

    def start(self, x, y):
        """
        Start a region task with the current mouse position.

        Parameters
        ----------
        x : float
            x coordinate of the mouse to start with
        y : float
            y coordinate of the mouse, unused

        """
        self.last_x = x
        if self.min_x is not None and x is not None:
            self.last_x = max(self.last_x, self.min_x)
        if self.max_x is not None and x is not None:
            self.last_x = min(self.last_x, self.max_x)
        self.last_y = y

    def start_region(self):
        """
        Start a new region with the current mouse position.

        This starts a region with one edge at the current x coordinate.
        """
        x = self.last_x

        (region_id, start, end) = self.region_model.find_region(x)

        if region_id is not None:
            self.region_id = region_id

            if (x - start) < (end - x):
                self.region_edge = end

            else:
                self.region_edge = start

            self.region_model.adjust_region(
                self.region_id, x, self.region_edge
            )

        else:
            self.region_edge = x
            self.region_id = self.region_model.region_id
            self.region_model.region_id += 1

    def stop(self):
        """
        Stop the task.
        :return:
        """
        if self.region_id is not None:
            self.stop_region()

    def stop_region(self):
        """
        Stop modifying the current region.
        """
        self.region_edge = None
        self.region_id = None
        self.region_model.finish_regions()

    def handle_key(self, key):
        """
        Handle a key press.

        Parameters
        ----------
        key : char
            key that was pressed by the user.

        """
        if key == "b" or key == "r":
            if self.region_id is None:
                self.start_region()
                self.update_help()

                return False

            else:
                self.stop_region()
                self.update_help()

                return True

        if key == "e":
            if self.region_edge is None:
                _id, _edge = self.region_model.closest_region(self.last_x)
                self.region_id, self.region_edge = _id, _edge

            else:
                self.stop_region()

            self.update_help()

            return False

        if key == "d":
            if self.region_id is not None:
                self.region_model.delete_region(self.region_id)
                self.stop()
                self.update_help()

                return True

            else:
                _id, _edge = self.region_model.closest_region(self.last_x)
                region_id, _ = _id, _edge

                if region_id is not None:
                    self.region_model.delete_region(region_id)

                self.update_help()

                return True

        if key == "*":
            if self.region_id is not None and self.region_edge is not None:
                if self.last_x < self.region_edge:
                    # we're left of the anchor, so we want [None, region_edge]
                    self.region_model.adjust_region(
                        self.region_id, None, self.region_edge
                    )

                else:
                    # we're right of the anchor, so we want [region_edge, None]
                    self.region_model.adjust_region(
                        self.region_id, self.region_edge, None
                    )

                self.stop_region()
                self.update_help()

        return False

    def handle_mouse(self, x, y):
        """
        Handle a mouse cursor move within the view.

        This receives updates for the mouse movement as long as
        the task is active.  We modify the other edge of the
        active region to this x value.

        Parameters
        ----------
        x : float
            x coordinate of mouse in data space
        y : float
            y coordinate of mouse in data space

        """
        self.last_x = x
        self.last_y = y

        if self.min_x is not None:
            self.last_x = max(self.last_x, self.min_x)

        if self.max_x is not None:
            self.last_x = min(self.last_x, self.max_x)

        # we are in region mode
        if self.region_id is not None:
            start = self.last_x
            end = self.region_edge
            self.region_model.adjust_region(self.region_id, start, end)

        return False

    def description(self):
        """
        Returns the description to use when offering this task from the
        top-level controller.

        Returns
        -------
            str description of the task
        """
        return "create a <b>region</b> with edge at cursor"

    def update_help(self):
        """
        Update the controller help text when a region is active
        or not.
        """
        if self.region_id is not None:
            CONTROLLER.set_help_text(
                """Drag to desired region width.<br/>\n
                  <b>R</b> to set the region<br/>\n
                  <b>D</b> to delete/cancel the current region<br/>
                  <b>*</b> to extend to maximum on this side
                  """
            )

        else:
            CONTROLLER.set_help_text(
                """<b> Edit Regions: </b><br/>\n
                  <b>R</b> to start a new region<br/>\n
                  <b>E</b> to edit nearest region<br/>\n
                  <b>D</b> to delete the nearest region
                  """
            )

    def helptext(self):
        """
        Returns the help for this task for when it is active

        Returns
        -------
            str HTML help text for the task
        """
        s = (
            """<b> Edit Regions </b> <br/>\n
            <b>R</b> to start a new region or set the edge if editing<br/>\n
            <b>E</b> to edit nearest region<br/>\n
            <b>D</b> to delete/cancel the current/nearest region
            """
        )

        return s


class Handler:
    """
    A Handler is a top-level key handler for the controller.

    Handlers are always active and only handle a single key.
    The Handler function will be called with the key to
    allow for a single function to be used in multiple
    Handlers without ambiguity.

    Parameters
    ----------
    key : str
        Key this handler listens for
    description : str
        Description of handler effect for the help text
    fn : callable
        Function to call with the key
    """

    def __init__(self, key, description, fn):
        self.key = key.lower()
        self.description = description
        self.fn = fn

    def handle(self, key, x, y):
        self.fn(key, x, y)
