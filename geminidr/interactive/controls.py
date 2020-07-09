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

from bokeh.events import PointEvent


""" This is the active controller.  It is activated when it's attached figure sees the mouse enter it's view.  """
controller = None


class Controller(object):
    """
    Controller is an interaction manager for our custom interfaces.

    We want rich client-server driven interactions and bokeh does a poor job
    of this by itself.  These new interactions are instead routed through here.
    In the case of key presses, this is routed in via a custom URL endpoint in
    the bokeh server from a javascript key listener in index.html.
    """
    def __init__(self, figure, aperture_model, band_model, helptext):
        self.figure = figure
        self.helptext = helptext
        self.tasks = dict()
        self.tasks['a'] = ApertureTask(aperture_model)
        self.tasks['b'] = BandTask(band_model)
        self.task = None
        self.x = None
        self.y = None
        # we need to always know where the mouse is in case someone
        # starts an Aperture or Band
        figure.on_event('mousemove', self.on_mouse_move)
        figure.on_event('mouseenter', self.on_mouse_enter)
        figure.on_event('mouseleave', self.on_mouse_leave)

        self.sethelptext()

    def sethelptext(self, text=None):
        if text:
            ht = text
        else:
            ht = """While the mouse is over the plot, choose from the following commands:<br/>\n"""
            for key, task in sorted(self.tasks.items()):
                ht = ht + "<b>[%s]</b> - %s<br/>\n" % (key, task.description())

        # This has to be done via a callback.  During the key press, we are outside the context of
        # the widget's bokeh document
        if self.helptext.document:
            # we now have an associated document, need to do this inside that context
            self.helptext.document.add_next_tick_callback(lambda: self.helptext.update(text=ht))
        else:
            self.helptext.text = ht

    def on_mouse_enter(self, event):
        global controller
        controller = self

    def on_mouse_leave(self, event):
        global controller
        if self == controller:
            if self.task:
                self.task.stop()
                self.task = None
            controller = None

    def on_mouse_move(self, event: PointEvent):
        self.x = event.x
        self.y = event.y
        self.handle_mouse(self.x, self.y)

    def handle_key(self, key):
        if self.task:
            if self.task.handle_key(key):
                self.task = None
                self.sethelptext()
        else:
            if key in self.tasks:
                self.task = self.tasks[key]
                self.sethelptext(self.task.helptext())
                self.task.start(self.x, self.y)

    def handle_mouse(self, x, y):
        self.x = x
        self.y = y
        if self.task:
            if self.task.handle_mouse(x, y):
                self.task = None
                self.sethelptext()


class Task(ABC):
    @abstractmethod
    def handle_key(self, key):
        pass

    @abstractmethod
    def handle_mouse(self, x, y):
        pass

    def helptext(self):
        return ""


class ApertureTask(Task):
    def __init__(self, aperture_model):
        self.aperture_model = aperture_model
        self.aperture_center = None
        self.aperture_id = None
        # self.aperture_model.adjust_aperture(1, -1000, -1000)

    def start(self, x, y):
        self.aperture_center = x
        self.aperture_id = self.aperture_model.aperture_id
        self.aperture_model.aperture_id += 1
        # self.aperture_model.adjust_aperture(self.aperture_id, x, x)

    def stop(self):
        self.aperture_center = None
        self.aperture_id = None

    def handle_key(self, key):
        if key == 'a':
            self.stop()
            return True
        return False

    def handle_mouse(self, x, y):
        # we are in aperture mode
        start = x
        end = self.aperture_center + (self.aperture_center-x)
        self.aperture_model.adjust_aperture(self.aperture_id, start, end)
        return False

    def description(self):
        return "create an <b>aperture</b> centered at cursor"

    def helptext(self):
        return """Drag to desired aperture width<br/>\n<b>[a]</b> to set the aperture"""


class BandTask(Task):
    def __init__(self, band_model):
        self.band_model = band_model
        self.band_edge = None
        self.band_id = None

    def start(self, x, y):
        self.band_edge = x
        self.band_id = self.band_model.band_id
        self.band_model.band_id += 1

    def stop(self):
        self.band_edge = None
        self.band_id = None

    def handle_key(self, key):
        if key == 'b':
            self.stop()
            return True
        return False

    def handle_mouse(self, x, y):
        # we are in band mode
        start = x
        end = self.band_edge
        self.band_model.adjust_band(self.band_id, start, end)
        return False

    def description(self):
        return "create a <b>band</b> with edge at cursor"

    def helptext(self):
        return """Drag to desired band width.<br/>\n<b>[b]</b> to set the band"""
