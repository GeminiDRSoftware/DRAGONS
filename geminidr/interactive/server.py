from abc import ABC, abstractmethod

# bokeh basics

# bokeh widgets
from bokeh.layouts import row
from bokeh.models import Slider, TextInput
from bokeh.server.server import Server

# numpy

# Offsets, may be updated by bokeh controls


def bkapp(doc):
    """
    Callback for bokeh on server start.

    When the bokeh service starts, it is given this method as the root level
    handler.  When the user's browser opens, the `doc` for that page will be
    passed down through here.

    Parameters
    ----------
    doc
        Document reference for the user's browser tab to hold the user interface.

    Returns
    -------
    none
    """
    global visualizer

    visualizer.visualize(doc)


bokeh_server = None
visualizer = None


def start_server():
    """
    Start the bokeh server.

    This will begin an IO loop to handle user interaction via
    bokeh.  Until the server is explicitly stopped, the method
    will block and this call will not return.

    Returns
    -------
    none
    """
    global bokeh_server

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.
    bokeh_server = Server({'/': bkapp}, num_procs=1)
    bokeh_server.start()

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.

    bokeh_server.io_loop.add_callback(bokeh_server.show, "/")
    bokeh_server.io_loop.start()

    bokeh_server.stop()


class PrimitiveVisualizer(ABC):
    def __init__(self, params, fields):
        self._params = params
        self._fields = {field.name : field for field in fields}

    @abstractmethod
    def visualize(self):
        """
        Perform the visualization.

        This is called via bkapp by the bokeh server and happens
        when the bokeh server is spun up to interact with the user.

        Returns
        -------
        none
        """
        pass

    @abstractmethod
    def result(self):
        """
        Get the result of the visualization.

        Returns
        -------
        Implementation-specific value relevant to the primitive
        """
        return None

    def make_slider_for(self, title, value, step, field, handler):
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
        step : int
            Step size
        field : :class:`~config.Config`
            Type information taken from the primitive.  This is used to
            make good decisions when constructing UI elements
        handler : method
            Function to handle callbacks when value of the slider changes

        Returns
        -------
            :class:`bokeh.models.Slider` slider widget for bokeh interface
        """
        start = min(value, field.min) if field.min else min(value, 0)
        end = max(value, field.max) if field.max else max(10, value*2)
        slider = Slider(start=start, end=end, value=value, step=step, title=title)
        slider.width = 256

        text_input = TextInput()
        text_input.width = 64
        text_input.value = str(value)
        component = row(slider, text_input)

        def update_slider(attr, old, new):
            if old != new:
                ival = int(new)
                if ival > slider.end and not field.max:
                    slider.end = ival
                if 0 <= ival < slider.start and field.min is None:
                    slider.start = ival
                if slider.start <= ival <= slider.end:
                    slider.value = ival

        def update_text_input(attr, old, new):
            if new != old:
                text_input.value = str(new)

        slider.on_change("value", update_text_input, handler)
        text_input.on_change("value", update_slider)
        return component
