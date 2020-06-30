from abc import ABC, abstractmethod

from bokeh.core.property.container import Seq
from bokeh.layouts import row
from bokeh.models import Slider, TextInput, ColumnDataSource


class PrimitiveVisualizerNew(ABC):

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

    def make_slider_for(self, title, value, step, min_value, max_value, handler):
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
        min : int
            Minimum slider value, or None defaults to min(value,0)
        max : int
            Maximum slider value, or None defaults to value*2
        handler : method
            Function to handle callbacks when value of the slider changes

        Returns
        -------
            :class:`bokeh.models.Slider` slider widget for bokeh interface
        """
        start = min(value, min_value) if min_value else min(value, 0)
        end = max(value, max_value) if max_value else max(10, value*2)
        slider = Slider(start=start, end=end, value=value, step=step, title=title)
        slider.width = 256

        text_input = TextInput()
        text_input.width = 64
        text_input.value = str(value)
        component = row(slider, text_input)

        def update_slider(attr, old, new):
            if old != new:
                ival = int(new)
                if ival > slider.end and not max_value:
                    slider.end = ival
                if 0 <= ival < slider.start and min_value is None:
                    slider.start = ival
                if slider.start <= ival <= slider.end:
                    slider.value = ival

        def update_text_input(attr, old, new):
            if new != old:
                text_input.value = str(new)

        slider.on_change("value", update_text_input, handler)
        text_input.on_change("value", update_slider)
        return component


class GIControlListener(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def giupdate(self, data):
        pass


class GICoordsSource:
    def __init__(self):
        self.listeners = list()

    def add_gilistener(self, coords_listener):
        if not isinstance(coords_listener, GICoordsListener):
            raise ValueError("Must pass a GICoordsListener implementation")
        self.listeners.append(coords_listener)

    def ginotify(self, x_coords, y_coords):
        for l in self.listeners:
            l.giupdate(x_coords, y_coords)


class GICoordsListener(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def giupdate(self, x_coords, y_coords):
        pass


class GIScatter(GICoordsListener):
    def __init__(self, fig, x_coords=None, y_coords=None, color="blue", radius=5):
        if x_coords is None:
            x_coords = []
        if y_coords is None:
            y_coords = []
        self.source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.scatter = fig.scatter(x='x', y='y', source=self.source, color=color, radius=radius)

    def giupdate(self, x_coords, y_coords):
        self.source.data = {'x': x_coords, 'y': y_coords}

    def clear_selection(self):
        self.source.selected.update(indices=[])
        # self.source.selected.indices = Seq()

class GILine(GICoordsListener):
    def __init__(self, fig, x_coords=[], y_coords=[], color="red"):
        if x_coords is None:
            x_coords = []
        if y_coords is None:
            y_coords = []
        self.line_source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.line = fig.line(x='x', y='y', source=self.line_source, color=color)

    def giupdate(self, x_coords, y_coords):
        self.line_source.data = {'x': x_coords, 'y': y_coords}
