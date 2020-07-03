import sys
from abc import ABC, abstractmethod

from bokeh.core.property.container import Seq
from bokeh.layouts import row
from bokeh.models import Slider, TextInput, ColumnDataSource, BoxAnnotation, Label


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


class BandControls(object):
    def __init__(self, container, band_id, model, start, end):
        self.band_id = band_id
        self.model = model
        self.min_slider = Slider(title="Band %s Start" % band_id, value=0, step=1, start=start, end=end)
        self.max_slider = Slider(title="Band %s End" % band_id, value=0, step=1, start=start, end=end)
        container.children.append(self.min_slider)
        container.children.append(self.max_slider)
        self.min_slider.on_change("value", self.handler)
        self.max_slider.on_change("value", self.handler)

    def handler(self, attr, old, new):
        if old != new:
            min_val = self.min_slider.value
            max_val = self.max_slider.value
            if min_val > max_val:
                swapper = min_val
                min_val = max_val
                max_val = swapper
            self.model.adjust_band(self.band_id, min_val, max_val)


class BandListener(ABC):
    @abstractmethod
    def adjust_band(self, band_id, start, stop):
        pass

    def delete_band(self, band_id):
        pass


class BandModel(object):
    def __init__(self):
        self.listeners = list()

    def add_listener(self, listener):
        if not isinstance(listener, BandListener):
            raise ValueError("must be a BandListener")
        self.listeners.append(listener)

    def adjust_band(self, band_id, start, stop):
        for listener in self.listeners:
            listener.adjust_band(band_id, start, stop)


class GIBands(BandListener):
    def __init__(self, fig, model):
        # example:
        # # a finite region
        # center = BoxAnnotation(top=0.6, bottom=-0.3, left=7, right=12, fill_alpha=0.1, fill_color='navy')
        # p.add_layout(center)

        self.model = model
        model.add_listener(self)
        self.bands = dict()
        self.fig = fig

        # Unfortunately, it seems the annotations MUST exist before plotting
        # So this is a hack until I can work around that somehow
        it_goes_to_eleven = -1 * sys.float_info.max
        for band_id in range(20):
            band = BoxAnnotation(left=it_goes_to_eleven, right=it_goes_to_eleven, fill_alpha=0.1, fill_color='navy')
            self.fig.add_layout(band)
            self.bands[band_id] = band

    def adjust_band(self, band_id, start, stop):
        if band_id in self.bands:
            band = self.bands[band_id]
            band.left = start
            band.right = stop
        else:
            band = BoxAnnotation(left=start, right=stop, fill_alpha=0.1, fill_color='navy')
            self.fig.add_layout(band)
            self.bands[band_id] = band

    def delete_band(self, band_id):
        if band_id in self.bands:
            band = self.bands[band_id]
            # TODO remove it


class ApertureModel(object):
    def __init__(self):
        self.start = 100
        self.end = 300
        self.aperture_id = 1
        self.listeners = list()

    def add_listener(self, listener):
        self.listeners.append(listener)

    def adjust_aperture(self, aperture_id, start, end):
        for l in self.listeners:
            l.handle_aperture(aperture_id, start, end)


class ApertureView(object):
    def __init__(self, model, figure, y):
        self.boxes = dict()

        # left, right bars - line between - label aperture #
        ## This really isn't great.  Maybe with some work in creating a pixel-based glass pane overlay
        ## of some sort, if there even is a Bokeh equivalent.  For now, stubbing out to Box
        # self.label = Label(x=(model.start+model.end)/2-5, y=y+10, text="%s" % model.aperture_id)
        # figure.add_layout(self.label)
        # self.left_source = ColumnDataSource({'x': [model.start, model.start], 'y': [y-20, y+20]})
        # self.left = figure.line(x='x', y='y', source=self.left_source, color="purple")
        # self.right_source = ColumnDataSource({'x': [model.end, model.end], 'y': [y-20, y+20]})
        # self.right = figure.line(x='x', y='y', source=self.right_source, color="purple")
        # self.line_source = ColumnDataSource({'x': [model.start, model.end], 'y': [y, y]})
        # self.line = figure.line(x='x', y='y', source=self.line_source, color="purple")

        self.figure = figure
        model.add_listener(self)

        # Unfortunately, it seems the annotations MUST exist before plotting
        # So this is a hack until I can work around that somehow
        it_goes_to_eleven = -1 * sys.float_info.max
        for ap_id in range(20):
            ap = BoxAnnotation(left=it_goes_to_eleven, right=it_goes_to_eleven, fill_alpha=0.1, fill_color='green')
            self.figure.add_layout(ap)
            self.boxes[ap_id] = ap

    def handle_aperture(self, aperture_id, start, end):
        if aperture_id in self.boxes:
            box = self.boxes[aperture_id]
            box.left = start
            box.right = end
        else:
            box = BoxAnnotation(left=start, right=end, fill_alpha=0.1, fill_color='green')
            self.boxes[aperture_id] = box
            self.figure.add_layout(box)


class ApertureControls(object):
    def __init__(self, container, aperture_id, model, start, end):
        self.aperture_id = aperture_id
        self.model = model
        self.min_slider = Slider(title="Aperture %s Start" % aperture_id, value=0, step=1, start=start, end=end)
        self.max_slider = Slider(title="Aperture %s End" % aperture_id, value=0, step=1, start=start, end=end)
        container.children.append(self.min_slider)
        container.children.append(self.max_slider)
        self.min_slider.on_change("value", self.handler)
        self.max_slider.on_change("value", self.handler)

    def handler(self, attr, old, new):
        if old != new:
            min_val = self.min_slider.value
            max_val = self.max_slider.value
            if min_val > max_val:
                swapper = min_val
                min_val = max_val
                max_val = swapper
            self.model.adjust_aperture(self.aperture_id, min_val, max_val)
