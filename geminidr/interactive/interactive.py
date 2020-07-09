from abc import ABC, abstractmethod

from bokeh.layouts import row
from bokeh.models import Slider, TextInput, ColumnDataSource, BoxAnnotation, Button, CustomJS, Label

from geminidr.interactive import server


class PrimitiveVisualizer(ABC):
    def __init__(self):
        self.submit_button = Button(label="Submit")
        self.submit_button.on_click(self.submit_button_handler)
        callback = CustomJS(code="""
            window.close();
        """)
        self.submit_button.js_on_click(callback)

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
        server.bokeh_server.io_loop.stop()

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
        doc.on_session_destroyed(self.submit_button_handler)

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
        min_value : int
            Minimum slider value, or None defaults to min(value,0)
        max_value : int
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


class BandListener(ABC):
    @abstractmethod
    def adjust_band(self, band_id, start, stop):
        pass

    def delete_band(self, band_id):
        pass


class BandModel(object):
    def __init__(self):
        self.band_id = 1
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
        self.model = model
        model.add_listener(self)
        self.bands = dict()
        self.fig = fig

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


class SingleApertureView(object):
    def __init__(self, figure, aperture_id, start, end, y):
        # ymin = figure.y_range.computed_start
        # ymax = figure.y_range.computed_end
        ymin = figure.y_range.start
        ymax = figure.y_range.end
        ymid = (ymax-ymin)*.8+ymin
        ytop = ymid + 0.05*(ymax-ymin)
        ybottom = ymid - 0.05*(ymax-ymin)
        self.box = BoxAnnotation(left=start, right=end, fill_alpha=0.1, fill_color='green')
        figure.add_layout(self.box)
        self.label = Label(x=(start+end)/2-5, y=ymid, text="%s" % aperture_id)
        figure.add_layout(self.label)
        self.left_source = ColumnDataSource({'x': [start, start], 'y': [ybottom, ytop]})
        self.left = figure.line(x='x', y='y', source=self.left_source, color="purple")
        self.right_source = ColumnDataSource({'x': [end, end], 'y': [ybottom, ytop]})
        self.right = figure.line(x='x', y='y', source=self.right_source, color="purple")
        self.line_source = ColumnDataSource({'x': [start, end], 'y': [ymid, ymid]})
        self.line = figure.line(x='x', y='y', source=self.line_source, color="purple")

        self.figure = figure

        figure.y_range.on_change('start', lambda attr, old, new: self.update_viewport())
        figure.y_range.on_change('end', lambda attr, old, new: self.update_viewport())
        # feels like I need this to convince the aperture lines to update on zoom
        figure.y_range.js_on_change('end', CustomJS(args=dict(plot=figure),
                                                    code="plot.properties.renderers.change.emit()"))

    def update_viewport(self):
        ymin = self.figure.y_range.start
        ymax = self.figure.y_range.end
        ymid = (ymax-ymin)*.8+ymin
        ytop = ymid + 0.05*(ymax-ymin)
        ybottom = ymid - 0.05*(ymax-ymin)
        self.left_source.data = {'x': self.left_source.data['x'], 'y': [ybottom, ytop]}
        self.right_source.data = {'x': self.right_source.data['x'], 'y': [ybottom, ytop]}
        self.line_source.data = {'x':  self.line_source.data['x'], 'y': [ymid, ymid]}
        self.label.y = ymid

    def update(self, start, end):
        self.box.left = start
        self.box.right = end
        self.left_source.data = {'x': [start, start], 'y': self.left_source.data['y']}
        self.right_source.data = {'x': [end, end], 'y': self.right_source.data['y']}
        self.line_source.data = {'x': [start, end], 'y': self.line_source.data['y']}


class ApertureView(object):
    def __init__(self, model, figure, y):
        self.aps = dict()
        self.y = y

        self.figure = figure
        model.add_listener(self)

    def handle_aperture(self, aperture_id, start, end):
        if aperture_id in self.aps:
            ap = self.aps[aperture_id]
            ap.update(start, end)
        else:
            ap = SingleApertureView(self.figure, aperture_id, start, end, self.y)
            self.aps[aperture_id] = ap

