from abc import ABC, abstractmethod

from bokeh.layouts import row
from bokeh.models import Slider, TextInput, ColumnDataSource, BoxAnnotation, Button, CustomJS, Label
from bokeh.plotting import figure

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


class GISlider(object):
    def __init__(self, title, value, step, min_value, max_value, obj=None, attr=None, handler=None):
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
        obj : object
            Instance to modify the attribute of when slider changes
        attr : str
            Name of attribute in obj to be set with the new value
        handler : method
            Function to call after setting the attribute

        Returns
        -------
            :class:`bokeh.models.Slider` slider widget for bokeh interface
        """
        self.obj = obj
        self.attr = attr
        self.handler = handler

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

        def handle_value(attr, old, new):
            if self.obj and self.attr:
                self.obj.__setattr__(self.attr, new)
            self.handler()

        slider.on_change("value", update_text_input, handle_value)
        text_input.on_change("value", update_slider)
        self.component = component


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


class GIMaskedCoords(GICoordsSource):
    def __init__(self, x_coords, y_coords):
        super().__init__()
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.mask = [True] * len(x_coords)

    def add_gilistener(self, coords_listener):
        super().add_gilistener(coords_listener)
        coords_listener.giupdate(self.x_coords[self.mask], self.y_coords[self.mask])

    def addmask(self, coords):
        for i in coords:
            self.mask[i] = False
        self.ginotify(self.x_coords[self.mask], self.y_coords[self.mask])

    def unmask(self, coords):
        for i in coords:
            self.mask[i] = True
        self.ginotify(self.x_coords[self.mask], self.y_coords[self.mask])


class GIDifferencingCoords(GICoordsSource, GICoordsListener):
    def __init__(self, coords_source, fn):
        super().__init__()
        self.fn = fn
        self.mask = []
        coords_source.add_gilistener(self)

    def update_mask(self, mask):
        self.mask = mask

    def giupdate(self, x_coords, y_coords):
        x = x_coords[self.mask]
        y = y_coords[self.mask] - self.fn(x)

        self.ginotify(x, y)


class GIFigure(object):
    """
    This abstracts out any bugfixes or special handling we may need.  We may be able to deprecate it
    if bokeh bugs are fixed or if the benefits don't outweigh the complexity.
    """
    def __init__(self, title='Plot',
                 plot_width=600, plot_height=500,
                 x_axis_label='X', y_axis_label='Y',
                 tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                 band_model=None, aperture_model=None):
        # primarily, we set the backend to webgl for performance and we fix a bokeh bug to ensure rendering updates
        self.figure = figure(plot_width=plot_width, plot_height=plot_height, title=title, x_axis_label=x_axis_label,
                             y_axis_label=y_axis_label, tools=tools, output_backend="webgl")

        # If we have bands or apertures to show, show them
        if band_model:
            self.bands = GIBands(self, band_model)
        if aperture_model:
            self.aperture_view = GIApertureView(aperture_model, self)

        # This is a workaround for a bokeh bug.  Without this, things like the background shading used for
        # apertures and bands will not update properly after the figure is visible.
        self.figure.js_on_change('center', CustomJS(args=dict(plot=self.figure),
                                                    code="plot.properties.renderers.change.emit()"))


class GIScatter(GICoordsListener):
    def __init__(self, gifig, x_coords=None, y_coords=None, color="blue", radius=5):
        """
        Scatter plot

        Parameters
        ----------
        gifig : :class:`GIFigure` figure to plot in
        x_coords : array of float coordinates
        y_coords : array of float coordinates
        color : color value, default "blue"
        radius : int radius in pixels for the dots
        """
        if x_coords is None:
            x_coords = []
        if y_coords is None:
            y_coords = []
        self.source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.scatter = gifig.figure.scatter(x='x', y='y', source=self.source, color=color, radius=radius)

    def giupdate(self, x_coords, y_coords):
        self.source.data = {'x': x_coords, 'y': y_coords}

    def clear_selection(self):
        self.source.selected.update(indices=[])

    def replot(self):
        self.scatter.replot()


class GILine(GICoordsListener):
    def __init__(self, gifig, x_coords=[], y_coords=[], color="red"):
        """
        Line plot

        Parameters
        ----------
        gifig : :class:`GIFigure` figure to plot in
        x_coords : array of float coordinates
        y_coords : array of float coordinates
        color : color for line, default "red"
        """
        if x_coords is None:
            x_coords = []
        if y_coords is None:
            y_coords = []
        self.line_source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.line = gifig.figure.line(x='x', y='y', source=self.line_source, color=color)

    def giupdate(self, x_coords, y_coords):
        self.line_source.data = {'x': x_coords, 'y': y_coords}


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
        self.band_id = 1
        self.listeners = list()

    def add_listener(self, listener):
        if not isinstance(listener, GIBandListener):
            raise ValueError("must be a BandListener")
        self.listeners.append(listener)

    def adjust_band(self, band_id, start, stop):
        for listener in self.listeners:
            listener.adjust_band(band_id, start, stop)


class GIBands(GIBandListener):
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
            self.fig.figure.add_layout(band)
            self.bands[band_id] = band

    def delete_band(self, band_id):
        if band_id in self.bands:
            band = self.bands[band_id]
            # TODO remove it


class GIApertureModel(object):
    def __init__(self):
        self.start = 100
        self.end = 300
        self.aperture_id = 1
        self.listeners = list()
        self.spare_ids = list()

    def add_listener(self, listener):
        self.listeners.append(listener)

    def add_aperture(self, start, end):
        if self.spare_ids:
            aperture_id = self.spare_ids.pop(0)
        else:
            aperture_id = self.aperture_id
            self.aperture_id += 1
        self.adjust_aperture(aperture_id, start, end)
        return aperture_id

    def adjust_aperture(self, aperture_id, start, end):
        for l in self.listeners:
            l.handle_aperture(aperture_id, start, end)

    def delete_aperture(self, aperture_id):
        for l in self.listeners:
            l.delete_aperture(aperture_id)
        self.spare_ids.append(aperture_id)


class GISingleApertureView(object):
    def __init__(self, gifig, aperture_id, start, end):
        self.box = None
        self.label = None
        self.left_source = None
        self.left = None
        self.right_source = None
        self.right = None
        self.line_source = None
        self.line = None
        self.figure = None
        if gifig.figure.document:
            gifig.figure.document.add_next_tick_callback(lambda: self.build_ui(gifig, aperture_id, start, end))
        else:
            self.build_ui(gifig, aperture_id, start, end)

    def build_ui(self, gifig, aperture_id, start, end):
        figure = gifig.figure
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

        self.gifig = gifig

        figure.y_range.on_change('start', lambda attr, old, new: self.update_viewport())
        figure.y_range.on_change('end', lambda attr, old, new: self.update_viewport())
        # feels like I need this to convince the aperture lines to update on zoom
        figure.y_range.js_on_change('end', CustomJS(args=dict(plot=figure),
                                                           code="plot.properties.renderers.change.emit()"))

    def update_viewport(self):
        ymin = self.gifig.figure.y_range.start
        ymax = self.gifig.figure.y_range.end
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

    def delete(self):
        self.gifig.figure.renderers.remove(self.line)
        self.gifig.figure.renderers.remove(self.left)
        self.gifig.figure.renderers.remove(self.right)


class GIApertureView(object):
    def __init__(self, model, gifig):
        self.aps = dict()

        self.gifig = gifig
        model.add_listener(self)

    def handle_aperture(self, aperture_id, start, end):
        if aperture_id in self.aps:
            ap = self.aps[aperture_id]
            ap.update(start, end)
        else:
            ap = GISingleApertureView(self.gifig, aperture_id, start, end)
            self.aps[aperture_id] = ap

    def delete_aperture(self, aperture_id):
        if aperture_id in self.aps:
            ap = self.aps[aperture_id]
            ap.delete()
