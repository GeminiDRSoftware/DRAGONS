from abc import ABC, abstractmethod

from astropy.units import Quantity
from bokeh.layouts import row, column
from bokeh.models import Slider, TextInput, ColumnDataSource, BoxAnnotation, Button, CustomJS, Label, Column, Div
from bokeh.plotting import figure

from geminidr.interactive import server


def _dequantity(x, y):
    """
    Utility to convert richer Quantity based values into raw ndarrays.

    Parameters
    ----------
    x : `~ndarray` or `~Quantity`
        X values
    y : `~ndarray` or `~Quantity`
        Y values

    Returns
    -------
    x, y as `~ndarray`s
    """
    if isinstance(x, Quantity):
        x = x.value
    if isinstance(y, Quantity):
        y = y.value
    return x, y


class PrimitiveVisualizer(ABC):
    def __init__(self, log=None):
        """
        Initialize a visualizer.

        This base class creates a submit button suitable for any subclass
        to use and it also listens for the UI window to close, executing a
        submit if that happens.  The submit button will cause the `bokeh`
        event loop to exit and the code will resume executing in whatever
        top level call you are visualizing from.
        """
        self.log = log

        self.submit_button = Button(label="Submit")
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
        server.stop_server()

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
        self.doc = doc
        doc.on_session_destroyed(self.submit_button_handler)

    def do_later(self, fn):
        if self.doc is None:
            if self.log is not None:
                self.log.warn("Call to do_later, but no document is set.  Does this PrimitiveVisualizer call "
                              "super().visualize(doc)?")
        else:
            self.doc.add_next_tick_callback(lambda: fn())

    def make_modal(self, widget, message):
        callback = CustomJS(args=dict(source=widget), code="""
            console.log("checking button state");
            if (source.disabled) {
                openModal('%s');
            } else {
                closeModal();
            }
        """ % message)
        widget.js_on_change('disabled', callback)


def build_text_slider(title, value, step, min_value, max_value, obj=None, attr=None, handler=None,
                      throttled=False):
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

    slider = slider
    text_input = text_input

    def update_slider(attrib, old, new):
        if old != new:
            ival = None
            try:
                ival = int(new)
            except ValueError:
                ival = float(new)
            if ival > slider.end and not max_value:
                slider.end = ival
            if 0 <= ival < slider.start and min_value is None:
                slider.start = ival
            if slider.start <= ival <= slider.end:
                slider.value = ival

    def update_text_input(attrib, old, new):
        if new != old:
            text_input.value = str(new)

    def handle_value(attrib, old, new):
        if isinstance(new, str):
            new = float(new)
        if obj and attr:
            obj.__setattr__(attr, new)
        if handler:
            handler()

    if throttled:
        # Since here the text_input calls handle_value, we don't
        # have to call it from the slider as it will happen as
        # a side-effect of update_text_input
        slider.on_change("value_throttled", update_text_input)
        text_input.on_change("value", update_slider, handle_value)
    else:
        slider.on_change("value", update_text_input, handle_value)
        # since slider is listening to value, this next line will cause the slider
        # to call the handle_value method and we don't need to do so explicitly
        text_input.on_change("value", update_slider)
    return component


# class GICoordsSource:
#     """
#     A source for coordinate data.
#
#     Downstream code can subscribe for updates on this to be notified when the
#     coordinates change for some reason.
#     """
#     def __init__(self):
#         self.listeners = list()
#
#     def add_coord_listener(self, coords_listener):
#         """
#         Add a listener - a function that will take x and y
#         arguments as lists of values.
#         """
#         if callable(coords_listener):
#             self.listeners.append(coords_listener)
#         else:
#             raise ValueError("Must pass a fn(x,y)")
#
#     def notify_coord_listeners(self, x_coords, y_coords):
#         """
#         Notify all registered users of the updated coordinagtes.
#
#         Coordinates are set as two separate arrays of `ndarray`
#         x and y coordinates.
#
#         Parameters
#         ----------
#         x_coords : ndarray
#             x coordinate array
#         y_coords : ndarray
#             y coordinate array
#
#         """
#         for l in self.listeners:
#             l(x_coords, y_coords)


class GIModelSource(object):
    """"
    An object for reporting updates to a model (such as a fit line).

    This is an interface for adding subscribers to model updates.  For
    example, you may have a best fit line model and you may want to
    have UI classes subscribe to it so they know when the fit line has
    changed.
    """
    def __init__(self):
        """
        Create the model source.
        """
        self.model_listeners = list()

    def add_model_listener(self, listener):
        """
        Add the listener.

        Parameters
        ----------
        listener : function
            The listener to notify when the model updates.  This should be
            a function with no arguments.
        """
        if not callable(listener):
            raise ValueError("GIModelSource expects a callable in add_listener")
        self.model_listeners.append(listener)

    def notify_model_listeners(self):
        """
        Call all listeners to let them know the model has changed.
        """
        for listener_fn in self.model_listeners:
            listener_fn()


class GIDifferencingModel(object):
    """
    A coordinate model for tracking the difference in x/y
    coordinates and what is calculated by a function(x).

    This is useful for plotting differences between the
    real coordinates and what a model function is predicting
    for the given x values.  It will listen to changes to
    both an underlying :class:`GICoordsSource` and a
    :class:`GIModelSource` and when either update, it
    recalculates the differences and sends out x, (y-fn(x))
    coordinates to any listeners.
    """
    def __init__(self, coords, cmodel, fn):
        """
        Create the differencing model.

        Parameters
        ----------
        coords : :class:`GICoordsSource`
            Coordinates to serve as basis for the difference
        cmodel : :class:`GIModelSource`
            The model source, to be notified when the model changes
        fn : function
            The function, related to the model, to call to get modelled y-values
        """
        super().__init__()
        # Separating the fn from the model source is a bit hacky.  I need to revisit this.
        # For now, Chebyshev1D and Spline are different enough that I am holding out for
        # more examples of models.
        # TODO merge fn concept into model source
        self.fn = fn
        self.data_x_coords = None
        self.data_y_coords = None
        coords.add_coord_listener(self.update_coords)
        cmodel.add_model_listener(self.update_model)
        self.coord_listeners = list()

    def add_coord_listener(self, l):
        self.coord_listeners.append(l)
        if self.data_x_coords is not None:
            l(self.data_x_coords, self.data_y_coords - self.fn(self.data_x_coords))

    def update_coords(self, x_coords, y_coords):
        """
        Handle an update to the coordinates.

        We respond to updates in the source coordinates by
        recalculating model outputs for the new x inputs
        and publishing to our listeners an updated set of
        x, (y-fn(x)) values.

        Parameters
        ----------
        x_coords : ndarray
            X coordinates
        y_coord : ndarray
            Y coordinatess

        """
        self.data_x_coords = x_coords
        self.data_y_coords = y_coords

    def update_model(self):
        """"
        Called by the :class:`GIModelSource` to let us know the model
        has been updated.

        We respond to a model update by recalculating the x, (y-fn(x))
        values and publishing them out to our subscribers.
        """
        x = self.data_x_coords
        y = self.data_y_coords - self.fn(x)
        for fn in self.coord_listeners:
            fn(x, y)


class GIMaskedSigmadCoords(object):
    """
    This is a helper class for handling masking of coordinate
    values.

    This class tracks an initial, static, set of x/y coordinates
    and a changeable list of masked coordinate indices.  Whenever
    the mask is updated, we publish the subset of the coordinates
    that pass the mask out to our listeners.

    A typical use for this would be to make 2 overlapping scatter
    plots.  One will be in a base color, such as black.  The other
    will be in a different color, such as blue.  The blue plot can
    be made using this coordinate source and the effect is a plot
    of all points, with the masked points in blue.  This is currently
    done in the Spline logic, for example.
    """
    def __init__(self, x_coords, y_coords):
        """
        Create the masked coords source with the given set of coordinates.

        Parameters
        ----------
        x_coords : ndarray
            x coordinates
        y_coords : ndarray
            y coordinates
        """
        super().__init__()
        self.x_coords = x_coords
        self.y_coords = y_coords
        # intially, all points are masked = included
        self.mask = [True] * len(x_coords)
        # initially, all points are not sigma = excluded
        self.sigma = [False] * len(x_coords)
        self.mask_listeners = list()
        self.sigma_listeners = list()
        self.coord_listeners = list()

    def set_coords(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords
        # intially, all points are masked = included
        self.mask = [True] * len(x_coords)
        # initially, all points are not sigma = excluded
        self.sigma = [False] * len(x_coords)
        self.notify_coord_listeners(self.x_coords, self.y_coords)
        for mask_listener in self.mask_listeners:
            mask_listener(self.x_coords[self.mask], self.y_coords[self.mask])
        for sigma_listener in self.sigma_listeners:
            sigma_listener([], [])

    def add_coord_listener(self, coords_listener):
        """
        Add a listener for updates.

        Since we have the coordinates at construction time, this call
        will also immediately notify the passed listener of the currently
        passing masked coordinates.

        Parameters
        ----------
        coords_listener : :class:`GICoordsListener` or function
            The listener to add

        """
        self.coord_listeners.append(coords_listener)
        coords_listener(self.x_coords, self.y_coords)

    def add_mask_listener(self, mask_listener: callable):
        if callable(mask_listener):
            self.mask_listeners.append(mask_listener)
            mask_listener(self.x_coords[self.mask], self.y_coords[self.mask])
        else:
            raise ValueError("add_mask_listener takes a callable function")

    def add_sigma_listener(self, sigma_listener: callable):
        if callable(sigma_listener):
            self.sigma_listeners.append(sigma_listener)
        else:
            raise ValueError("add_sigma_listener takes a callable function")

    def addmask(self, coords):
        """
        Set the given cooridnate indices as masked (so, visible)

        This also notifies all listeners of the updated set of passing
        coordinates.

        Parameters
        ----------
        coords : array of int
            List of coordinates to enable in the mask

        """
        for i in coords:
            self.mask[i] = False
        self.sigma = [False] * len(self.x_coords[self.mask])
        for fn in self.coord_listeners:
            fn(self.x_coords, self.y_coords)
        for mask_listener in self.mask_listeners:
            mask_listener(self.x_coords[self.mask], self.y_coords[self.mask])
        for sigma_listener in self.sigma_listeners:
            sigma_listener([], [])

    def unmask(self, coords):
        """
        Set the given coordinate indices as unmasked (so, not visible)

        This also notifies all listeners of the updated set of passing
        coordinates.

        Parameters
        ----------
        coords : array of int
            list of coordinate indices to hide

        """
        for i in coords:
            self.mask[i] = True
        self.sigma = [False] * len(self.x_coords[self.mask])
        for sigma_listener in self.sigma_listeners:
            sigma_listener([], [])
        for fn in self.coord_listeners:
            fn(self.x_coords, self.y_coords)
        for mask_listener in self.mask_listeners:
            mask_listener(self.x_coords[self.mask], self.y_coords[self.mask])

    def set_sigma(self, coords):
        """
        Set the given cooridnate indices as excluded by sigma (so, highlight accordingly)

        This also notifies all listeners of the updated set of passing
        coordinates.

        Parameters
        ----------
        coords : array of int
            List of coordinates to flag as sigma excluded

        """
        self.sigma = [False] * len(self.x_coords[self.mask])
        for i in coords:
            self.sigma[i] = True
        for sigma_listener in self.sigma_listeners:
            sigma_listener(self.x_coords[self.mask][self.sigma], self.y_coords[self.mask][self.sigma])


def build_figure(title='Plot',
                 plot_width=600, plot_height=500,
                 x_axis_label='X', y_axis_label='Y',
                 tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                 band_model=None, aperture_model=None, x_range=None, y_range=None):
    # This wrapper around figure provides somewhat limited value, but for now I think it is
    # worth it.  It primarily does three things:
    #
    #  * allows alternate defaults for things like the list of tools and backend
    #  * integrates aperture and band information to reduce boilerplate in the visualizer code
    #  * wraps any bugfix hackery we need to do so it always happens and we don't have to remember it everywhere

    fig = figure(plot_width=plot_width, plot_height=plot_height, title=title, x_axis_label=x_axis_label,
                 y_axis_label=y_axis_label, tools=tools, output_backend="webgl", x_range=x_range,
                 y_range=y_range)

    # If we have bands or apertures to show, show them
    if band_model:
        bands = GIBandView(fig, band_model)
    if aperture_model:
        aperture_view = GIApertureView(aperture_model, fig)

    # This is a workaround for a bokeh bug.  Without this, things like the background shading used for
    # apertures and bands will not update properly after the figure is visible.
    fig.js_on_change('center', CustomJS(args=dict(plot=fig),
                                           code="plot.properties.renderers.change.emit()"))

    return fig


def build_cds(x_coords=[], y_coords=[]):
    if x_coords is None:
        x_coords = []
    if y_coords is None:
        y_coords = []
    return ColumnDataSource({'x': x_coords, 'y': y_coords})


def connect_update_coords(source):
    def update_coords(x_coords, y_coords):
        x, y = _dequantity(x_coords, y_coords)
        source.data = {'x': x, 'y': y}
    return update_coords


def clear_selection(source):
    source.selected.update(indices=[])


class GIMaskedSigmadScatter:
    def __init__(self, fig, coords, color="red",
                 masked_color="blue", sigma_color="orange", radius=5):
        """
        Masked/Sigmad Scatter plot

        Parameters
        ----------
        gifig : :class:`GIFigure`
            figure to plot in
        coords : :class:`GIMaskedSigmaCoords`
            coordinate holder that also tracks masking and sigma
        color : str
            color value for unselected points (initially none of them), default "red"
        masked_color : str
            color for masked (included) points, default "blue"
        sigma_color : str
            color for sigma-excluded points, default "orange"
        radius : int
            radius in pixels for the dots
        """
        if not isinstance(coords, GIMaskedSigmadCoords):
            raise ValueError("coords passed must be a GIMaskedSigmadCoords instance")
        x_coords = coords.x_coords
        y_coords = coords.y_coords
        self.source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.masked_source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.sigmad_source = ColumnDataSource({'x': [], 'y': []})
        self.scatter = fig.scatter(x='x', y='y', source=self.source, color=color, radius=radius)
        self.masked_scatter = fig.scatter(x='x', y='y', source=self.masked_source, color=masked_color, radius=radius)
        self.sigma_scatter = fig.scatter(x='x', y='y', source=self.sigmad_source, color=sigma_color, radius=radius)

        coords.add_coord_listener(connect_update_coords(self.source))
        coords.add_mask_listener(connect_update_coords(self.masked_source))
        coords.add_sigma_listener(connect_update_coords(self.sigmad_source))

    def clear_selection(self):
        """
        Clear the selection in the scatter plot.

        This is useful once we have applied the selection in some way,
        to reset the plot back to an unselected state.
        """
        self.source.selected.update(indices=[])
        self.masked_source.selected.update(indices=[])
        self.sigmad_source.selected.update(indices=[])


class GIPatch:
    def __init__(self, fig, x_coords=[], y_coords=[], color="blue"):
        if x_coords is None:
            x_coords = []
        if y_coords is None:
            y_coords = []
        self.patch_source = ColumnDataSource({'x': x_coords, 'y': y_coords})
        self.patch = fig.patch(x='x', y='y', source=self.patch_source, color=color)

    def update_coords(self, x_coords, y_coords):
        x, y = _dequantity(x_coords, y_coords)
        self.patch_source.data = {'x': x, 'y': y}


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
        # Right now, the band model is effectively stateless, other
        # than maintaining the set of registered listeners.  That is
        # because the bands are not used for anything, so there is
        # no need to remember where they all are.  This is likely to
        # change in future and that information should likely be
        # kept in here.
        self.band_id = 1
        self.listeners = list()

    def add_listener(self, listener):
        """
        Add a listener to this band model.

        The listener can either be a :class:`GIBandListener` or
        it can be a function,  The function should expect as
        arguments, the `band_id`, and `start`, and `stop` x
        range values.

        Parameters
        ----------
        listener : :class:`GIBandListener` or function

        """
        if not isinstance(listener, GIBandListener):
            raise ValueError("must be a BandListener")
        self.listeners.append(listener)

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
        for listener in self.listeners:
            listener.adjust_band(band_id, start, stop)


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
        fig : :class:`GIFigure`
            the figure to display the bands in
        model : :class:`GIBandModel`
            the model for the band information (may be shared by multiple :class:`GIBandView`s)
        """
        self.model = model
        model.add_listener(self)
        self.bands = dict()
        self.fig = fig

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
        if band_id in self.bands:
            band = self.bands[band_id]
            band.left = start
            band.right = stop
        else:
            band = BoxAnnotation(left=start, right=stop, fill_alpha=0.1, fill_color='navy')
            self.fig.figure.add_layout(band)
            self.bands[band_id] = band

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
        if band_id in self.bands:
            band = self.bands[band_id]
            # TODO remove it


class GIApertureModel(object):
    """
    Model for tracking the Apertures.

    This tracks the apertures and a list of subscribers
    to notify when there are any changes.
    """
    def __init__(self):
        """
        Create the apertures model
        """
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
        listener : :class:`GIApertureListener` or function
            The listener to notify if there are any updates
        """
        self.listeners.append(listener)

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
        aperture_id = self.aperture_id
        self.aperture_id += 1
        self.adjust_aperture(aperture_id, start, end)
        return aperture_id

    def adjust_aperture(self, aperture_id, start, end):
        """
        Adjust an existing aperture by ID to a new range.
        This will alert all subscribed listeners.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to adjust
        start : float
            X coordinate of the new start of range
        end : float
            X coordiante of the new end of range

        """
        for l in self.listeners:
            l.handle_aperture(aperture_id, start, end)

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
        for listener in self.listeners:
            listener.delete_aperture(aperture_id)
        self.aperture_id = self.aperture_id-1

    def clear_apertures(self):
        while self.aperture_id > 1:
            self.delete_aperture(self.aperture_id-1)


class GISingleApertureView(object):
    def __init__(self, fig, aperture_id, start, end):
        """
        Create a visible glyph-set to show the existance
        of an aperture on the given figure.  This display
        will update as needed in response to panning/zooming.

        Parameters
        ----------
        gifig : :class:`GIFigure`
            Figure to attach to
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
        self.fig = None
        if fig.document:
            fig.document.add_next_tick_callback(lambda: self.build_ui(fig, aperture_id, start, end))
        else:
            self.build_ui(fig, aperture_id, start, end)

    def build_ui(self, fig, aperture_id, start, end):
        """
        Build the view in the figure.

        This call creates the UI elements for this aperture in the
        parent figure.  It also wires up event listeners to adjust
        the displayed glyphs as needed when the view changes.

        Parameters
        ----------
        fig : :class:`Figure`
            bokeh figure to attach glyphs to
        aperture_id : int
            ID of this aperture, displayed
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
        self.label = Label(x=(start+end)/2-5, y=ymid, text="%s" % aperture_id)
        fig.add_layout(self.label)
        self.left_source = ColumnDataSource({'x': [start, start], 'y': [ybottom, ytop]})
        self.left = fig.line(x='x', y='y', source=self.left_source, color="purple")
        self.right_source = ColumnDataSource({'x': [end, end], 'y': [ybottom, ytop]})
        self.right = fig.line(x='x', y='y', source=self.right_source, color="purple")
        self.line_source = ColumnDataSource({'x': [start, end], 'y': [ymid, ymid]})
        self.line = fig.line(x='x', y='y', source=self.line_source, color="purple")

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

    def update(self, start, end):
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
        self.label.x = (start+end)/2-5

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


class GIApertureSliders(object):
    def __init__(self, view, fig, model, aperture_id, start, end):
        self.view = view
        self.model = model
        self.aperture_id = aperture_id
        self.start = start
        self.end = end

        slider_start = fig.x_range.start
        slider_end = fig.x_range.end

        title = "<h3>Aperture %s</h3>" % aperture_id
        self.label = Div(text=title)
        self.lower_slider = build_text_slider("Start", start, 0.01, slider_start, slider_end,
                                              obj=self, attr="start", handler=self.do_update)
        self.upper_slider = build_text_slider("End", end, 0.01, slider_start, slider_end,
                                              obj=self, attr="end", handler=self.do_update)
        button = Button(label="Delete")
        button.on_click(self.delete_from_model)

        self.component = Column(self.label, self.lower_slider, self.upper_slider, button)

    def delete_from_model(self):
        self.model.delete_aperture(self.aperture_id)

    def update_viewport(self, start, end):
        if self.start < start or self.end > end:
            self.lower_slider.children[0].disabled = True
            self.lower_slider.children[1].disabled = True
        else:
            self.lower_slider.children[0].disabled = False
            self.lower_slider.children[1].disabled = False
            self.lower_slider.children[0].start = start
            self.lower_slider.children[0].end = end
        if self.upper_slider.children[0].value < start or self.upper_slider.children[0].value > end:
            self.upper_slider.children[0].disabled = True
            self.upper_slider.children[1].disabled = True
        else:
            self.upper_slider.children[0].disabled = False
            self.upper_slider.children[1].disabled = False
            self.upper_slider.children[0].start = start
            self.upper_slider.children[0].end = end

    def do_update(self):
        if self.start > self.end:
            self.model.adjust_aperture(self.aperture_id, self.end, self.start)
        else:
            self.model.adjust_aperture(self.aperture_id, self.start, self.end)


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
        model : :class:`GIApertureModel`
            Model for tracking the apertures, may be shared across multiple views
        fig : :class:`~Figure`
            bokeh plot for displaying the bands
        """
        self.aps = list()
        self.ap_sliders = list()

        self.fig = fig
        self.controls = column()
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
        self.view_start = start
        self.view_end = end
        for ap_slider in self.ap_sliders:
            ap_slider.update_viewport(start, end)

    def handle_aperture(self, aperture_id, start, end):
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
            ap.update(start, end)
        else:
            ap = GISingleApertureView(self.fig, aperture_id, start, end)
            self.aps.append(ap)
            slider = GIApertureSliders(self, self.fig, self.model, aperture_id, start, end)
            self.ap_sliders.append(slider)
            self.controls.children.append(slider.component)

    def delete_aperture(self, aperture_id):
        """
        Remove an aperture by ID.  If the ID is not recognized, do nothing.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to remove

        Returns
        -------

        """
        if aperture_id <= len(self.aps):
            ap = self.aps[aperture_id-1]
            ap.delete()
            self.controls.children.remove(self.ap_sliders[aperture_id-1].component)
            del self.aps[aperture_id-1]
            del self.ap_sliders[aperture_id-1]
        for ap in self.aps[aperture_id-1:]:
            ap.aperture_id = ap.aperture_id-1
            ap.label.text = "%s" % ap.aperture_id
        for ap_slider in self.ap_sliders[aperture_id-1:]:
            ap_slider.aperture_id = ap_slider.aperture_id-1
            ap_slider.label.text = "<h3>Aperture %s</h3>" % ap_slider.aperture_id
