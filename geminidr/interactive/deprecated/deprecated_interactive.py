from astropy.units import Quantity
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from geminidr.interactive.interactive import connect_figure_extras


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

    connect_figure_extras(fig, aperture_model, band_model)

    return fig