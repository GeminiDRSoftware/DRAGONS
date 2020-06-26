import numpy as np
from astropy.modeling import models, fitting
from bokeh.layouts import row
from bokeh.models import Button, Column, ColumnDataSource, CustomJS, Slider, ColorPicker, Panel, Tabs
from bokeh.plotting import figure

from geminidr.interactive import server, interactive
from geminidr.interactive.interactive import GIScatter, GILine, GICoordsSource
from gempy.library import astromodels


class ChebyshevModel(GICoordsSource):
    def __init__(self, order, location, dispaxis, sigma_clip, in_coords, spectral_coords, ext):
        super().__init__()
        self.order = order
        self.location = location
        self.dispaxis = dispaxis
        self.sigma_clip = sigma_clip
        self.in_coords = in_coords
        self.spectral_coords = spectral_coords
        self.ext = ext
        self.m_final = None
        self.model_dict = None

    def recalc_chebyshev(self):
        """
        Recalculate the Chebyshev1D based on the currently set parameters.

        Whenever one of the parameters that goes into the spline function is
        changed, we come back in here to do the recalculation.  Additionally,
        the resulting spline is used to update the line and the masked underlying
        scatter plot.

        Returns
        -------
        none
        """
        order = self.order
        location = self.location
        dispaxis = self.dispaxis
        sigma_clip = self.sigma_clip
        in_coords = self.in_coords
        ext = self.ext

        m_init = models.Chebyshev1D(degree=order, c0=location,
                                    domain=[0, ext.shape[dispaxis] - 1])
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                   sigma_clip, sigma=3)
        try:
            self.m_final, _ = fit_it(m_init, in_coords[1 - dispaxis], in_coords[dispaxis])
        except (IndexError, np.linalg.linalg.LinAlgError):
            # This hides a multitude of sins, including no points
            # returned by the trace, or insufficient points to
            # constrain the request order of polynomial.
            self.m_final = models.Chebyshev1D(degree=0, c0=location,
                                              domain=[0, ext.shape[dispaxis] - 1])
        self.model_dict = astromodels.chebyshev_to_dict(self.m_final)

        self.ginotify(self.spectral_coords, self.m_final(self.spectral_coords))


class Chebyshev1DVisualizer(interactive.PrimitiveVisualizerNew):
    def __init__(self, model, min_order, max_order):
        """
        Create a chebyshev1D visualizer.

        This makes a visualizer for letting a user interactively set the
        Chebyshev parameters.  The class handles some common logic for setting up
        the web document and for holding the result of the interaction.  In
        future, other visualizers will follow a similar pattern.

        Parameters
        ----------
        """
        super().__init__()

        if not isinstance(model, ChebyshevModel):
            raise ValueError("Chebyshev1DVisualizerNew requires ChebyshevModel")

        self.model = model

        self.min_order = min_order
        self.max_order = max_order

        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.p = None
        self.spline = None
        self.scatter = None
        self.scatter_touch = None
        self.line = None
        self.scatter_source = None
        self.line_source = None
        self.spectral_coords = None
        self.model_dict = None
        self.m_final = None

        self.scatter2 = None
        self.line2 = None

    def button_handler(self, stuff):
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

    def order_slider_handler(self, attr, old, new):
        """
        Handle a change in the order slider
        """
        self.model.order = new
        self.model.recalc_chebyshev()

    def visualize(self, doc):
        """
        Build the visualization in bokeh in the given browser document.

        Parameters
        ----------
        doc
            Bokeh provided document to add visual elements to

        Returns
        -------
        none
        """
        order_slider = self.make_slider_for("Order", self.model.order, 1, self.min_order, self.max_order,
                                            self.order_slider_handler)

        button = Button(label="Submit")
        button.on_click(self.button_handler)
        callback = CustomJS(code="""
            window.close();
        """)
        button.js_on_click(callback)

        # Create a blank figure with labels
        p = figure(plot_width=600, plot_height=500,
                        title='Interactive Chebyshev',
                        x_axis_label='X', y_axis_label='Y',
                        output_backend="webgl")

        # p2 is just to show that we can have multiple tabs with plots running off the same dataset
        # TODO make plots like the other IRAF options we were shown
        p2 = figure(plot_width=600, plot_height=500,
                        title='Interactive Chebyshev (tab 2)',
                        x_axis_label='X', y_axis_label='Y',
                        output_backend="webgl")

        self.scatter = GIScatter(p, self.model.in_coords[0], self.model.in_coords[1], color="blue", radius=5)
        self.line = GILine(p)

        self.scatter2 = GIScatter(p2, self.model.in_coords[0], self.model.in_coords[1], color="blue", radius=5)
        self.line2 = GILine(p2)

        controls = Column(order_slider, button)

        self.model.add_gilistener(self.line)
        self.model.add_gilistener(self.line2)

        self.model.recalc_chebyshev()

        tab1 = Panel(child=p, title="Chebyshev Fit")
        tab2 = Panel(child=p2, title="Chebyshev Fit 2 (TODO something else)")
        tabs = Tabs(tabs=[tab1, tab2])
        layout = row(controls, tabs)

        doc.add_root(layout)

        doc.on_session_destroyed(self.button_handler)


def interactive_chebyshev(ext,  order, location, dispaxis, sigma_clip, in_coords, spectral_coords,
                          min_order, max_order):
    """
    Build a spline via user interaction.

    This method spins up bokeh and uses a web-based bokeh gui to create a spline
    from user input.  Values passed in are used for the data points and as a
    starting point for the interface.

    Parameters
    ----------
    ext
        FITS extension from astrodata
    wave
    zpt
    zpt_err
    order
        order for the spline calculation
    niter
        number of iterations for the spline calculation
    grow
        grow for the spline calculation
    min_order
        minimum value for order slider
    max_order
        maximum value for order slider, or None to infer

    Returns
    -------
        dict, :class:`models.Chebyshev1D`
    """
    model = ChebyshevModel(order, location, dispaxis, sigma_clip,
                           in_coords, spectral_coords, ext)
    server.visualizer = Chebyshev1DVisualizer(model, min_order, max_order)

    server.start_server()

    server.visualizer = None

    return model.model_dict, model.m_final
