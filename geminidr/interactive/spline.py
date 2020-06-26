from abc import ABC, abstractmethod

import numpy as np
from bokeh.layouts import row
from bokeh.models import Button, Column, CustomJS
from bokeh.plotting import figure

from geminidr.interactive import server, interactive
from geminidr.interactive.interactive import GICoordsSource, GILine, GIScatter
from gempy.library import astromodels


class SplineListener(ABC):
    @abstractmethod
    def handle_spline(self, splinex, spline):
        pass


class SplineModel:
    def __init__(self, ext, wave, zpt, zpt_err, order, niter, grow):
        self.ext = ext
        self.wave = wave
        self.zpt = zpt
        self.zpt_err = zpt_err
        self.order = order
        self.niter = niter
        self.grow = grow

        self.mask_points = GICoordsSource()
        self.fit_line = GICoordsSource()

    def recalc_spline(self):
        """
        Recalculate the spline based on the currently set parameters.

        Whenever one of the parameters that goes into the spline function is
        changed, we come back in here to do the recalculation.  Additionally,
        the resulting spline is used to update the line and the masked underlying
        scatter plot.

        Returns
        -------
        none
        """
        wave = self.wave
        zpt = self.zpt
        zpt_err = self.zpt_err
        order = self.order
        niter = self.niter
        grow = self.grow
        ext = self.ext

        self.spline = astromodels.UnivariateSplineWithOutlierRemoval(wave.value, zpt.value,
                                                                     w=1. / zpt_err.value,
                                                                     order=order,
                                                                     niter=niter,
                                                                     grow=grow)

        splinex = np.linspace(min(wave), max(wave), ext.shape[0])

        self.mask_points.ginotify(wave[self.spline.mask], zpt[self.spline.mask])
        self.fit_line.ginotify(splinex, self.spline(splinex))


class SplineVisualizer(interactive.PrimitiveVisualizerNew):
    def __init__(self, ext, wave, zpt, zpt_err, order, niter, grow, min_order, max_order,
                 min_niter, max_niter, min_grow, max_grow):
        """
        Create a spline visualizer.

        This makes a visualizer for letting a user interactively set the
        spline parameters.  The class handles some common logic for setting up
        the web document and for holding the result of the interaction.  In
        future, other visualizers will follow a similar pattern.

        Parameters
        ----------
        params : dict
            Parameter values from the primitive.  These are the user supplied values
            or defaults and may have come from command line overrides
        fields : iterable of :class:`config.Field`
            These don't reflect overrides from the user, but do provide us with helpful
            validation information such as min/max values.
        """
        super().__init__()
        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.model = SplineModel(ext, wave, zpt, zpt_err, order, niter, grow)
        self.p = None
        self.spline = None
        self.scatter = None
        self.scatter_touch = None
        self.line = None
        self.scatter_source = None
        self.line_source = None

        self.min_order = min_order
        self.max_order = max_order
        self.min_niter = min_niter
        self.max_niter = max_niter
        self.min_grow = min_grow
        self.max_grow = max_grow

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
        self.model.recalc_spline()

    def niter_slider_handler(self, attr, old, new):
        """
        Handle a change in the iterations slider
        """
        self.model.niter = new
        self.model.recalc_spline()

    def grow_slider_handler(self, attr, old, new):
        """
        Handle a change in the grow slider
        """
        self.model.grow = new
        self.model.recalc_spline()

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
        wave = self.model.wave
        zpt = self.model.zpt
        order = self.model.order
        niter = self.model.niter
        grow = self.model.grow

        order_slider = self.make_slider_for("Order", order, 1, self.min_order, self.max_order, self.order_slider_handler)
        niter_slider = self.make_slider_for("Num Iterations", niter, 1,  self.min_niter, self.max_niter,
                                            self.niter_slider_handler)
        grow_slider = self.make_slider_for("Grow", grow, 1, self.min_grow, self.max_grow, self.grow_slider_handler)

        button = Button(label="Submit")
        button.on_click(self.button_handler)
        callback = CustomJS(code="""
            window.close();
        """)
        button.js_on_click(callback)

        # Create a blank figure with labels
        self.p = figure(plot_width=600, plot_height=500,
                        title='Interactive Spline',
                        x_axis_label='X', y_axis_label='Y')
        # We can plot this here because it never changes
        # the overlay we plot later since it does change, giving
        # the illusion of "coloring" these points
        self.scatter_touch = self.p.scatter(wave, zpt, color="blue", radius=5)

        self.scatter = GIScatter(self.p, color="black")
        self.model.mask_points.add_gilistener(self.scatter)

        self.line = GILine(self.p)
        self.model.fit_line.add_gilistener(self.line)

        controls = Column(order_slider, niter_slider, grow_slider, button)

        self.model.recalc_spline()

        layout = row(controls, self.p)

        doc.add_root(layout)

        doc.on_session_destroyed(self.button_handler)

    def result(self):
        """
        Get the result of the user interaction.

        Returns
        -------
        :class:`astromodels.UnivariateSplineWithOutlierRemoval`
        """
        return self.model.spline


def interactive_spline(ext, wave, zpt, zpt_err, order, niter, grow, min_order, max_order, min_niter, max_niter,
                       min_grow, max_grow):
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
    fields dict
        dictionary of field-name to config values used to build
        sensible UI widgets (such as min/max values for sliders)

    Returns
    -------

    """
    params = dict(
        ext=ext,
        wave=wave,
        zpt=zpt,
        zpt_err=zpt_err,
        order=order,
        niter=niter,
        grow=grow
    )

    server.visualizer = SplineVisualizer(ext, wave, zpt, zpt_err, order, niter, grow, min_order, max_order,
                                         min_niter, max_niter, min_grow, max_grow)

    server.start_server()

    return server.visualizer.result()
