import numpy as np
from bokeh.layouts import row
from bokeh.models import Button, Column, ColumnDataSource, CustomJS
from bokeh.plotting import figure

from geminidr.interactive import server
from gempy.library import astromodels


class SplineVisualizer(server.PrimitiveVisualizer):
    def __init__(self, params, fields):
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
        super().__init__(params, fields)
        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.p = None
        self.spline = None
        self.scatter = None
        self.line = None
        self.scatter_source = None
        self.line_source = None

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
        self._params["order"] = new
        self.recalc_spline()

    def niter_slider_handler(self, attr, old, new):
        self._params["niter"] = new
        self.recalc_spline()

    def grow_slider_handler(self, attr, old, new):
        self._params["grow"] = new
        self.recalc_spline()

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
        wave = self._params["wave"]
        zpt = self._params["zpt"]
        order = self._params["order"]
        niter = self._params["niter"]
        grow = self._params["grow"]

        order_slider = self.make_slider_for("Order", order, 1, self._fields["order"], self.order_slider_handler)
        niter_slider = self.make_slider_for("Num Iterations", niter, 1, self._fields["niter"],
                                            self.niter_slider_handler)
        grow_slider = self.make_slider_for("Grow", grow, 1, self._fields["grow"], self.grow_slider_handler)

        button = Button(label="Submit")
        button.on_click(self.button_handler)
        callback = CustomJS(code="""
            // alert("In custom js...");
            window.close();
        """)
        button.js_on_click(callback)

        controls = Column(order_slider, niter_slider, grow_slider, button)

        # Create a blank figure with labels
        self.p = figure(plot_width=600, plot_height=500,
                        title='Interactive Spline',
                        x_axis_label='X', y_axis_label='Y')
        # We can plot this here because it never changes
        # the overlay we plot later since it does change, giving
        # the illusion of "coloring" these points
        self.p.scatter(wave, zpt, color="blue", radius=50)

        self.recalc_spline()

        layout = row(controls, self.p)

        doc.add_root(layout)

        doc.on_session_destroyed(self.button_handler)

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
        wave = self._params["wave"]
        zpt = self._params["zpt"]
        zpt_err = self._params["zpt_err"]
        order = self._params["order"]
        niter = self._params["niter"]
        grow = self._params["grow"]
        ext = self._params["ext"]

        self.spline = astromodels.UnivariateSplineWithOutlierRemoval(wave.value, zpt.value,
                                                                     w=1. / zpt_err.value,
                                                                     order=order,
                                                                     niter=niter,
                                                                     grow=grow)

        splinex = np.linspace(min(wave), max(wave), ext.shape[0])

        if self.scatter:
            # Scatter plot exists, we just need to update it
            # mask causes it to only show the black dots where they belong for the current spline curve
            # blue dots are always present but these dots will draw over top of them
            self.scatter_source.data = {'x': wave[self.spline.mask], 'y': zpt[self.spline.mask]}
        else:
            self.scatter_source = ColumnDataSource({'x': wave[self.spline.mask], 'y': zpt[self.spline.mask]})
            self.scatter = self.p.scatter(x='x', y='y', source=self.scatter_source, color="black", radius=50)

        if self.line:
            # Spline line exists, we just need to update it
            self.line_source.data = {'x': splinex, 'y': self.spline(splinex)}
        else:
            self.line_source = ColumnDataSource({'x': splinex, 'y': self.spline(splinex)})
            self.line = self.p.line(x='x', y='y', source=self.line_source, color="red")

    def result(self):
        """
        Get the result of the user interaction.

        Returns
        -------
        :class:`astromodels.UnivariateSplineWithOutlierRemoval`
        """
        return self.spline


def interactive_spline(ext, wave, zpt, zpt_err, order, niter, grow, *, fields):
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

    server.visualizer = SplineVisualizer(params, fields)

    server.start_server()

    return server.visualizer.result()
