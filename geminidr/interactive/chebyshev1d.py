import numpy as np
from astropy.modeling import models, fitting
from bokeh.layouts import row
from bokeh.models import Button, Column, ColumnDataSource, CustomJS, Slider, ColorPicker
from bokeh.plotting import figure

from geminidr.interactive import server
from gempy.library import astromodels


class Chebyshev1DVisualizer(server.PrimitiveVisualizer):
    def __init__(self, params, fields):
        """
        Create a chebyshev1D visualizer.

        This makes a visualizer for letting a user interactively set the
        Chebyshev parameters.  The class handles some common logic for setting up
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
        self.scatter_touch = None
        self.line = None
        self.scatter_source = None
        self.line_source = None
        self.spectral_coords = None
        self.model_dict = None
        self.m_final = None

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
        self._params["order"] = new
        self.recalc_chebyshev()

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
        order = self._params["order"]
        self.spectral_coords = self._params["spectral_coords"]

        order_slider = self.make_slider_for("Order", order, 1, self._fields["trace_order"], self.order_slider_handler)

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

        self.line_source = ColumnDataSource({'x': [], 'y': []})
        self.line = self.p.line(x='x', y='y', source=self.line_source, color="red")

        line_slider = Slider(start=1, end=32, value=1, step=1, title="Line Size")
        line_slider.width = 256
        callback_line = CustomJS(args=dict(line=self.line),
                                 code="""
                                 line.glyph.line_width = cb_obj.value;
                                 """)
        line_slider.js_on_change('value', callback_line)

        line_color_picker = ColorPicker(color="red", title="Line Color:", width=200)
        callback_line_color = CustomJS(args=dict(line=self.line),
                                       code="""
                                       line.glyph.line_color = cb_obj.color;
                                       line.glyph.fill_color = cb_obj.color;
                                       """)
        line_color_picker.js_on_change('color', callback_line_color)

        controls = Column(order_slider, button,
                          line_slider, line_color_picker)

        self.recalc_chebyshev()

        layout = row(controls, self.p)

        doc.add_root(layout)

        doc.on_session_destroyed(self.button_handler)

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
        order = self._params["order"]
        location = self._params["location"]
        dispaxis = self._params["dispaxis"]
        sigma_clip = self._params["sigma_clip"]
        in_coords = self._params["in_coords"]
        ext = self._params["ext"]

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
        self.line_source.data = {'x': self.spectral_coords, 'y': self.m_final(self.spectral_coords)}

    def result(self):
        """
        Get the result of the user interaction.

        Returns dictionary of model parameters as expected by the primitive, and the model function
        -------
        dict, :class:`models.Chebyshev1D`
        """
        return self.model_dict, self.m_final


def interactive_chebyshev(ext,  order, location, dispaxis, sigma_clip, in_coords, spectral_coords, *, fields):
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
        dict, :class:`models.Chebyshev1D`
    """
    params = dict(
        ext=ext,
        order=order,
        location=location,
        dispaxis=dispaxis,
        sigma_clip=sigma_clip,
        in_coords=in_coords,
        spectral_coords=spectral_coords
    )

    server.visualizer = Chebyshev1DVisualizer(params, fields)

    server.start_server()

    return server.visualizer.result()
