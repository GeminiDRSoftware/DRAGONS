import numpy as np
from astropy.modeling import models, fitting
from bokeh.layouts import row
from bokeh.models import Button, Column, Panel, Tabs, ColumnDataSource, Div

from geminidr.interactive import server, interactive
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIScatter, GILine, GICoordsSource, GICoordsListener, \
    GIBandModel, GIBands, GIApertureModel, GIApertureView, GIFigure, GIDifferencingCoords, GISlider, GIMaskedCoords
from gempy.library import astromodels


class ChebyshevModel(GICoordsSource, GICoordsListener):
    def __init__(self, order, location, dispaxis, sigma_clip, coords, spectral_coords, ext):
        super().__init__()
        self.order = order
        self.location = location
        self.dispaxis = dispaxis
        self.sigma_clip = sigma_clip
        self.coords = coords
        self.spectral_coords = spectral_coords
        self.ext = ext
        self.m_final = None
        self.model_dict = None
        self.x = []
        self.y = []

        # do this last since it will trigger an update, which triggers a recalc
        self.coords.add_gilistener(self)

    def giupdate(self, x_coords, y_coords):
        self.x = x_coords
        self.y = y_coords
        self.recalc_chebyshev()

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
        # in_coords = self.in_coords
        ext = self.ext

        m_init = models.Chebyshev1D(degree=order, c0=location,
                                    domain=[0, ext.shape[dispaxis] - 1])
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                   sigma_clip, sigma=3)
        try:
            x = self.x
            y = self.y
            self.m_final, _ = fit_it(m_init, x, y)
        except (IndexError, np.linalg.linalg.LinAlgError):
            # This hides a multitude of sins, including no points
            # returned by the trace, or insufficient points to
            # constrain the request order of polynomial.
            self.m_final = models.Chebyshev1D(degree=0, c0=location,
                                              domain=[0, ext.shape[dispaxis] - 1])
        self.model_dict = astromodels.chebyshev_to_dict(self.m_final)

        self.ginotify(self.spectral_coords, self.m_final(self.spectral_coords))


class DifferencingModel(GICoordsSource, GICoordsListener):
    def __init__(self, cmodel):
        super().__init__()
        self.cmodel = cmodel
        self.data_x_coords = cmodel.coords.x_coords[cmodel.coords.mask]
        self.data_y_coords = cmodel.coords.y_coords[cmodel.coords.mask]
        cmodel.add_gilistener(self)
        # cmodel.coords.add_gilistener(self)

    def giupdate(self, x_coords, y_coords):
        # hacking this in, should not use internal fields of the masked coords, need to
        # refine how I do listeners (maybe just a fn instead of a class, or maybe with a source name)
        # though also, this whole class should be refactored to not know about Chebyshev
        x = self.cmodel.coords.x_coords[self.cmodel.coords.mask]
        y = self.cmodel.coords.y_coords[self.cmodel.coords.mask] - self.cmodel.m_final(x)
        self.ginotify(x, y)


class Chebyshev1DVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, x, y, model, min_order, max_order):
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
            raise ValueError("Chebyshev1DVisualizer requires ChebyshevModel")

        self.x = x
        self.y = y
        self.model = model

        self.min_order = min_order
        self.max_order = max_order

        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.p = None
        self.spline = None
        self.scatter = None
        self.masked_scatter = None
        self.scatter_touch = None
        self.line = None
        self.scatter_source = None
        self.line_source = None
        self.spectral_coords = None
        self.model_dict = None
        self.m_final = None

        self.scatter2 = None
        self.line2 = None

        self.controls = None

    def mask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection() # source.selected.indices.clear()
        self.masked_scatter.clear_selection()
        self.model.coords.addmask(indices)
        # self.masked_scatter.replot()

    def unmask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection() # source.selected.indices.clear()
        self.masked_scatter.clear_selection()
        self.model.coords.unmask(indices)
        # self.masked_scatter.replot()

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

        super().visualize(doc)

        # Just sandboxing a basic band UI for the x ranges from Kathleen's demo
        band_model = GIBandModel()
        # Just sandboxing a sample Aperture UI
        aperture_model = GIApertureModel()

        order_slider = GISlider("Order", self.model.order, 1, self.min_order, self.max_order,
                                self.model, "order", self.model.recalc_chebyshev)

        mask_button = Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        # Add custom tooling
        source = ColumnDataSource(data=dict(x=[], y=[]))

        # Create a blank figure with labels
        p = GIFigure(plot_width=600, plot_height=500,
                     title='Interactive Chebyshev',
                     x_axis_label='X', y_axis_label='Y',
                     tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                     band_model=band_model, aperture_model=aperture_model)

        self.p = p

        # placeholder, just seeing the custom tool is working
        p.figure.line('x', 'y', source=source)

        # p2 is just to show that we can have multiple tabs with plots running off the same dataset
        # TODO make plots like the other IRAF options we were shown
        p2 = GIFigure(plot_width=600, plot_height=500,
                      title='Interactive Chebyshev (tab 2)',
                      x_axis_label='X', y_axis_label='Y',
                     band_model=band_model, aperture_model=aperture_model)

        self.scatter = GIScatter(p, self.x, self.y, color="red", radius=5)
        self.masked_scatter = GIScatter(p, self.x, self.y, color="blue", radius=5)
        self.model.coords.add_gilistener(self.masked_scatter)
        self.line = GILine(p)

        differencing_model = DifferencingModel(self.model)

        self.line2 = GILine(p2)

        # helptext is where the Controller will put help messages for the end user
        # This controls area is a vertical set of UI controls we are placing on the left
        # side of the UI
        helptext = Div(text="")
        self.controls = Column(order_slider.component, self.submit_button, mask_button, unmask_button, helptext)

        # The line will update against the model
        # Our second line, in tab 2, will update vs the difference (which, in turn, listens to the model)
        # both will fire an update when we recalculate the cheybshev
        self.model.add_gilistener(self.line)
        differencing_model.add_gilistener(self.line2)

        # recalculate the chebyshev, causing the data updates to fire and update the UI as well
        self.model.recalc_chebyshev()

        # add the two plots as tabs and place them with controls to the left
        tab1 = Panel(child=p.figure, title="Chebyshev Fit")
        tab2 = Panel(child=p2.figure, title="Chebyshev Differential")
        tabs = Tabs(tabs=[tab1, tab2], name="tabs")
        layout = row(self.controls, tabs)

        # setup controller for key commands
        controller = Controller(self.p, aperture_model, band_model, helptext)

        doc.add_root(layout)


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
    order
        order for the spline calculation
    location
    dispaxis
    sigma_clip
    in_coords
    spectral_coords
    min_order : int
        minimum value for order slider
    max_order : int
        maximum value for order slider, or None to infer

    Returns
    -------
        dict, :class:`models.Chebyshev1D`
    """
    masked_coords = GIMaskedCoords(in_coords[1-dispaxis], in_coords[dispaxis])
    model = ChebyshevModel(order, location, dispaxis, sigma_clip,
                           masked_coords, spectral_coords, ext)
    server.visualizer = Chebyshev1DVisualizer(in_coords[1-dispaxis], in_coords[dispaxis], model, min_order, max_order)

    server.start_server()

    server.visualizer = None

    return model.model_dict, model.m_final
