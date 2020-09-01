import numpy as np
from astropy.modeling import models, fitting
from bokeh.layouts import row
from bokeh.models import Button, Column, Panel, Tabs, Div

from geminidr.interactive import server, interactive
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GILine, GICoordsSource, \
    GIBandModel, GIApertureModel, GIFigure, GISlider, GIMaskedSigmadCoords, \
    GIModelSource, GIDifferencingModel, GIMaskedSigmadScatter
from gempy.library import astromodels


__all__ = ["interactive_chebyshev2d", ]


class Chebyshev2DModel(GICoordsSource, GIModelSource):
    def __init__(self, spectral_order, spatial_order, dispaxis, sigma_clip, coords, in_coords, ref_coords, ext):
        super().__init__()
        GIModelSource.__init__(self)
        self.spectral_order = spectral_order
        self.spatial_order = spatial_order
        self.dispaxis = dispaxis
        self.sigma_clip = sigma_clip
        self.sigma=3
        self.coords = coords
        self.in_coords = in_coords
        self.ref_coords = ref_coords
        self.ext = ext
        self.m_final = None
        self.model_dict = None
        self.x = []
        self.y = []

        # do this last since it will trigger an update, which triggers a recalc
        self.coords.add_mask_listener(self.update_coords)

    def update_coords(self, x_coords, y_coords):
        # The masked coordinates changed, so update our copy and recalculate the model
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
        orders = (self.spectral_order, self.spatial_order)
        dispaxis = self.dispaxis
        sigma_clip = self.sigma_clip
        ext = self.ext

        #-----
        self.m_init = models.Chebyshev2D(x_degree=orders[1 - dispaxis],
                                    y_degree=orders[dispaxis],
                                    x_domain=[0, ext.shape[1]],
                                    y_domain=[0, ext.shape[0]])
        self.fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                   sigma_clip, sigma=3)
        self.m_final, self.fit_mask = self.fit_it(self.m_init, *self.in_coords, self.ref_coords[1 - dispaxis])
        self.m_inverse, self.masked = self.fit_it(self.m_init, *self.ref_coords, self.in_coords[1 - dispaxis])
        # self.coords.set_sigma(fit_mask)
        #-----

        self.model_dict = astromodels.chebyshev_to_dict(self.m_final)

        # notify listeners of new x/y plot data based on our model function
        self.notify_coord_listeners(self.in_coords[0], self.m_final(self.in_coords[0], self.in_coords[1]))
        # notify model listeners that our model function has changed
        # self.notify_model_listeners()

    def model_calculate(self, x):
        # we need this wrapper since self.m_final changes with each recalc
        return self.m_final(x)


class Chebyshev2DVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, x, y, model, min_order, max_order):
        """
        Create a chebyshev2D visualizer.

        This makes a visualizer for letting a user interactively set the
        Chebyshev parameters.  The class handles some common logic for setting up
        the web document and for holding the result of the interaction.  In
        future, other visualizers will follow a similar pattern.

        Parameters
        ----------
        """
        super().__init__()

        if not isinstance(model, Chebyshev2DModel):
            raise ValueError("Chebyshev2DVisualizer requires Chebyshev2DModel")

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
        self.sigma_scatter = None
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
        self.scatter.clear_selection()
        self.model.coords.addmask(indices)

    def unmask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.unmask(indices)

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

        spectral_order_slider = GISlider("Spectral Order", self.model.spectral_order, 1, self.min_order, self.max_order,
                                         self.model, "spectral_order", self.model.recalc_chebyshev)
        spatial_order_slider = GISlider("Spatial Order", self.model.spectral_order, 1, self.min_order, self.max_order,
                                        self.model, "spatial_order", self.model.recalc_chebyshev)
        sigma_slider = GISlider("Sigma", self.model.sigma, 0.1, 2, 10,
                                self.model, "sigma", self.model.recalc_chebyshev)

        mask_button = Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        # Create a blank figure with labels
        p = GIFigure(plot_width=600, plot_height=500,
                     title='Interactive Chebyshev',
                     x_axis_label='X', y_axis_label='Y',
                     tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap")

        self.p = p

        self.scatter = GIMaskedSigmadScatter(p, self.model.coords)

        self.line = GILine(p)
        self.model.add_coord_listener(self.line.update_coords)

        # p2 goes in tab 2 and shows the difference between the data y values and the model calculated values
        p2 = GIFigure(plot_width=600, plot_height=500,
                      title='Model Differential',
                      x_axis_label='X', y_axis_label='Y')
        self.line2 = GILine(p2)
        # differencing_model = GIDifferencingModel(self.model.coords, self.model, self.model.model_calculate)
        # differencing_model.add_coord_listener(self.line2.update_coords)

        self.controls = Column(spectral_order_slider.component, spatial_order_slider.component, sigma_slider.component,
                               self.submit_button, mask_button, unmask_button)

        # recalculate the chebyshev, causing the data updates to fire and update the UI as well
        self.model.recalc_chebyshev()

        # add the two plots as tabs and place them with controls to the left
        tab1 = Panel(child=p.figure, title="Chebyshev Fit")
        tab2 = Panel(child=p2.figure, title="Chebyshev Differential")
        tabs = Tabs(tabs=[tab1, tab2], name="tabs")
        layout = row(self.controls, tabs)

        doc.add_root(layout)


def interactive_chebyshev2d(ext,  orders, dispaxis, sigma_clip, in_coords, ref_coords,
                            min_order, max_order):
    masked_coords = GIMaskedSigmadCoords(in_coords[1-dispaxis], in_coords[dispaxis])
    model = Chebyshev2DModel(orders[0], orders[1], dispaxis, sigma_clip,
                             masked_coords, in_coords, ref_coords, ext)
    server.set_visualizer(Chebyshev2DVisualizer(in_coords[1-dispaxis], in_coords[dispaxis],
                                                model, min_order, max_order))

    server.start_server()

    server.set_visualizer(None)

    return model.m_init, model.fit_it, model.m_final, model.m_inverse, model.masked
