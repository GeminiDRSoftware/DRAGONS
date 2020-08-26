import numpy as np
from bokeh.layouts import row, column
from bokeh.models import Column, Div, Button

from geminidr.interactive import server, interactive
from geminidr.interactive.interactive import GICoordsSource, GILine, GIScatter, GIFigure, GISlider, _dequantity, \
    GIMaskedSigmadCoords, GIApertureModel, GIApertureView
from gempy.library import astromodels


__all__ = ["interactive_find_source_apertures", ]


class FindSourceAperturesModel:
    def __init__(self, ext, locations, all_limits):
        # These are the heart of the model.  The users of the model
        # register to listen to these two coordinate sets to get updates.
        # Whenever there is a call to recalc_spline, these coordinate
        # sets will update and will notify all registered listeners.
        self.mask_points = GICoordsSource()
        self.fit_line = GICoordsSource()

        self.ext = ext
        self.locations = locations
        self.all_limits = all_limits

    def update_coords(self, x, y):
        self.recalc_spline()

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
        pass


class FindSourceAperturesVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, model):
        """
        Create a spline visualizer.

        This makes a visualizer for letting a user interactively set the
        spline parameters.  The class handles some common logic for setting up
        the web document and for holding the result of the interaction.  In
        future, other visualizers will follow a similar pattern.

        Parameters
        ----------
        ext :
            Astrodata extension to visualize spline for
        coords : `~MaskedSigmadCoords`
            coordinates
        weights :
            weights
        order : int
            order to initially use for the visualization (this may be adjusted interactively)
        niter : int
            iterations to perform in doing the spline (this may be adjusted interactively)
        grow : int
            how far out to extend rejection (this may be adjusted interactively)
        min_order : int
            minimum value for order in UI
        max_order : int
            maximum value for order in UI
        min_niter : int
            minimum value for niter in UI
        max_niter : int
            maximum value for niter in UI
        min_grow : int
            minimum value for grow in UI
        max_grow : int
            maximum value for grow in UI
        """
        super().__init__()
        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.model = model

        self.aperture_model = None
        self.details = None

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

        order = 1

        order_slider = GISlider("Order", order, 1, 1, 10,
                                self.model, "order", self.model.recalc_spline)

        # Create a blank figure with labels
        self.p = GIFigure(plot_width=600, plot_height=500,
                          title='Source Apertures',
                          tools="pan,wheel_zoom,box_zoom,reset",
                          x_range=(0, self.model.ext.shape[0]),
                          y_range=(0, self.model.ext.shape[1]))

        self.aperture_model = GIApertureModel()
        aperture_view = GIApertureView(self.aperture_model, self.p)
        for loc, limits in zip(self.model.locations, self.model.all_limits):
            self.aperture_model.add_aperture(limits[0], limits[1])
        self.aperture_model.add_listener(self)

        controls = Column(order_slider.component, aperture_view.controls, self.submit_button)

        self.details = Div(text="")
        self.model.fit_line.add_coord_listener(self.update_details)
        self.model.recalc_spline()
        self.update_details(None, None)

        col = column(self.p.figure, self.details)
        layout = row(controls, col)

        doc.add_root(layout)

    def handle_aperture(self, aperture_id, start, end):
        location = (start+end)/2
        self.model.locations[aperture_id-1] = location
        self.model.all_limits[aperture_id-1] = (start, end)
        self.update_details(None, None)

    def delete_aperture(self, aperture_id):
        # ruh-roh raggy
        self.update_details(None, None)

    def update_details(self, x, y):
        text = ""
        for loc, limits in zip(self.model.locations, self.model.all_limits):
            text = text + """
                <b>Location:</b> %s<br/>
                <b>Lower Limit:</b> %s<br/>
                <b>Upper Limit:</b> %s<br/>
                <br/>
            """ % (loc, limits[0], limits[1])
        self.details.text = text

    def result(self):
        """
        Get the result of the user interaction.

        Returns
        -------
        :class:`astromodels.UnivariateSplineWithOutlierRemoval`
        """
        return self.model.locations, self.model.all_limits


def interactive_find_source_apertures(ext, locations, all_limits):
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
    weights
    order
        order for the spline calculation
    niter
        number of iterations for the spline calculation
    grow
        grow for the spline calculation
    min_order : int
        minimum value for order in UI
    max_order : int
        maximum value for order in UI
    min_niter : int
        minimum value for niter in UI
    max_niter : int
        maximum value for niter in UI
    min_grow : int
        minimum value for grow in UI
    max_grow : int
        maximum value for grow in UI
    x_axis_label : str (optional)
        Label for X-Axis
    y_axis_label : str (optional)
        Label for Y-Axis

    Returns
    -------
    :class:`astromodels.UnivariateSplineWithOutlierRemoval`
    """
    model = FindSourceAperturesModel(ext, locations, all_limits)
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)

    server.start_server()

    return fsav.result()
