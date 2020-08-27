import numpy as np
from bokeh.layouts import row, column
from bokeh.models import Column, Div, Button

from geminidr.interactive import server, interactive
from geminidr.interactive.interactive import GICoordsSource, GILine, GIScatter, GIFigure, GISlider, _dequantity, \
    GIMaskedSigmadCoords, GIApertureModel, GIApertureView
from gempy.library import astromodels


__all__ = ["interactive_find_source_apertures", ]


class FindSourceAperturesModel:
    def __init__(self, ext, profile, prof_mask, tracing, threshold, sizing_method, peaks_and_snrs, max_apertures):

        self.ext = ext
        self.profile = profile
        self.prof_mask = prof_mask
        self.tracing = tracing
        self.threshold = threshold
        self.sizing_method = sizing_method
        self.peaks_and_snrs = peaks_and_snrs
        self.max_apertures = max_apertures

        self.locations = None
        self.all_limits = None

        self.listeners = list()

    def add_listener(self, l):
        # l should be fn(locations, all_limits)
        self.listeners.append(l)

    def recalc_apertures(self):
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
        max_apertures = self.max_apertures
        if not isinstance(max_apertures, int):
            max_apertures = int(max_apertures)
        # Reverse-sort by SNR and return only the locations
        self.locations = np.array(sorted(self.peaks_and_snrs.T, key=lambda x: x[1],
                                    reverse=True)[:max_apertures]).T[0]
        self.all_limits = self.tracing.get_limits(np.nan_to_num(self.profile), self.prof_mask, peaks=self.locations,
                                        threshold=self.threshold, method=self.sizing_method)
        for l in self.listeners:
            l(self.locations, self.all_limits)


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

        max_apertures_slider = GISlider("Max Apertures", self.model.max_apertures, 1, 1, 20,
                                        self.model, "max_apertures", self.model.recalc_apertures,
                                        throttled=True)

        # Create a blank figure with labels
        self.p = GIFigure(plot_width=600, plot_height=500,
                          title='Source Apertures',
                          tools="pan,wheel_zoom,box_zoom,reset",
                          x_range=(0, self.model.profile.shape[0]))

        self.aperture_model = GIApertureModel()
        aperture_view = GIApertureView(self.aperture_model, self.p)
        self.aperture_model.add_listener(self)

        def apl(locations, all_limits):
            self.aperture_model.clear_apertures()
            for loc, limits in zip(locations, all_limits):
                self.aperture_model.add_aperture(limits[0], limits[1])
        self.model.add_listener(apl)
        # self.model.recalc_apertures()

        line = GILine(self.p, range(self.model.profile.shape[0]), self.model.profile, color="black")
        controls = Column(max_apertures_slider.component, aperture_view.controls, self.submit_button)

        self.details = Div(text="")
        self.model.recalc_apertures()
        self.update_details()

        col = column(self.p.figure, self.details)
        layout = row(controls, col)

        doc.add_root(layout)

    def handle_aperture(self, aperture_id, start, end):
        location = (start+end)/2
        self.model.locations[aperture_id-1] = location
        self.model.all_limits[aperture_id-1] = (start, end)
        self.update_details()

    def delete_aperture(self, aperture_id):
        # ruh-roh raggy
        self.update_details()

    def update_details(self):
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


def interactive_find_source_apertures(ext, profile, prof_mask, tracing, threshold, sizing_method, peaks_and_snrs, max_apertures):
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
    if max_apertures is None:
        max_apertures = 50
    model = FindSourceAperturesModel(ext, profile, prof_mask, tracing, threshold, sizing_method,
                                     peaks_and_snrs, max_apertures)
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)

    server.start_server()

    return fsav.result()
