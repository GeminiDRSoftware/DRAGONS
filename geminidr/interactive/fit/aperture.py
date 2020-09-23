import numpy as np
from bokeh.layouts import row, column
from bokeh.models import Div, Button
from bokeh.plotting import figure

from geminidr.interactive import server, interactive
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIApertureModel, GIApertureView, build_text_slider
from gempy.library import tracing
from geminidr.gemini.lookups import DQ_definitions as DQ


__all__ = ["interactive_find_source_apertures", ]


class FindSourceAperturesModel(GIApertureModel):
    def __init__(self, ext, profile, prof_mask, threshold, sizing_method, max_apertures):
        """
        Create an aperture model with the given initial set of inputs.

        This creates an aperture model that we can use to do the aperture fitting.
        This model allows us to tweak the various inputs using the UI and then
        recalculate the fit as often as desired.

        Parameters
        ----------
        ext : :class:`~astrodata.core.AstroData`
            extension this model should fit against
        profile : :class:`~numpy.ndarray`
        prof_mask : :class:`~numpy.ndarray`
        threshold : float
            threshold for detection
        sizing_method : str
            for example, 'peak'
        max_apertures : int
            maximum number of apertures to detect
        """
        super().__init__()

        self.ext = ext
        self.profile = profile
        self.prof_mask = prof_mask
        self.threshold = threshold
        self.sizing_method = sizing_method
        self.max_apertures = max_apertures

        self.locations = None
        self.all_limits = None

        self.recalc_listeners = list()

    def add_recalc_listener(self, listener):
        """
        Add a listener function to call when the apertures get recalculated

        Parameters
        ----------
        listener : function
            Function taking two arguments - a list of locations and a list of tuple ranges
        """
        # listener should be fn(locations, all_limits)
        self.recalc_listeners.append(listener)

    def recalc_apertures(self):
        """
        Recalculate the apertures based on the current set of fitter inputs.

        This will redo the aperture detection.  Then it calls to each registered
        listener function and calls it with a list of N locations and N limits.
        """
        max_apertures = self.max_apertures
        if not isinstance(max_apertures, int):
            max_apertures = int(max_apertures)

        # TODO: find_peaks might not be best considering we have no
        #   idea whether sources will be extended or not
        widths = np.arange(3, 20)
        peaks_and_snrs = tracing.find_peaks(self.profile, widths, mask=self.prof_mask & DQ.not_signal,
                                            variance=1.0, reject_bad=False,
                                            min_snr=3, min_frac=0.2)

        if peaks_and_snrs.size == 0:
            self.locations = []
            self.all_limits = []
        else:
            # Reverse-sort by SNR and return only the locations
            self.locations = np.array(sorted(peaks_and_snrs.T, key=lambda x: x[1],
                                      reverse=True)[:max_apertures]).T[0]
            self.all_limits = tracing.get_limits(np.nan_to_num(self.profile), self.prof_mask, peaks=self.locations,
                                                 threshold=self.threshold, method=self.sizing_method)
        for listener in self.recalc_listeners:
            listener(self.locations, self.all_limits)
        for l in self.listeners:
            for i, loclim in enumerate(zip(self.locations, self.all_limits)):
                loc = loclim[0]
                lim = loclim[1]
                l.handle_aperture(i+1, loc, lim[0], lim[1])

    def add_aperture(self, location, start, end):
        aperture_id = len(self.locations)+1
        np.append(self.locations, location)
        np.append(self.all_limits, (start, end))
        for l in self.listeners:
            l.handle_aperture(aperture_id, location, start, end)
        return aperture_id

    def adjust_aperture(self, aperture_id, location, start, end):
        """
        Adjust an existing aperture by ID to a new range.
        This will alert all subscribed listeners.

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to adjust
        location : float
            X coodinate of the new aperture location
        start : float
            X coordinate of the new start of range
        end : float
            X coordiante of the new end of range

        """
        if location < start or location > end:
            raise ValueError("Location of aperture must be between start and end")
        self.locations[aperture_id-1] = location
        self.all_limits[aperture_id-1] = [start, end]
        for l in self.listeners:
            l.handle_aperture(aperture_id, location, start, end)

    def delete_aperture(self, aperture_id):
        """
        Delete an aperture by ID.

        Parameters
        ----------
        aperture_id : int
            Aperture id to delete
        """
        np.delete(self.locations, aperture_id-1)
        np.delete(self.all_limits, aperture_id-1)
        for listener in self.listeners:
            listener.delete_aperture(aperture_id)

    def clear_apertures(self):
        """
        Remove all apertures, calling delete on the listeners for each.
        """
        for iap in range(len(self.locations), 1, -1):
            for listener in self.listeners:
                listener.delete_aperture(iap)
        self.locations = []
        self.all_limits = []

    def get_profile(self):
        return self.profile


class FindSourceAperturesVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, model):
        """
        Create a view for finding apertures with the given
        :class:`geminidr.interactive.fit.aperture.FindSourceAperturesModel`

        Parameters
        ----------
        model : :class:`geminidr.interactive.fit.aperture.FindSourceAperturesModel`
            Model to use for tracking the input parameters and recalculating fresh sets as needed
        """
        super().__init__()
        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.model = model

        self.details = None
        self.fig = None

    def clear_and_recalc(self):
        """
        Clear apertures and recalculate a new set.
        """
        self.model.clear_apertures()
        self.model.recalc_apertures()

    def add_aperture(self):
        """
        Add a new aperture in the middle of the current display area.

        This is used when the user adds an aperture.  We don't know where
        they want it yet so we just place one in the screen center.

        """
        x = (self.fig.x_range.start + self.fig.x_range.end) / 2
        self.model.add_aperture(x, x)
        self.update_details()

    def visualize(self, doc):
        """
        Build the visualization in bokeh in the given browser document.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            Bokeh provided document to add visual elements to
        """
        super().visualize(doc)

        max_apertures_slider = build_text_slider("Max Apertures", self.model.max_apertures, 1, 1, 20,
                                                 self.model, "max_apertures", self.clear_and_recalc,
                                                 throttled=True)
        threshold_slider = build_text_slider("Threshold", self.model.threshold, 0.01, 0, 1,
                                             self.model, "threshold", self.clear_and_recalc,
                                             throttled=True)

        # Create a blank figure with labels
        self.fig = figure(plot_width=600, plot_height=500,
                          title='Source Apertures',
                          tools="pan,wheel_zoom,box_zoom,reset",
                          x_range=(0, self.model.profile.shape[0]))

        aperture_view = GIApertureView(self.model, self.fig)
        self.model.add_listener(self)

        self.fig.line(x=range(self.model.profile.shape[0]), y=self.model.profile, color="black")

        add_button = Button(label="Add Aperture")
        add_button.on_click(self.add_aperture)

        helptext = Div()
        controls = column(children=[max_apertures_slider, threshold_slider,
                          aperture_view.controls, add_button, self.submit_button, helptext])

        self.details = Div(text="")
        self.model.recalc_apertures()
        self.update_details()

        col = column(self.fig, self.details)
        layout = row(controls, col)

        Controller(self.fig, self.model, None, helptext)

        doc.add_root(layout)

    def handle_aperture(self, aperture_id, location, start, end):
        """
        Handle updated aperture information.

        This is called when a given aperture has a change
        to it's start and end location.  The model is
        updated and then the text describing the apertures
        is updated

        Parameters
        ----------
        aperture_id : int
            ID of the aperture to add/update
        location : float
            new location of aperture
        start : float
            new start of aperture
        end : float
            new end of aperture
        """
        if aperture_id == len(self.model.locations)+1:
            self.model.locations = np.append(self.model.locations, location)
            self.model.all_limits.append((start, end))
        else:
            self.model.locations[aperture_id-1] = location
            self.model.all_limits[aperture_id-1] = (start, end)
        self.update_details()

    def delete_aperture(self, aperture_id):
        """
        Delete an aperture by ID

        Parameters
        ----------
        aperture_id : int
            `int` id of the aperture to delete
        """
        self.model.locations = np.delete(self.model.locations, aperture_id-1)
        del self.model.all_limits[aperture_id-1]
        self.update_details()

    def update_details(self):
        """
        Update the details text area with the latest aperture data.

        """
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
        Get the result of the find.

        Returns
        -------
            list of float, list of tuple : list of locations and list of the limits as tuples

        """
        return self.model.locations, self.model.all_limits


def interactive_find_source_apertures(ext, profile, prof_mask, threshold, sizing_method, max_apertures):
    """
    Perform an interactive find of source apertures with the given initial parameters.

    This will do all the bokeh initialization and display a UI for interactively modifying
    parameters from their initial values.  The user can also interact directly with the
    found aperutres as desired.  When the user hits the `Submit` button, this method will
    return the results of the find to the caller.

    Parameters
    ----------
    ext : :class:`~astrodata.core.AstroData`
        extension this model should fit against
    profile : :class:`~numpy.ndarray`
    prof_mask : :class:`~numpy.ndarray`
    threshold : float
        threshold for detection
    sizing_method : str
        for example, 'peak'
    max_apertures : int
        maximum number of apertures to detect
    """
    if max_apertures is None:
        max_apertures = 50
    model = FindSourceAperturesModel(ext, profile, prof_mask, threshold, sizing_method,
                                     max_apertures)
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)

    server.start_server()

    return fsav.result()
