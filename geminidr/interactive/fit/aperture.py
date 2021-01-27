import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Button, Div, TextInput
from bokeh.plotting import figure

from geminidr.interactive import interactive, server
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import (GIApertureModel, GIApertureView,
                                              build_text_slider)
from gempy.library.tracing import find_apertures

__all__ = ["interactive_find_source_apertures", ]


class FindSourceAperturesModel(GIApertureModel):
    def __init__(self, ext, masked_data, prof_mask, **aper_params):
        """
        Create an aperture model with the given initial set of inputs.

        This creates an aperture model that we can use to do the aperture
        fitting.  This model allows us to tweak the various inputs using the
        UI and then recalculate the fit as often as desired.

        """
        super().__init__()
        self.ext = ext
        self.masked_data = masked_data
        self.prof_mask = prof_mask

        self.direction = aper_params['direction']
        self.percentile = aper_params['percentile']
        self.threshold = aper_params['threshold']
        self.sizing_method = aper_params['sizing_method']
        self.max_apertures = aper_params['max_apertures']

        if self.max_apertures is None:
            self.max_apertures = 50

        self.recalc_apertures()

    def find_closest(self, x):
        aperture_id = None
        location = None
        delta = None
        for i, loc in enumerate(self.locations):
            new_delta = abs(loc-x)
            if delta is None or new_delta < delta:
                aperture_id = i+1
                location = loc
                delta = new_delta
        return (aperture_id, location,
                self.all_limits[aperture_id-1][0],
                self.all_limits[aperture_id-1][1])

    def recalc_apertures(self):
        """
        Recalculate the apertures based on the current set of fitter inputs.

        This will redo the aperture detection.  Then it calls to each
        registered listener function and calls it with a list of N locations
        and N limits.

        """
        max_apertures = self.max_apertures
        if not isinstance(max_apertures, int):
            max_apertures = int(max_apertures)

        self.locations, self.all_limits, self.profile = find_apertures(
            self.masked_data, self.prof_mask, self.percentile,
            self.max_apertures, self.threshold, self.sizing_method,
            self.direction)

        for listener in self.listeners:
            for i, loclim in enumerate(zip(self.locations, self.all_limits)):
                loc = loclim[0]
                lim = loclim[1]
                listener.handle_aperture(i+1, loc, lim[0], lim[1])

    def add_aperture(self, location, start, end):
        aperture_id = len(self.locations)+1
        np.append(self.locations, location)
        np.append(self.all_limits, (start, end))
        for listener in self.listeners:
            listener.handle_aperture(aperture_id, location, start, end)
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
        for listener in self.listeners:
            listener.handle_aperture(aperture_id, location, start, end)

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
        super().__init__(title='Find Source Apertures')
        self.model = model

        self.details = None
        self.fig = None

    def clear_and_recalc(self, *args):
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
        self.model.add_aperture(x, x, x)
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

        current_file = TextInput(title="Current file:",
                                 value=self.model.ext.filename,
                                 disabled=True,
                                 background='white')

        max_apertures_slider = build_text_slider(
            title="Max Apertures",
            value=self.model.max_apertures,
            step=1,
            min_value=1,
            max_value=20,
            obj=self.model,
            attr="max_apertures",
            handler=self.clear_and_recalc,
            throttled=True
        )
        percentile_slider = build_text_slider(
            title="Percentile",
            value=self.model.percentile or 95,
            step=1,
            min_value=0,
            max_value=100,
            obj=self.model,
            attr="percentile",
            handler=self.clear_and_recalc,
            throttled=True
        )
        threshold_slider = build_text_slider(
            title="Threshold",
            value=self.model.threshold,
            step=0.01,
            min_value=0,
            max_value=1,
            obj=self.model,
            attr="threshold",
            handler=self.clear_and_recalc,
            throttled=True
        )

        # Create a blank figure with labels
        self.fig = figure(
            # plot_width=600,
            plot_height=500,
            title='Source Apertures',
            tools="pan,wheel_zoom,box_zoom,reset",
            x_range=(0, self.model.profile.shape[0])
        )
        self.fig.height_policy = 'fixed'
        self.fig.width_policy = 'fit'

        aperture_view = GIApertureView(self.model, self.fig)
        self.model.add_listener(self)

        self.fig.line(x=range(self.model.profile.shape[0]),
                      y=self.model.profile,
                      color="black")

        add_button = Button(label="Add Aperture")
        add_button.on_click(self.add_aperture)

        helptext = Div()
        controls = column(children=[current_file,
                                    max_apertures_slider,
                                    percentile_slider,
                                    threshold_slider,
                                    aperture_view.controls,
                                    add_button,
                                    self.submit_button,
                                    helptext])

        self.details = Div(text="")
        self.model.recalc_apertures()
        self.update_details()

        col = column(self.fig, self.details)
        col.sizing_mode = 'scale_width'
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


def interactive_find_source_apertures(ext, masked_data, prof_mask, **kwargs):
    """
    Perform an interactive find of source apertures with the given initial
    parameters.

    This will do all the bokeh initialization and display a UI for
    interactively modifying parameters from their initial values.  The user can
    also interact directly with the found aperutres as desired.  When the user
    hits the `Submit` button, this method will return the results of the find
    to the caller.

    """
    model = FindSourceAperturesModel(ext, masked_data, prof_mask, **kwargs)
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)
    server.start_server()
    return fsav.result()
