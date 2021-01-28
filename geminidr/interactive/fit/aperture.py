import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (Button, CheckboxGroup, Div, Slider, Spacer, Spinner,
                          TextInput)
from bokeh.plotting import figure

from geminidr.interactive import interactive, server
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIApertureModel, GIApertureView
from gempy.library.tracing import find_apertures

__all__ = ["interactive_find_source_apertures", ]


def TextInputLine(title, value, handler):
    spinner = Spinner(value=value, width=64)
    spinner.on_change("value", handler)
    return row([Div(text=title, align='end'),
                Spacer(width_policy='max'),
                spinner])


def TextSlider(title, value, start, end, step, handler, is_float=False):
    spinner = Spinner(value=value, width=64, step=step)
    slider = Slider(start=start, end=end, value=value, step=step,
                    title=title, width=256)
    in_update = False

    def _handler(attr, old, new):
        nonlocal in_update
        if in_update:
            # To avoid triggering the handler with both spinner and slider
            return
        in_update = True
        spinner.value = new
        slider.value = new
        handler(attr, old, new)
        in_update = False

    spinner.on_change("value", _handler)
    slider.on_change("value_throttled", _handler)

    return row([slider,
                Spacer(width_policy='max'),
                spinner])


def CheckboxLine(title, value, handler):
    checkbox = CheckboxGroup(labels=[""], active=[0] if value else [],
                             width=40, width_policy='fixed', align='center')
    checkbox.on_click(handler)
    return row([Div(text=title, align='end'),
                Spacer(width_policy='max'),
                checkbox])


class FindSourceAperturesModel(GIApertureModel):
    def __init__(self, ext, **aper_params):
        """
        Create an aperture model with the given initial set of inputs.

        This creates an aperture model that we can use to do the aperture
        fitting.  This model allows us to tweak the various inputs using the
        UI and then recalculate the fit as often as desired.

        """
        super().__init__()
        self.ext = ext
        self._aper_param_names = (
            'direction', 'max_apertures', 'min_sky_region', 'percentile',
            'sec_regions', 'sizing_method', 'threshold', 'use_snr')
        # We need proper attributes for params but they will be set by widgets
        for name in self._aper_param_names:
            setattr(self, name, aper_params[name])
        self.recalc_apertures()

    @property
    def aper_params(self):
        return {name: getattr(self, name) for name in self._aper_param_names}

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
        self.locations, self.all_limits, self.profile = find_apertures(
            self.ext, **self.aper_params)

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
        model : :class:`FindSourceAperturesModel`
            Model to use for tracking the input parameters and recalculating
            fresh sets as needed
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

        def _maxaper_handler(attr, old, new):
            self.model.max_apertures = int(new) if new is not None else None
            self.clear_and_recalc()

        max_apertures_widget = TextInputLine(
            title="Max Apertures (no limit by default)",
            value=self.model.max_apertures,
            handler=_maxaper_handler
        )

        def _percentile_handler(attr, old, new):
            self.model.percentile = new
            self.clear_and_recalc()

        percentile_slider = TextSlider(
            title="Percentile (use mean if no value)",
            value=self.model.percentile,
            start=0, end=100, step=1,
            handler=_percentile_handler
        )

        def _min_sky_region_handler(attr, old, new):
            self.model.min_sky_region = new
            self.clear_and_recalc()

        min_sky_region_widget = TextInputLine(
            title="Min sky region",
            value=self.model.min_sky_region,
            handler=_min_sky_region_handler
        )

        def _use_snr_handler(new):
            self.model.use_snr = 0 in new
            self.clear_and_recalc()

        use_snr_widget = CheckboxLine(
            title="Use S/N ratio ?",
            value=self.model.use_snr,
            handler=_use_snr_handler
        )

        def _threshold_handler(attr, old, new):
            self.model.threshold = new
            self.clear_and_recalc()

        threshold_slider = TextSlider(
            title="Threshold",
            value=self.model.threshold,
            start=0, end=1, step=0.01,
            handler=_threshold_handler,
            is_float=True
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
                                    max_apertures_widget,
                                    percentile_slider,
                                    min_sky_region_widget,
                                    use_snr_widget,
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
            list of float, list of tuple :
                list of locations and list of the limits as tuples

        """
        return self.model.locations, self.model.all_limits


def interactive_find_source_apertures(ext, **kwargs):
    """
    Perform an interactive find of source apertures with the given initial
    parameters.

    This will do all the bokeh initialization and display a UI for
    interactively modifying parameters from their initial values.  The user can
    also interact directly with the found aperutres as desired.  When the user
    hits the `Submit` button, this method will return the results of the find
    to the caller.

    """
    model = FindSourceAperturesModel(ext, **kwargs)
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)
    server.start_server()
    return fsav.result()
