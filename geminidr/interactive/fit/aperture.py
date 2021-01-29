import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (Button, CheckboxGroup, Div, Select, Slider, Spacer,
                          Spinner, TextInput)
from bokeh.plotting import figure

from geminidr.interactive import interactive, server
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIApertureModel, GIApertureView
from gempy.library.tracing import find_apertures

__all__ = ["interactive_find_source_apertures", ]


class CustomWidget:
    def __init__(self, title, model, attr, handler=None):
        self.title = title
        self.attr = attr
        self.model = model
        self._handler = handler

    @property
    def value(self):
        return getattr(self.model, self.attr)

    def handler(self, attr, old, new):
        print(f'Calling handler for {self.__class__.__name__}, {self.attr}, '
              f'{attr}, {new}')
        if self._handler is not None:
            self._handler(new)
        else:
            setattr(self.model, self.attr, new)


class TextInputLine(CustomWidget):
    def build(self):
        self.spinner = Spinner(value=self.value, width=64)
        self.spinner.on_change("value", self.handler)
        return row([Div(text=self.title, align='end'),
                    Spacer(width_policy='max'),
                    self.spinner])

    def reset(self):
        self.spinner.value = self.value


class TextSlider(CustomWidget):
    def build(self, start, end, step):
        self.in_update = False
        self.spinner = Spinner(value=self.value, width=64, step=step)
        self.slider = Slider(start=start, end=end, value=self.value, step=step,
                             title=self.title, width=256)
        self.spinner.on_change("value", self.handler)
        self.slider.on_change("value", self.handler)

        return row([self.slider,
                    Spacer(width_policy='max'),
                    self.spinner])

    def reset(self):
        self.spinner.value = self.value
        self.slider.value = self.value

    def handler(self, attr, old, new):
        if self.in_update:
            # To avoid triggering the handler with both spinner and slider
            return
        self.in_update = True
        self.spinner.value = new
        self.slider.value = new
        super().handler(attr, old, new)
        self.in_update = False


class CheckboxLine(CustomWidget):
    def build(self):
        self.checkbox = CheckboxGroup(labels=[""],
                                      active=[0] if self.value else [],
                                      width=40, width_policy='fixed',
                                      align='center')
        self.checkbox.on_click(self.handler)
        return row([Div(text=self.title, align='end'),
                    Spacer(width_policy='max'),
                    self.checkbox])

    def reset(self):
        self.checkbox.active = [0] if self.value else []

    def handler(self, new):
        super().handler(None, None, new)


class SelectLine(CustomWidget):
    def build(self):
        self.select = Select(value=self.value, options=["peak", "integral"],
                             width=128)
        self.select.on_change("value", self.handler)
        return row([Div(text=self.title, align='end'),
                    Spacer(width_policy='max'),
                    self.select])

    def reset(self):
        self.select.value = self.value


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
        self._aper_params = aper_params.copy()
        self.reset()
        self.recalc_apertures()

    @property
    def aper_params(self):
        """Return the actual param dict from the instance attributes."""
        return {name: getattr(self, name) for name in self._aper_params.keys()}

    def reset(self):
        """Reset model to its initial values."""
        for name, value in self._aper_params.items():
            setattr(self, name, value)

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
        locations, self.all_limits, self.profile = find_apertures(
            self.ext, **self.aper_params)
        self.locations = list(locations)

        for listener in self.listeners:
            for i, (loc, limits) in enumerate(
                    zip(self.locations, self.all_limits), start=1):
                listener.handle_aperture(i, loc, limits[0], limits[1])

    def add_aperture(self, location, start, end):
        aperture_id = len(self.locations)+1
        self.locations.append(location)
        self.all_limits.append((start, end))

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
        self.all_limits[aperture_id-1] = (start, end)
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
        del self.locations[aperture_id-1]
        del self.all_limits[aperture_id-1]

        for listener in self.listeners:
            listener.delete_aperture(aperture_id)

    def clear_apertures(self):
        """Remove all apertures, calling delete on the listeners for each."""
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

    def clear_and_recalc(self):
        """Clear apertures and recalculate a new set."""
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

        def _maxaper_handler(new):
            self.model.max_apertures = int(new) if new is not None else None

        maxaper_input = TextInputLine("Max Apertures (empty means no limit)",
                                      self.model, attr="max_apertures",
                                      handler=_maxaper_handler)

        percentile_slider = TextSlider("Percentile (use mean if no value)",
                                       self.model, attr="percentile")

        minsky_input = TextInputLine("Min sky region", self.model,
                                     attr="min_sky_region")

        def _use_snr_handler(new):
            self.model.use_snr = 0 in new

        use_snr_widget = CheckboxLine("Use S/N ratio ?", self.model,
                                      attr="use_snr", handler=_use_snr_handler)

        threshold_slider = TextSlider("Threshold", self.model,
                                      attr="threshold")

        sizing_widget = SelectLine("Sizing method", self.model,
                                   attr="sizing_method")

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

        self.fig.step(x=range(self.model.profile.shape[0]),
                      y=self.model.profile,
                      color="black", mode="center")

        add_button = Button(label="Add Aperture")
        add_button.on_click(self.add_aperture)

        def _reset_handler():
            self.model.reset()
            for widget in (maxaper_input, minsky_input, use_snr_widget,
                           threshold_slider, percentile_slider, sizing_widget):
                widget.reset()
            self.clear_and_recalc()

        reset_button = Button(label="Reset", default_size=200)
        reset_button.on_click(_reset_handler)

        find_button = Button(label="Find apertures", button_type='success',
                             default_size=200)
        find_button.on_click(self.clear_and_recalc)

        helptext = Div()
        controls = column(children=[
            current_file,
            maxaper_input.build(),
            percentile_slider.build(start=0, end=100, step=1),
            minsky_input.build(),
            use_snr_widget.build(),
            threshold_slider.build(start=0, end=1, step=0.01),
            sizing_widget.build(),
            row([reset_button, find_button]),
            aperture_view.controls,
            add_button,
            self.submit_button,
            helptext,
        ])

        self.details = Div(text="")
        self.model.recalc_apertures()
        self.update_details()

        col = column(self.fig, self.details)
        col.sizing_mode = 'scale_width'
        layout = row(controls, col)

        Controller(self.fig, self.model, None, helptext)

        doc.add_root(layout)

    def handle_aperture(self, aperture_id, location, start, end):
        self.update_details()

    def delete_aperture(self, aperture_id):
        self.update_details()

    def update_details(self):
        """Update the details text area with the latest aperture data."""
        text = ""
        for i, (loc, limits) in enumerate(
                zip(self.model.locations, self.model.all_limits), start=1):
            text += (f"Aperture #{i}\t <b>Location:</b> {loc:.2f}"
                     f" <b>Lower Limit:</b> {limits[0]:.2f}"
                     f" <b>Upper Limit:</b> {limits[1]:.2f}<br/>")
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
