import numpy as np
from bokeh.layouts import row, column
from bokeh.models import Column, Div, Button

from geminidr.interactive import server, interactive
from geminidr.interactive.interactive import GILine, GISlider, GIApertureModel, GIApertureView, build_figure
from gempy.library import astromodels, tracing
from geminidr.gemini.lookups import DQ_definitions as DQ


__all__ = ["interactive_find_source_apertures", ]


class FindSourceAperturesModel:
    def __init__(self, ext, profile, prof_mask, threshold, sizing_method, max_apertures):

        self.ext = ext
        self.profile = profile
        self.prof_mask = prof_mask
        self.threshold = threshold
        self.sizing_method = sizing_method
        self.max_apertures = max_apertures

        self.locations = None
        self.all_limits = None

        self.listeners = list()

    def add_listener(self, l):
        # l should be fn(locations, all_limits)
        self.listeners.append(l)

    def recalc_apertures(self):
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
        for l in self.listeners:
            l(self.locations, self.all_limits)

    def delete_aperture(self, aperture_id):
        del self.locations[aperture_id-1]
        del self.all_limits[aperture_id-1]


class FindSourceAperturesVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, model):
        super().__init__()
        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.model = model

        self.aperture_model = None
        self.details = None
        self.fig = None

    def clear_and_recalc(self):
        self.aperture_model.clear_apertures()
        self.model.recalc_apertures()

    def add_aperture(self):
        x = (self.fig.x_range.start + self.fig.figure.x_range.end) / 2
        self.aperture_model.add_aperture(x, x)
        self.update_details()

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
                                        self.model, "max_apertures", self.clear_and_recalc,
                                        throttled=True)
        threshold_slider = GISlider("Threshold", self.model.threshold, 1, 0, 1,
                                    self.model, "threshold", self.clear_and_recalc,
                                    throttled=True)

        # Create a blank figure with labels
        self.fig = build_figure(plot_width=600, plot_height=500,
                                title='Source Apertures',
                                tools="pan,wheel_zoom,box_zoom,reset",
                                x_range=(0, self.model.profile.shape[0]))

        self.aperture_model = GIApertureModel()
        aperture_view = GIApertureView(self.aperture_model, self.fig)
        self.aperture_model.add_listener(self)

        def apl(locations, all_limits):
            self.aperture_model.clear_apertures()
            for loc, limits in zip(locations, all_limits):
                self.aperture_model.add_aperture(limits[0], limits[1])
        self.model.add_listener(apl)
        # self.model.recalc_apertures()

        line = GILine(self.fig, range(self.model.profile.shape[0]), self.model.profile, color="black")

        add_button = Button(label="Add Aperture")
        add_button.on_click(self.add_aperture)

        controls = Column(max_apertures_slider.component, threshold_slider.component,
                          aperture_view.controls, add_button, self.submit_button)

        self.details = Div(text="")
        self.model.recalc_apertures()
        self.update_details()

        col = column(self.fig, self.details)
        layout = row(controls, col)

        doc.add_root(layout)

    def handle_aperture(self, aperture_id, start, end):
        location = (start+end)/2
        if aperture_id == len(self.model.locations)+1:
            self.model.locations = np.append(self.model.locations, location)
            self.model.all_limits.append( (start, end) )
        else:
            self.model.locations[aperture_id-1] = location
            self.model.all_limits[aperture_id-1] = (start, end)
        self.update_details()

    def delete_aperture(self, aperture_id):
        # ruh-roh raggy
        self.model.locations = np.delete(self.model.locations, aperture_id-1)
        del self.model.all_limits[aperture_id-1]
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
        return self.model.locations, self.model.all_limits


def interactive_find_source_apertures(ext, profile, prof_mask, threshold, sizing_method, max_apertures):
    if max_apertures is None:
        max_apertures = 50
    model = FindSourceAperturesModel(ext, profile, prof_mask, threshold, sizing_method,
                                     max_apertures)
    fsav = FindSourceAperturesVisualizer(model)
    server.set_visualizer(fsav)

    server.start_server()

    return fsav.result()
