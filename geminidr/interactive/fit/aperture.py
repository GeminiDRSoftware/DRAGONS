import math

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Div, Button, Title, Range, Range1d, Column, Label, CustomJS
from bokeh.plotting import figure
from holoviews.streams import Pipe, Stream

from geminidr.interactive import server, interactive
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIApertureModel, GIApertureView, build_text_slider, build_range_slider, \
    GIApertureSliders
from gempy.library import tracing
from geminidr.gemini.lookups import DQ_definitions as DQ

# Datashader sandbox
import numpy as np, datashader as ds, xarray as xr
from datashader import transfer_functions as tf
import holoviews as hv

hv.extension('bokeh')

renderer = hv.renderer('bokeh')


__all__ = ["interactive_find_source_apertures", ]


# I want a standalone aperture control panel to work with the bokeh/datashader
# streaming visualization of the model.  This is towards supporting a high number
# of apertures concurrently.
# TODO dynamically adjust the control panel when dealing with large aperture counts
# i.e. >10, move to a select box for aperture number and single slider, perhaps
# or activate single slider for aperture closest to mouse click, etc.
class ApertureControlPanel:
    def __init__(self, fig, model):
        self.fig = fig
        self.model = model
        self.controls = column(Div(text='Controls go here'))
        self._sliders = dict()
        self.model.add_listener(self)
        self.view_start = self.fig.x_range.start
        self.view_end = self.fig.x_range.end
        fig.y_range.on_change('start', lambda attr, old, new: self.update_viewport(self.fig.x_range.start, self.view_end))
        fig.y_range.on_change('end', lambda attr, old, new: self.update_viewport(self.view_start, self.fig.x_range.end))
        # feels like I need this to convince the aperture lines to update on zoom
        fig.y_range.js_on_change('end', CustomJS(args=dict(plot=fig),
                                                 code="plot.properties.renderers.change.emit()"))
        for idx, loclims in enumerate(zip(self.model.locations, self.model.all_limits)):
            loc = loclims[0]
            lims = loclims[1]
            self.handle_aperture(idx, loc, lims[0], lims[1])

    def handle_aperture(self, idx, loc, start, end):
        # TODO hack for now, just so I can stress test on 700 apertures
        if idx >= 10:
            return

        slider = self._sliders.get(idx, None)
        if slider:
            pass
        else:
            slider = GIApertureSliders(self.fig, self.model, idx, loc, start, end)
            self._sliders[idx] = slider
            self.controls.children.append(slider.component)

    def delete_aperture(self, idx):
        slider = self._sliders.get(idx, None)
        if slider:
            self.controls.children.remove(slider.component)
            del self._sliders[idx]

    def do_update(self, *args, **kwargs):
        pass

    def update_viewport(self, start, end):
        """
        Handle a change in the view.

        We will adjust the slider ranges and/or disable them.

        Parameters
        ----------
        start
        end
        """
        # Bokeh docs provide no indication of the datatype or orientation of the start/end
        # tuples, so I have left the doc blank for now
        self.view_start = start
        self.view_end = end
        for ap_slider in self._sliders.values():
            ap_slider.update_viewport(start, end)


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

        self.locations = []
        self.all_limits = np.reshape([], (-1, 2))

        self.recalc_listeners = list()

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
        return aperture_id, location, self.all_limits[aperture_id-1][0], self.all_limits[aperture_id-1][1]

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
            self.all_limits = np.reshape([], (-1, 2))
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
        self.locations = np.append(self.locations, location)
        self.all_limits = np.append(self.all_limits, (start, end))
        # TODO make this cleaner, having issues with new flow in hv refactor
        self.all_limits = np.reshape(self.all_limits, (-1, 2))
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
    def __init__(self, model, filename_info=''):
        """
        Create a view for finding apertures with the given
        :class:`geminidr.interactive.fit.aperture.FindSourceAperturesModel`

        Parameters
        ----------
        model : :class:`geminidr.interactive.fit.aperture.FindSourceAperturesModel`
            Model to use for tracking the input parameters and recalculating fresh sets as needed
        """
        super().__init__(title='Find Source Apertures', filename_info=filename_info)
        self.model = model

        self.aperture_control_panel = None
        self.details = None
        self.fig = None
        self.hvimage = None

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

    # def _make_data_array_for_datashader(self, aperture_model, x_max, y_max):
    #     MINI = 0.01
    #
    #     da1 = [0, ]
    #     da2 = [0, ]
    #     y = [0, y_max]
    #     x = [0, ]
    #     ranges = list()
    #     for i, loclim in enumerate(zip(aperture_model.locations, aperture_model.all_limits)):
    #         lim = loclim[1]
    #         ranges.append((lim[0], lim[1]))
    #     ranges.sort(key=lambda x: x[0])
    #     # print(ranges)
    #     for range in ranges:
    #         x.extend((range[0]-MINI, range[0], range[1], range[1] + MINI))
    #         da1.extend((0, 1, 1, 0))
    #         da2.extend((0, 1, 1, 0))
    #     x.append(x_max)
    #     da1.append(0)
    #     da2.append(0)
    #     return xr.DataArray(
    #         [da1, da2,],
    #         coords=[('y', y),
    #                 ('x', x)],
    #         name='Z')

    def _make_data_array_for_holoviews(self, aperture_model, x_max, y_max, as_data_array=True):
        MINI = 0.01

        # da1 = [0, ]
        # da2 = [0, ]
        y = [0, y_max]
        x = [0, ]
        datarr = [(0,0)]
        ranges = list()
        for i, loclim in enumerate(zip(aperture_model.locations, aperture_model.all_limits)):
            lim = loclim[1]
            ranges.append((lim[0], lim[1]))
        ranges.sort(key=lambda x: x[0])
        # print(ranges)
        for range in ranges:
            x.extend((range[0]-MINI, range[0], range[1], range[1] + MINI))
            datarr.append((0,0))
            datarr.append((1,1))
            datarr.append((1,1))
            datarr.append((0,0))
            # da1.extend((0, 1, 1, 0))
            # da2.extend((0, 1, 1, 0))
        x.append(x_max)
        # da1.append(0)
        # da2.append(0)
        datarr.append((0,0))
        if as_data_array:
            return xr.DataArray(
                datarr,  #[da1, da2,],
                coords=[('x', x), ('y', y)],
                # coords=[('y', y),
                #         ('x', x)],
                name='Z')
        else:
            return x, y, datarr

    # def _make_holoviews_image_quadmeshed(self, aperture_model, x_max, y_max):
    #     da = self._make_data_array_for_datashader(aperture_model, x_max, y_max)
    #     canvas = ds.Canvas()
    #     cmap = ['#d1efd1', '#ffffff']
    #     qm = canvas.quadmesh(da, x='x', y='y')
    #     sh = tf.shade(qm)
    #
    #     self.aperture_pipe = Pipe(data=[])
    #     self.image_dmap = hv.DynamicMap(hv.Image, streams=[self.aperture_pipe])
    #     self.image_dmap.opts(cmap=cmap)
    #
    #     self.aperture_pipe.send(sh)
    #
    #     return hv.render(self.image_dmap)

    # def _reload_apertures_image_quadmeshed(self):
    #     x_max = self.model.profile.shape[0]
    #     y_max = math.ceil(np.nanmax(self.model.profile) * 1.05)
    #
    #     da = self._make_data_array_for_datashader(self.model, x_max, y_max)
    #     canvas = ds.Canvas()
    #     qm = canvas.quadmesh(da, x='x', y='y')
    #     sh = tf.shade(qm)
    #
    #     self.aperture_pipe.send(sh)

    def _make_holoviews_quadmeshed(self, aperture_model, x_max, y_max):
        da = self._make_data_array_for_holoviews(aperture_model, x_max, y_max)
        cmap = ['#ffffff', '#d1efd1']

        # self.aperture_pipe = Pipe(data=[])
        # self.qm_dmap = hv.DynamicMap(hv.QuadMesh, streams=[self.aperture_pipe])
        xyz = Stream.define('XYZ', data=da)
        self.qm_dmap = hv.DynamicMap(hv.QuadMesh, streams=[xyz()])
        self.qm_dmap.opts(cmap=cmap)
        # self.aperture_pipe.send(da)

        return hv.render(self.qm_dmap)
        # self.qm_renderer = renderer.get_plot(self.qm_dmap, curdoc())
        # return self.qm_renderer

        # self.quad_mesh = hv.QuadMesh(da)
        # return hv.render(self.quad_mesh)

    def _reload_apertures_quadmeshed(self):
        x_max = self.model.profile.shape[0]
        y_max = math.ceil(np.nanmax(self.model.profile) * 1.05)

        da = self._make_data_array_for_holoviews(self.model, x_max, y_max)

        # self.aperture_pipe.send(da)
        self.qm_dmap.event(data=da)
        # self.qm_renderer.refresh()
        print("reloaded")

    def visualize(self, doc):
        """
        Build the visualization in bokeh in the given browser document.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            Bokeh provided document to add visual elements to
        """
        super().visualize(doc)
        self.details = Div(text="")

        self.model.recalc_apertures()
        # for i in range(1, 700, 1):
        #     loc = float(i)*3.0
        #     start = loc-1.0
        #     end = loc+1.0
        #     self.model.add_aperture(loc, start, end)

        max_apertures_slider = build_text_slider("Max Apertures", self.model.max_apertures, 1, 1, 20,
                                                 self.model, "max_apertures", self.clear_and_recalc,
                                                 throttled=True)
        threshold_slider = build_text_slider("Threshold", self.model.threshold, 0.01, 0, 1,
                                             self.model, "threshold", self.clear_and_recalc,
                                             throttled=True)

        ymax = np.nanmax(self.model.profile)*1.05
        self.hvimage = self._make_holoviews_quadmeshed(self.model, self.model.profile.shape[0], math.ceil(ymax))

        self.hvimage.width = 600
        self.hvimage.line(x=range(self.model.profile.shape[0]), y=self.model.profile, color="black")
        self.hvimage.y_range = Range1d(start=0, end=ymax, bounds=(0, None))

        self.hvimage.title = Title(text='Source Apertures')
        self.hvimage.plot_height = 500

        self.aperture_control_panel = ApertureControlPanel(self.hvimage, self.model)

        self.model.add_listener(self)

        add_button = Button(label="Add Aperture")
        add_button.on_click(self.add_aperture)

        helptext = Div()
        controls = column(children=[max_apertures_slider, threshold_slider,
                                    self.aperture_control_panel.controls,
                                    add_button, self.submit_button, helptext])

        self.update_details()

        col = column(self.hvimage, self.details)
        col.sizing_mode = 'scale_width'
        layout = row(controls, col)

        Controller(self.hvimage, self.model, None, helptext, showing_residuals=False)

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
        self._reload_apertures_quadmeshed()
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
        self._reload_apertures_quadmeshed()
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
