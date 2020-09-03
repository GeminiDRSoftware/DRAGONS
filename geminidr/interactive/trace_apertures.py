import numpy as np
from astropy.modeling import models, fitting

from bokeh.layouts import row, column
from bokeh.models import Tabs, Panel, Button, Div, CustomJS

from geminidr.interactive import interactive, server
from geminidr.interactive.chebyshev1d import ChebyshevModel
from geminidr.interactive.interactive import GIMaskedSigmadCoords, GIFigure, GIMaskedSigmadScatter, GILine, GISlider, \
    GIDifferencingModel
from gempy.library import tracing
from gempy.library.astrotools import boxcar


class TraceApertureInfo:
    """
    Convenience class to hold a bunch of information for an aperture
    """
    def __init__(self, aperture, location, ref_coords, in_coords, m_init, fit_it):
        self.aperture = aperture
        self.number = aperture['number']
        self.aper_upper = aperture['aper_upper']
        self.aper_lower = aperture['aper_lower']
        self.location = location
        self.ref_coords = ref_coords
        self.in_coords = in_coords
        self.m_init = m_init
        self.fit_it = fit_it

        # outputs
        self.model_dict = None
        self.m_final = None


class TraceApertureList:
    """
    I thought I was going to have some clever functionality in
    here instead of a glorified list().
    """
    def __init__(self):
        self.apertures = list()

    def add_aperture(self, ap_info):
        self.apertures.append(ap_info)


class TraceApertureUI:
    def __init__(self, ap_info, model, min_order, max_order):
        # This holds all the bits that go into a tab for an aperture

        self.model = model

        p = GIFigure(plot_width=600, plot_height=400,
                     title='Chebyshev Fit',
                     x_axis_label='X', y_axis_label='Y',
                     tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap")

        self.scatter = GIMaskedSigmadScatter(p, model.coords)

        line = GILine(p)
        model.add_coord_listener(line.update_coords)

        model.recalc_chebyshev()

        p2 = GIFigure(plot_width=600, plot_height=200,
                      title='Fit Differential',
                      x_axis_label='X', y_axis_label='Y')
        self.line2 = GILine(p2)
        differencing_model = GIDifferencingModel(model.coords, model, model.model_calculate)
        differencing_model.add_coord_listener(self.line2.update_coords)

        order_slider = GISlider("Order", model.m_init.degree, 1, min_order, max_order,
                                model.m_init, "degree", model.recalc_chebyshev)
        sigma_slider = GISlider("Sigma", model.sigma, 0.1, 2, 10,
                                model, "sigma", model.recalc_chebyshev)

        mask_button = Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        info = Div(text="<table><tr><th align='left'>Aperture:</th><td>%s</td></tr><tr><th align='left'>Location:</th>"
                        "<td>%s</td></tr><tr><th align='left'>Upper:</th><td>%s</td></tr>"
                        "<tr><th align='left'>Lower:</th><td>%s</td></tr></table>"
                        % (ap_info.number, ap_info.location, ap_info.aper_upper, ap_info.aper_lower))

        controls = column(order_slider.component, sigma_slider.component, mask_button, unmask_button, info)

        self.component = row(controls, column(p.figure, p2.figure))

    def mask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.addmask(indices)

    def unmask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.unmask(indices)


class TraceApertureVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, aptable, locations, ext, dispaxis, step, nsum, max_missed,
                 order, sigma_clip, max_shift, min_order, max_order, *, log=None):
        super().__init__(log=log)

        self.locations = locations
        self.ext = ext
        self.dispaxis = dispaxis
        self.step = step
        self.nsum = nsum
        self.max_missed = max_missed
        self.max_shift = max_shift

        all_ref_coords, all_in_coords, spectral_coords = calc_coords(locations, ext, dispaxis, step, nsum, max_missed,
                                                                     max_shift, viewer=None)
        self.ap_list = TraceApertureList()
        for aperture in aptable:
            location = aperture['c0']
            coords = np.array([list(c1) + list(c2)
                               for c1, c2 in zip(all_ref_coords.T, all_in_coords.T)
                               if c1[dispaxis] == location])
            values = np.array(sorted(coords, key=lambda c: c[1 - dispaxis])).T
            ref_coords, in_coords = values[:2], values[2:]
            m_init = models.Chebyshev1D(degree=order, c0=location,
                                        domain=[0, ext.shape[dispaxis] - 1])
            fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                       sigma_clip, sigma=3)
            self.ap_list.add_aperture(TraceApertureInfo(aperture, location, ref_coords, in_coords, m_init, fit_it))

        api_models = list()
        for ap_info in self.ap_list.apertures:
            # masked_coords is a UI helper that understands the coordinates, which are currently masked/excluded,
            # and which were sigmad by the fit.  These three states get communicated to the plotting
            masked_coords = GIMaskedSigmadCoords(ap_info.in_coords[1 - dispaxis], ap_info.in_coords[dispaxis])
            model = ChebyshevModel(order, ap_info.location, dispaxis, sigma_clip,
                                   masked_coords, spectral_coords, ext, ap_info.m_init, ap_info.fit_it)
            api_models.append((ap_info, model))

        self.api_models = api_models
        self.min_order = min_order
        self.max_order = max_order

        # GISLider is my own creation - this one starts with a value of 'self.step' and when you adjust
        # the slider, it changes the "step" attribute on the self object.  The label in the UI is "Step"
        # It gets added to pure bokeh calls using self.step_slider.component
        self.step_slider = GISlider("Step", self.step, 1, 1, 100, self, "step")
        self.nsum_slider = GISlider("Lines to Sum", self.nsum, 1, 1, 100, self, "nsum")
        self.max_missed_slider = GISlider("Max Steps Missed", self.max_missed, 1, 1, 100, self, "max_missed")
        self.max_shift_slider = GISlider("Max Shift per Pixel", self.max_shift, 0.001, 0.001, 1, self, "max_shift")

        # This is the button below the top-level parameters.
        # Right now, if you click it, it calls update_fits
        # which causes the coords to get regenerated in here
        # You would want to change the implementation to "set this
        # flag to 'we're not done yet' and then call to the
        # server.stop_server() to exit bokeh.  Then
        # your primitive code responds to the not done flag
        # and regenerates the coords and calls back into the
        # top level call
        self.button = Button(label="Update Fits")
        self.button.on_click(self.update_fits)

        # Pop up the modal any time this submit button is disabled
        self.make_modal(self.button, "<b>Recalculating Points</b><br/>This may take 20 seconds")

    def _do_update_fits(self):
        all_ref_coords, all_in_coords, spectral_coords = calc_coords(self.locations, self.ext, self.dispaxis, self.step,
                                                                     self.nsum, self.max_missed,
                                                                     self.max_shift, viewer=None)
        for (api_info, model) in self.api_models:
            coords = np.array([list(c1) + list(c2)
                               for c1, c2 in zip(all_ref_coords.T, all_in_coords.T)
                               if c1[self.dispaxis] == api_info.location])
            values = np.array(sorted(coords, key=lambda c: c[1 - self.dispaxis])).T
            ref_coords, in_coords = values[:2], values[2:]
            model.update_in_coords(in_coords)
            model.recalc_chebyshev()
        self.button.disabled = False

    def update_fits(self):
        # button disable doesn't actually work like this since all of this happens on a single pass of the event queue
        self.button.disabled = True
        # do_later is a utility call that adds an event on the bokeh
        # event queue to call self._do_update_fits.  That then allows
        # the self.button.disabled=True logic to execute and the bokeh
        # UI to respond to it.
        self.do_later(self._do_update_fits)

    def visualize(self, doc):
        super().visualize(doc)

        # These are the "outside the aperture tabs" controls
        # These ctrls are what you want to replace with your dynamic list.
        # Though for prototyping your vision maybe we should skip that.
        # I'm more interested in the looping behavior.
        ctrls = column(self.step_slider.component, self.nsum_slider.component, self.max_missed_slider.component,
                       self.max_shift_slider.component, self.button)

        tabs = Tabs(tabs=[], name="tabs")

        for ap_info, model in self.api_models:
            tui = TraceApertureUI(ap_info, model, self.min_order, self.max_order)
            tab = Panel(child=tui.component, title="Aperture %s" % ap_info.number)

            tabs.tabs.append(tab)

        layout = row(ctrls, column(tabs, self.submit_button))

        doc.add_root(layout)


def calc_coords(locations, ext, dispaxis, step, nsum, max_missed, max_shift, viewer):
    for i, loc in enumerate(locations):
        c0 = int(loc + 0.5)
        spectrum = ext.data[c0] if dispaxis == 1 else ext.data[:, c0]
        start = np.argmax(boxcar(spectrum, size=3))

        # The coordinates are always returned as (x-coords, y-coords)
        ref_coords, in_coords = tracing.trace_lines(ext, axis=dispaxis,
                                                    start=start, initial=[loc],
                                                    rwidth=None, cwidth=5, step=step,
                                                    nsum=nsum, max_missed=max_missed,
                                                    initial_tolerance=None,
                                                    max_shift=max_shift,
                                                    viewer=viewer)
        if i:
            all_ref_coords = np.concatenate((all_ref_coords, ref_coords), axis=1)
            all_in_coords = np.concatenate((all_in_coords, in_coords), axis=1)
        else:
            all_ref_coords = ref_coords
            all_in_coords = in_coords

    spectral_coords = np.arange(0, ext.shape[dispaxis], step)

    return all_ref_coords, all_in_coords, spectral_coords


def interactive_trace_apertures(locations, step, nsum, max_missed, max_shift, aptable, ext, order, dispaxis, sigma_clip,
                                min_order, max_order, *, log=None):
    """
    This is the blocking call.  When this starts, bokeh will start and open a UI window.  The start_server() call
    will block.

    When the user hits the "submit" button below the aperture tabs, or closes the browser tab, this method will
    continue.  Then it is pulling out the api_models that were set in the visualizer as a return.

    The previous stuff is still around in chebyshev1d.interactive_chebyshev - and for now this file reuses some
    of the constructs from there.
    """
    visualizer = TraceApertureVisualizer(aptable, locations, ext, dispaxis, step, nsum, max_missed, order, sigma_clip,
                                         max_shift, min_order, max_order, log=log)
    server.set_visualizer(visualizer)

    server.start_server()
    server.set_visualizer(None)

    api_models = visualizer.api_models

    for (ap_info, model) in api_models:
        ap_info.model_dict = model.model_dict
        ap_info.m_final = model.m_final

    return visualizer.ap_list

