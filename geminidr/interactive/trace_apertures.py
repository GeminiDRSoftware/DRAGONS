from bokeh.layouts import row, column
from bokeh.models import Tabs, Panel, Button, Div

from geminidr.interactive import interactive, server
from geminidr.interactive.chebyshev1d import ChebyshevModel
from geminidr.interactive.interactive import GIMaskedSigmadCoords, GIFigure, GIMaskedSigmadScatter, GILine, GISlider


class TraceApertureInfo:
    def __init__(self, aperture, location, ref_coords, in_coords):
        self.aperture = aperture
        self.number = aperture['number']
        self.aper_upper = aperture['aper_upper']
        self.aper_lower = aperture['aper_lower']
        self.location = location
        self.ref_coords = ref_coords
        self.in_coords = in_coords

        # outputs
        self.model_dict = None
        self.m_final = None


class TraceApertureList:
    def __init__(self):
        self.apertures = list()

    def add_aperture(self, ap_info):
        self.apertures.append(ap_info)


class TraceApertureUI:
    def __init__(self, ap_info, model, min_order, max_order):
        self.model = model

        p = GIFigure(plot_width=600, plot_height=500,
                     title='Interactive Chebyshev',
                     x_axis_label='X', y_axis_label='Y',
                     tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap")

        self.scatter = GIMaskedSigmadScatter(p, model.coords)

        line = GILine(p)
        model.add_coord_listener(line.update_coords)

        model.recalc_chebyshev()

        order_slider = GISlider("Order", model.order, 1, min_order, max_order,
                                model, "order", model.recalc_chebyshev)
        sigma_slider = GISlider("Sigma", model.sigma, 0.1, 2, 10,
                                model, "sigma", model.recalc_chebyshev)

        mask_button = Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        info = Div(text="Aperture: %s<br/>\nLocation: %s<br/>\nUpper: %s<br/>Lower: %s<br/>"
                        % (ap_info.number, ap_info.location, ap_info.aper_upper, ap_info.aper_lower))

        controls = column(order_slider.component, sigma_slider.component, mask_button, unmask_button, info)

        self.component = row(controls, p.figure)

    def mask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.addmask(indices)

    def unmask_button_handler(self, stuff):
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.unmask(indices)


class TraceApertureVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, api_models, min_order, max_order):
        super().__init__()

        self.api_models = api_models
        self.min_order = min_order
        self.max_order = max_order

    def visualize(self, doc):
        super().visualize(doc)

        tabs = Tabs(tabs=[], name="tabs")

        for ap_info, model in self.api_models:
            tui = TraceApertureUI(ap_info, model, self.min_order, self.max_order)
            tab = Panel(child=tui.component, title="Aperture %s" % ap_info.number)

            tabs.tabs.append(tab)

        layout = column(tabs, self.submit_button)

        doc.add_root(layout)


def interactive_trace_apertures(ap_list, ext, order, dispaxis, sigma_clip,
                                spectral_coords, min_order, max_order):
    api_models = list()
    for ap_info in ap_list.apertures:
        masked_coords = GIMaskedSigmadCoords(ap_info.in_coords[1 - dispaxis], ap_info.in_coords[dispaxis])
        model = ChebyshevModel(order, ap_info.location, dispaxis, sigma_clip,
                               masked_coords, spectral_coords, ext)
        api_models.append((ap_info, model))

    server.set_visualizer(TraceApertureVisualizer(api_models, min_order, max_order))

    server.start_server()

    server.set_visualizer(None)

    for (ap_info, model) in api_models:
        ap_info.model_dict = model.model_dict
        ap_info.m_final = model.m_final
