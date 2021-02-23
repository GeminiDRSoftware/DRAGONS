"""
Interactive function and helper functions used to trace apertures.
"""
import numpy as np
from astropy import table
from bokeh import models as bm
from bokeh.layouts import column, layout, row, Spacer
from bokeh.plotting import figure

from gempy.library import astromodels, astrotools as at, config, tracing

from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import (
    connect_figure_extras, GIRegionListener, GIRegionModel, RegionEditor)

from .. import server
from .fit1d import (
    Fit1DPanel, Fit1DVisualizer, FittingParametersUI, InteractiveModel1D)


__all__ = ["interactive_trace_apertures", ]

DETAILED_HELP = """
    <h1>Help for traceApertures</h1>
    
    <p> Traces the spectrum in 2D spectral images for each aperture center 
        stored in the APERTURE on each extension. </p>
        
    <p> The panel on the right is used to subsample the spectrum. </p>
    
    TODO: Finish the HELP. 
    """


# noinspection PyMissingConstructor
class TraceAperturesTab(Fit1DPanel):
    """
    Panel for visualizing a 1-D fit, perhaps in a tab

    Parameters
    ----------
    visualizer : :class:`~geminidr.interactive.fit.fit1d.Fit1DVisualizer`
        visualizer to associate with
    fitting_parameters : dict
        parameters for this fit
    domain : list of pixel coordinates
        Used for new fit_1D fitter
    x : :class:`~numpy.ndarray`
        X coordinate values
    y : :class:`~numpy.ndarray`
        Y coordinate values
    min_order : int
        minimum order in UI
    max_order : int
        maximum order in UI
    xlabel : str
        label for X axis
    ylabel : str
        label for Y axis
    plot_width : int
        width of plot area in pixels
    plot_height : int
        height of plot area in pixels
    plot_residuals : bool
        True if we want the lower plot showing the differential between the fit and the data
    """
    def __init__(self, visualizer, fitting_parameters, domain, x, y,
                 weights=None, min_order=1, max_order=10, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True,
                 enable_user_masking=True, enable_regions=True):

        # Just to get the doc later
        self.visualizer = visualizer

        self.info_div = bm.Div(align="center")

        # Make a listener to update the info panel with the RMS on a fit
        def update_info(info_div, f):
            info_div.update(text=f'RMS: <b>{f.rms:.4f}</b>')
        listeners = [lambda f: update_info(self.info_div, f), ]

        self.fitting_parameters = fitting_parameters
        self.fit = InteractiveModel1D(
            fitting_parameters, domain, x, y, weights, listeners=listeners)

        fit = self.fit
        self.fitting_parameters_ui = FittingParametersUI(
            visualizer, fit, self.fitting_parameters, min_order, max_order)

        controls_ls = list()

        controls_column = self.fitting_parameters_ui.get_bokeh_components()

        reset_button = bm.Button(align=('center', 'end'),
                                 button_type='warning',
                                 label="Reset",
                                 width_policy='min')

        def reset_dialog_handler(result):
            if result:
                self.fitting_parameters_ui.reset_ui()
        self.reset_dialog = self.visualizer.make_ok_cancel_dialog(reset_button,
                                                                  'Reset will change all inputs for this tab back '
                                                                  'to their original values.  Proceed?',
                                                                  reset_dialog_handler)

        reset_button_row = row(
            Spacer(width_policy="max"),
            reset_button)

        controller_div = bm.Div(name="help_text",
                                default_size=100,
                                margin=(20, 0, 0, 0),
                                style={"color": "gray"},
                                width_policy="fit")

        controls_ls.extend(controls_column)
        controls_ls.append(reset_button_row)
        controls_ls.append(controller_div)
        controls = column(*controls_ls)

        self.controller_div = controller_div

        # Now the figures
        x_range = None
        y_range = None
        try:
            if self.fit.data and 'x' in self.fit.data.data and len(self.fit.data.data['x']) >= 2:
                x_min = min(self.fit.data.data['x'])
                x_max = max(self.fit.data.data['x'])
                x_pad = (x_max-x_min)*0.1
                x_range = bm.Range1d(x_min-x_pad, x_max+x_pad*2)
            if self.fit.data and 'y' in self.fit.data.data and len(self.fit.data.data['y']) >= 2:
                y_min = min(self.fit.data.data['y'])
                y_max = max(self.fit.data.data['y'])
                y_pad = (y_max-y_min)*0.1
                y_range = bm.Range1d(y_min-y_pad, y_max+y_pad)
        except:
            pass  # ok, we don't *need* ranges...
        if enable_user_masking:
            tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap"
        else:
            tools = "pan,wheel_zoom,box_zoom,reset"
        p_main = figure(plot_width=plot_width, plot_height=plot_height,
                        min_width=400, title='Fit', x_axis_label=xlabel,
                        y_axis_label=ylabel, tools=tools,
                        output_backend="webgl", x_range=x_range,
                        y_range=y_range)

        p_main.height_policy = 'fixed'
        p_main.width_policy = 'fit'

        if enable_regions:
            class Fit1DRegionListener(GIRegionListener):
                """
                Wrapper class so we can just detect when a bands are finished.

                We don't want to do an expensive recalc as a user is dragging
                a band around.
                """
                def __init__(self, fn):
                    """
                    Create a band listener that just updates on `finished`
                    Parameters
                    ----------
                    fn : function
                        function to call when band is finished.
                    """
                    self.fn = fn

                def adjust_region(self, region_id, start, stop):
                    pass

                def delete_region(self, region_id):
                    self.fn()

                def finish_regions(self):
                    self.fn()

            self.band_model = GIRegionModel()

            def update_regions():
                self.fit.model.regions = self.band_model.build_regions()
            self.band_model.add_listener(Fit1DRegionListener(update_regions))
            self.band_model.add_listener(Fit1DRegionListener(self.band_model_handler))

            connect_figure_extras(p_main, None, self.band_model)

            mask_handlers = (self.mask_button_handler,
                             self.unmask_button_handler)
        else:
            self.band_model = None
            mask_handlers = None

        self.controller = Controller(p_main, None, self.band_model, controller_div, mask_handlers=mask_handlers)
        fig_column = [p_main]

        if plot_residuals:
            # x_range is linked to the main plot so that zooming tracks between them
            p_resid = figure(plot_width=plot_width, plot_height=plot_height // 2,
                             min_width=400,
                             title='Fit Residuals',
                             x_axis_label=xlabel, y_axis_label='Delta',
                             tools="pan,box_zoom,reset",
                             output_backend="webgl", x_range=p_main.x_range, y_range=None)
            p_resid.height_policy = 'fixed'
            p_resid.width_policy = 'fit'
            connect_figure_extras(p_resid, None, self.band_model)
            fig_column.append(p_resid)
            # Initalizing this will cause the residuals to be calculated
            self.fit.data.data['residuals'] = np.zeros_like(self.fit.x)
            p_resid.scatter(x='x', y='residuals', source=self.fit.data,
                            size=5, legend_field='mask', **self.fit.mask_rendering_kwargs())

        # Initializing regions here ensures the listeners are notified of the region(s)
        if "regions" in fitting_parameters and fitting_parameters["regions"] is not None:
            region_tuples = at.cartesian_regions_to_slices(fitting_parameters["regions"])
            self.band_model.load_from_tuples(region_tuples)

        self.scatter = p_main.scatter(x='x', y='y', source=self.fit.data,
                                      size=5, legend_field='mask', **self.fit.mask_rendering_kwargs())
        self.fit.add_listener(self.model_change_handler)

        # TODO refactor? this is dupe from band_model_handler
        # hacking it in here so I can account for the initial
        # state of the band model (which used to be always empty)
        x_data = self.fit.data.data['x']
        for i in np.arange(len(x_data)):
            if not self.band_model or self.band_model.contains(x_data[i]):
                self.fit.band_mask[i] = 0
            else:
                self.fit.band_mask[i] = 1

        self.fit.perform_fit()
        self.line = p_main.line(x='xlinspace', y='model', source=self.fit.evaluation, line_width=3, color='black')

        info_row_spacer = Spacer(width_policy="fit",
                                 sizing_mode="stretch_width",
                                 min_width=10,
                                 max_width=800)

        info_row_ls = [info_row_spacer, self.info_div]

        if self.band_model:
            region_editor = RegionEditor(self.band_model)

            region_editor_wgt = region_editor.get_widget()
            region_editor_wgt.max_width = 500
            region_editor_wgt.sizing_mode = "stretch_width"
            region_editor_wgt.width_policy = "max"

            # Replace empty space by region editor
            info_row_ls.insert(0, region_editor_wgt)

        info_row = row(*info_row_ls,
                       min_width=100,
                       sizing_mode="stretch_width",
                       max_width=1000,
                       width_policy="max")

        fig_column.append(info_row)

        col = column(*fig_column)
        col.sizing_mode = 'scale_width'
        self.component = row(controls, col, spacing=10)


class TraceAperturesVisualizer(Fit1DVisualizer):
    """
    Custom visualizer for traceApertures().
    """
    def __init__(self, data_source, fitting_parameters, config,
                 reinit_params=None, reinit_extras=None,
                 modal_message=None,
                 modal_button_label=None,
                 order_param="order",
                 tab_name_fmt='{}',
                 xlabel='x', ylabel='y',
                 domains=None, function=None, title=None, primitive_name=None, filename_info=None,
                 template="fit1d.html",
                 template_variables=None,
                 **kwargs):

        super(Fit1DVisualizer, self).__init__(config=config,
                                              filename_info=filename_info,
                                              primitive_name=primitive_name,
                                              template=template,
                                              title=title)
        self.layout = None
        self.widgets = {}

        function = 'chebyshev'
        self.function = bm.Div(
            text=f'<p> Function: <b>{function.capitalize()}</b> </p>'
                 f'<p style="color: gray"> These are parameters used to '
                 f'(re)generate the input data for fitting. </p>',
            sizing_mode="fixed",
            width=212,  # ToDo: Hardcoded width. Would there be a better solution?
            )

        if reinit_params is not None or reinit_extras is not None:
            # Create left panel
            reinit_widgets = self.make_widgets_from_config(reinit_params, reinit_extras, modal_message is None)

            # This should really go in the parent class, like submit_button
            if modal_message:
                self.reinit_button = bm.Button(label=modal_button_label if modal_button_label else "Reconstruct points")
                self.reinit_button.on_click(self.reconstruct_points)
                self.make_modal(self.reinit_button, modal_message)
                reinit_widgets.append(self.reinit_button)

            self.reinit_panel = column(self.function, *reinit_widgets)
        else:
            # left panel with just the function selector (Chebyshev, etc.)
            self.reinit_panel = column(self.function)

        # Grab input coordinates or calculate if we were given a callable
        # TODO revisit the raging debate on `callable` for Python 3
        if callable(data_source):
            self.reconstruct_points_fn = data_source
            data = data_source(config, self.extras)
            # For this, we need to remap from
            # [[x1, y1, weights1], [x2, y2, weights2], ...]
            # to allx=[x1,x2..] ally=[y1,y2..] all_weights=[weights1,weights2..]
            allx = list()
            ally = list()
            all_weights = list()
            for dat in data:
                allx.append(dat[0])
                ally.append(dat[1])
                if len(dat) >= 3:
                    all_weights.append(dat[2])
            if len(all_weights) == 0:
                all_weights = None
        else:
            self.reconstruct_points_fn = None
            if reinit_params:
                raise ValueError("Saw reinit_params but data_source is not a callable")
            if reinit_extras:
                raise ValueError("Saw reinit_extras but data_source is not a callable")
            allx = data_source[0]
            ally = data_source[1]
            if len(data_source) >= 3:
                all_weights = data_source[2]
            else:
                all_weights = None

        # Some sanity checks now
        if isinstance(fitting_parameters, list):
            if not(len(fitting_parameters) == len(allx) == len(ally)):
                raise ValueError("Different numbers of models and coordinates")
            self.nfits = len(fitting_parameters)
        else:
            if allx.size != ally.size:
                raise ValueError("Different (x, y) array sizes")
            self.nfits = 1

        self.reinit_extras = [] if reinit_extras is None else reinit_extras

        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})
        if order_param and order_param in self.config._fields:
            field = self.config._fields[order_param]
            if hasattr(field, 'min') and field.min:
                kwargs['min_order'] = field.min
            else:
                kwargs['min_order'] = 1
            if hasattr(field, 'max') and field.max:
                kwargs['max_order'] = field.max
            else:
                kwargs['max_order'] = field.default * 2
        else:
            kwargs['min_order'] = 1
            kwargs['max_order'] = 10

        self.tabs = bm.Tabs(tabs=[], name="tabs")
        self.tabs.sizing_mode = 'scale_width'
        self.fits = []
        if self.nfits > 1:
            if domains is None:
                domains = [None] * len(fitting_parameters)
            if all_weights is None:
                all_weights = [None] * len(fitting_parameters)
            for i, (fitting_parms, domain, x, y, weights) in \
                    enumerate(zip(fitting_parameters, domains, allx, ally, all_weights), start=1):
                tui = Fit1DPanel(self, fitting_parms, domain, x, y, weights, **kwargs)
                tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(i))
                self.tabs.tabs.append(tab)
                self.fits.append(tui.fit)
        else:

            # ToDo: Review if there is a better way of handling this.
            if all_weights is None:
                all_weights = [None]

            # ToDo: the domains variable contains a list. I changed it to
            #  domains[0] and the code worked.
            tui = Fit1DPanel(self, fitting_parameters[0], domains[0], allx[0], ally[0], all_weights[0], **kwargs)
            tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(1))
            self.tabs.tabs.append(tab)
            self.fits.append(tui.fit)

    def visualize(self, doc):
        """
        Start the bokeh document using this visualizer. This is a customized
        version of Fit1DVisualizer.visualize() dedicated to traceApertures().

        This call is responsible for filling in the bokeh document with
        the user interface.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            bokeh document to draw the UI in
        """
        super(Fit1DVisualizer, self).visualize(doc)

        # Edit left/central column
        filename = bm.Div(
            css_classes=["filename"],
            id="_filename",
            margin=(0, 0, 0, 76),
            name="filename",
            height_policy="max",
            sizing_mode="fixed",
            text=f"Current filename: {self.filename_info}",
            style={
                "background": "white",
                "color": "#666666",
                "padding": "10px 0px 10px 0px",
                "vertical-align": "middle",
            },
            width=500,
            width_policy="fixed",
        )

        left_col = column(filename, self.tabs,
                          sizing_mode="stretch_width",
                          spacing=5)

        # Edit right column
        self.submit_button.align = ("end", "center")
        self.submit_button.width_policy = "min"

        self.reinit_panel.css_classes = ["data_source"]
        self.reinit_panel.sizing_mode = "fixed"

        right_col = column(self.submit_button, self.reinit_panel,
                           sizing_mode="fixed",
                           spacing=5)

        # Put all together
        all_content = row(left_col, right_col,
                          spacing=15,
                          sizing_mode="stretch_both")

        doc.template_variables["primitive_long_help"] = DETAILED_HELP
        doc.add_root(all_content)


def interactive_trace_apertures(ext, _config, _fit1d_params):
    """
    Run traceApertures() interactively.

    Parameters
    ----------
    ext : AstroData
        Single extension extracted from an AstroData object.
    _config : dict
        Dictionary containing the parameters from traceApertures().
    _fit1d_params : dict
        Dictionary containing initial parameters for fitting a model.

    Returns
    -------
    """
    ap_table = ext.APERTURE
    fit_par_list = [_fit1d_params] * len(ap_table)
    domain_list = [[ap['domain_start'], ap['domain_end']] for ap in ap_table]

    # Create parameters to add to the UI
    reinit_extras = {
        "max_missed": config.RangeField(
            "Max Missed", int, 5, min=0),
        "max_shift": config.RangeField(
            "Max Shifted", float, 0.05, min=0.001, max=0.1),
        "nsum": config.RangeField(
            "Number of lines to sum", int, 10, min=1),
        "step": config.RangeField(
            "Tracing step: ", int, 10, min=1),
    }

    if (2 - ext.dispersion_axis()) == 1:
        xlabel = "x / columns [px]"
        ylabel = "y / rows [px]"
    else:
        xlabel = "y / rows [px]"
        ylabel = "x / columns [px]"

    def data_provider(conf, extra):
        return trace_apertures_data_provider(ext, conf, extra)

    visualizer = TraceAperturesVisualizer(
        data_provider,
        config=_config,
        filename_info=ext.filename,
        fitting_parameters=fit_par_list,
        tab_name_fmt="Aperture {}",
        xlabel=xlabel,
        ylabel=ylabel,
        reinit_extras=reinit_extras,
        domains=domain_list,
        primitive_name="traceApertures()",
        template="trace_apertures.html",
        title="Trace Apertures")

    server.interactive_fitter(visualizer)
    models = visualizer.results()

    all_aperture_tables = []
    dispaxis = 2 - ext.dispersion_axis()  # python sense

    for final_model, ap in zip(models, ap_table):
        location = ap['c0']
        this_aptable = astromodels.model_to_table(final_model.model)

        # Recalculate aperture limits after rectification
        apcoords = final_model.evaluate(np.arange(ext.shape[dispaxis]))

        this_aptable["aper_lower"] = (
                ap["aper_lower"] + (location - apcoords.min()))

        this_aptable["aper_upper"] = (
                ap["aper_upper"] - (apcoords.max() - location))

        all_aperture_tables.append(this_aptable)

    new_aptable = table.vstack(all_aperture_tables, metadata_conflicts="silent")
    return new_aptable


def trace_apertures_data_provider(ext, conf, extra):
    """
    Function used by the interactive fitter to generate the a list with
    pairs of [x, y] data containing the knots used for tracing.

    Parameters
    ----------
    ext : AstroData
        Single extension of data containing an .APERTURE table.
    conf : dict
        Dictionary containing default traceApertures() parameters.
    extra : dict
        Dictionary containing extra parameters used to re-create the input data
        for fitting interactively.

    Returns
    -------
    list : pairs of (x, y) for each aperture center, where x is the
    spectral position of the knots, and y is the spacial position of the
    knots.
    """
    all_tracing_knots = []
    dispaxis = 2 - ext.dispersion_axis()  # python sense

    for _i, _loc in enumerate(ext.APERTURE['c0'].data):
        _c0 = int(_loc + 0.5)

        _spectrum = (ext.data[_c0] if dispaxis == 1
                     else ext.data[:, _c0])

        _start = np.argmax(at.boxcar(_spectrum, size=3))

        _, _in_coords = tracing.trace_lines(
            ext, axis=dispaxis, cwidth=5,
            initial=[_loc], initial_tolerance=None,
            max_missed=extra['max_missed'], max_shift=extra['max_shift'],
            nsum=extra['nsum'], rwidth=None, start=_start,
            step=extra['step'])

        _in_coords = np.ma.masked_array(_in_coords)

        # ToDo: This should not be required
        _in_coords.mask = np.zeros_like(_in_coords)

        spectral_tracing_knots = _in_coords[1 - dispaxis]
        spatial_tracing_knots = _in_coords[dispaxis]

        all_tracing_knots.append(
            [spectral_tracing_knots, spatial_tracing_knots])

    return all_tracing_knots
