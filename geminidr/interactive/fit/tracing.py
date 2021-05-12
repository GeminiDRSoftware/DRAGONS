"""
Interactive function and helper functions used to trace apertures.
"""
from copy import deepcopy

import numpy as np
from astropy import table
from bokeh import events
from bokeh import models as bm
from bokeh.layouts import column, row, Spacer
from bokeh.plotting import figure

from geminidr.interactive.controls import Controller
from geminidr.interactive.fit import help
from geminidr.interactive.interactive import (
    connect_figure_extras, GIRegionModel, RegionEditor)
from gempy.library import astromodels, astrotools as at, tracing
from gempy.library.config import RangeField
from .fit1d import (Fit1DPanel, Fit1DRegionListener, Fit1DVisualizer,
                    FittingParametersUI, InteractiveModel1D)
from .. import server

__all__ = ["interactive_trace_apertures", ]


# noinspection PyUnusedLocal,PyMissingConstructor
class FittingParametersForTracedDataUI(FittingParametersUI):
    """
    Represents the panel with the adjustable parameters for fitting the
    trace.
    """

    def build_column(self):
        """
        Override the standard column order.
        """
        column_title = bm.Div(
            text=f"Fit Function: <b>{self.vis.function_name.capitalize()}</b>",
            min_width=100,
            max_width=202,
            sizing_mode='stretch_width',
            style={"color": "black", "font-size": "115%", "margin-top": "5px"},
            width_policy='max',
        )

        rejection_title = bm.Div(
            text="Rejection Parameters",
            min_width=100,
            max_width=202,
            sizing_mode='stretch_width',
            style={"color": "black", "font-size": "115%", "margin-top": "10px"},
            width_policy='max',
        )

        column_list = [column_title, self.order_slider, rejection_title,
                       self.niter_slider, self.sigma_button,
                       self.sigma_lower_slider, self.sigma_upper_slider,
                       self.grow_slider]

        return column_list

    def sigma_button_handler(self, attr, old, new):
        """
        Handle the sigma clipping being turned on or off.

        This will also trigger a fit since the result may
        change.

        Parameters
        ----------
        attr : Any
            unused
        old : str
            old value of the toggle button
        new : str
            new value of the toggle button
        """
        self.fit.sigma_clip = bool(new)

        if self.fit.sigma_clip:
            self.fitting_parameters["sigma_upper"] = \
                self.sigma_upper_slider.children[0].value
            self.fitting_parameters["sigma_lower"] = \
                self.sigma_lower_slider.children[0].value
        else:
            self.fitting_parameters["sigma_upper"] = None
            self.fitting_parameters["sigma_lower"] = None
        self.fit.perform_fit()


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
    xlabel : str, optional
        label for X axis
    ylabel : str, optional
        label for Y axis
    plot_width : int, optional
        width of plot area in pixels
    plot_height : int, optional
        height of plot area in pixels
    weights : :class:`~numpy.ndarray`, optional
        Coordinates weights.
    """

    def __init__(self, visualizer, fitting_parameters, domain, x, y, idx=0,
                 weights=None, plot_height=600,
                 plot_width=600, plot_title="Trace Apertures - Fitting",
                 xlabel='x', ylabel='y'):

        # Just to get the doc later
        self.visualizer = visualizer
        self.index = idx
        self.rms_div = self.create_rms_div()

        listeners = [lambda f: self.update_info(self.rms_div, f), ]

        self.fitting_parameters = fitting_parameters

        self.fit = InteractiveModel1D(fitting_parameters, domain, x, y, weights,
                                      listeners=listeners)

        self.fitting_parameters_ui = FittingParametersForTracedDataUI(
            visualizer, self.fit, self.fitting_parameters)

        self.pars_column, self.controller_help = self.create_pars_column(
            fit_pars_ui=self.fitting_parameters_ui.get_bokeh_components(),
            rms_div=self.rms_div)

        self.plots_column, self.controller = self.create_plots_column(
            plot_height=plot_height, plot_title=plot_title,
            plot_width=plot_width, xlabel=xlabel, ylabel=ylabel)

        self.component = row(self.plots_column, self.pars_column,
                             css_classes=["tab-content"], spacing=5)

    def create_pars_column(self, fit_pars_ui, rms_div, column_width=220):
        """
        Creates the control panel on the left of the page where one can set
        what are the fit parameter values.
        """
        # Create the reset button, add its functionality and add it to the layout
        reset_button = bm.Button(align='start',
                                 button_type='danger',
                                 height=35,
                                 label="Reset",
                                 width=202)

        reset_dialog_message = ('Reset will change all inputs for this tab back'
                                ' to their original values. Proceed?')

        self.reset_dialog = self.visualizer.make_ok_cancel_dialog(
            reset_button, reset_dialog_message, self.reset_dialog_handler)

        controller_help = bm.Div(id=f"control_help_{self.index}",
                                 name=f"control_help_{self.index}",
                                 margin=(20, 0, 0, 0),
                                 max_width=column_width,
                                 width=column_width,
                                 style={
                                     "color": "gray",
                                     "padding": "5px",
                                 })

        controls_ls = fit_pars_ui
        controls_ls.append(reset_button)
        controls_ls.append(rms_div)
        controls_ls.append(controller_help)

        controls_col = column(*controls_ls,
                              css_classes=["column-fitpars"],
                              max_width=column_width,
                              margin=(0, 0, 0, 10),
                              width_policy="fit")

        return controls_col, controller_help

    def create_plots_column(self, plot_width, plot_height, plot_title, xlabel,
                            ylabel, enable_regions=True):
        """
        Creates the central plot area with the main plot, the residuals and
        a text field where the user can select regions.
        """
        # Now the figures
        x_range = None
        y_range = None

        if self.fit.data:

            if 'x' in self.fit.data.data:
                if len(self.fit.data.data['x']) >= 2:
                    x_min = min(self.fit.data.data['x'])
                    x_max = max(self.fit.data.data['x'])
                    x_pad = (x_max - x_min) * 0.1
                    x_range = bm.Range1d(x_min - x_pad, x_max + x_pad * 2)

            if 'y' in self.fit.data.data:
                if len(self.fit.data.data['y']) >= 2:
                    y_min = min(self.fit.data.data['y'])
                    y_max = max(self.fit.data.data['y'])
                    y_pad = (y_max - y_min) * 0.1
                    y_range = bm.Range1d(y_min - y_pad, y_max + y_pad)

        tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap"

        # Create main plot area ------------------------------------------------
        p_main = figure(css_classes=['plot-main'],
                        max_height=1000,
                        min_height=100,
                        min_width=400,
                        output_backend="webgl",
                        plot_width=plot_width,
                        plot_height=plot_height,
                        title=plot_title,
                        tools=tools,
                        x_axis_label=xlabel,
                        x_range=x_range,
                        y_axis_label=ylabel,
                        y_range=y_range)

        p_main.height_policy = 'fit'
        p_main.margin = (15, 0, 0, 15)
        p_main.width_policy = 'fit'

        # Enable region selection ----------------------------------------------
        if enable_regions:

            self.band_model = GIRegionModel()

            def update_regions():
                self.fit.model.regions = self.band_model.build_regions()

            # Handles Bands Regions
            self.band_model.add_listener(
                Fit1DRegionListener(update_regions))

            # Handles Bands Masks
            self.band_model.add_listener(
                Fit1DRegionListener(self.band_model_handler))

            connect_figure_extras(p_main, None, self.band_model)

            mask_handlers = (self.mask_button_handler,
                             self.unmask_button_handler)

        else:
            self.band_model = None
            mask_handlers = None

        _controller = Controller(p_main, None, self.band_model,
                                 self.controller_help,
                                 mask_handlers=mask_handlers)

        self.add_custom_cursor_behavior(p_main)

        # Create residual plot area --------------------------------------------
        # x_range is linked to the main plot so that zooming tracks between them
        p_resid = figure(min_width=400,
                         output_backend="webgl",
                         plot_height=plot_height // 2,
                         plot_width=plot_width,
                         title='Fit Residuals',
                         tools="pan,box_zoom,reset",
                         x_axis_label=xlabel,
                         x_range=p_main.x_range,
                         y_axis_label='Delta',
                         y_range=None)

        p_resid.height_policy = 'fixed'
        p_resid.margin = (0, 0, 0, 15)
        p_resid.width_policy = 'fit'

        connect_figure_extras(p_resid, None, self.band_model)

        # Initializing this will cause the residuals to be calculated
        self.fit.data.data['residuals'] = np.zeros_like(self.fit.x)

        p_resid.scatter(x='x', y='residuals', source=self.fit.data,
                        size=5, legend_field='mask', **self.fit.mask_rendering_kwargs())

        # Initializing regions here ensures the listeners are notified of the region(s)
        if "regions" in self.fitting_parameters and self.fitting_parameters["regions"] is not None:
            region_tuples = at.cartesian_regions_to_slices(self.fitting_parameters["regions"])
            self.band_model.load_from_tuples(region_tuples)

        self.scatter = p_main.scatter(x='x', y='y', source=self.fit.data,
                                      size=5, legend_field='mask', **self.fit.mask_rendering_kwargs())
        self.fit.add_listener(self.model_change_handler)

        # TODO refactor? this is dupe from band_model_handler
        #   hacking it in here so I can account for the initial
        #   state of the band model (which used to be always empty)
        x_data = self.fit.data.data['x']
        for i in np.arange(len(x_data)):
            if not self.band_model or self.band_model.contains(x_data[i]):
                self.fit.band_mask[i] = 0
            else:
                self.fit.band_mask[i] = 1

        self.fit.perform_fit()
        self.line = p_main.line(x='xlinspace', y='model',
                                source=self.fit.evaluation, line_width=3,
                                color='crimson')

        fig_column = [p_main, p_resid]

        if self.band_model:
            region_editor = RegionEditor(self.band_model)
            region_editor_wgt = region_editor.get_widget()
            region_editor_wgt.align = "start"
            region_editor_wgt.max_width = 835
            region_editor_wgt.margin = (0, 0, 15, 15)
            region_editor_wgt.sizing_mode = "stretch_width"
            region_editor_wgt.width_policy = "fit"
            fig_column.append(region_editor_wgt)

        col = column(*fig_column,
                     css_classes=["column-plot"],
                     height_policy='fit',
                     max_height=1000,
                     margin=(0, 10, 0, 0),
                     sizing_mode='scale_both',
                     width_policy='fit')

        return col, _controller

    @staticmethod
    def add_custom_cursor_behavior(p):
        """
        Customize cursor behavior depending on which tool is active.
        """
        pan_start = '''
            var mainPlot = document.getElementsByClassName('plot-main')[0];
            var active = [...mainPlot.getElementsByClassName('bk-active')];

            console.log(active);
            
            if ( active.some(e => e.title == "Pan") ) { 
                Bokeh.cursor = 'move'; }
        '''

        pan_end = '''
            var mainPlot = document.getElementsByClassName('plot-main')[0];
            var elm = mainPlot.getElementsByClassName('bk-canvas-events')[0];
            
            Bokeh.cursor = 'default';
            elm.style.cursor = Bokeh.cursor;
        '''

        mouse_move = """
            var mainPlot = document.getElementsByClassName('plot-main')[0];
            var elm = mainPlot.getElementsByClassName('bk-canvas-events')[0];
            elm.style.cursor = Bokeh.cursor;
        """

        p.js_on_event(events.MouseMove, bm.CustomJS(code=mouse_move))
        p.js_on_event(events.PanStart, bm.CustomJS(code=pan_start))
        p.js_on_event(events.PanEnd, bm.CustomJS(code=pan_end))

    def create_rms_div(self):
        """
        Creates a bm.Div placeholder to print out the RMS information.

        Returns
        -------
        bm.Div : element that displays the fitting RMS.
        """
        rms_div = bm.Div(align="start",
                         id=f"rms_div_{self.index}",
                         width=202,
                         margin=(15, 5, 5, 5),
                         max_width=202,
                         style={
                             "background-color": "white",
                             "border-top": "1px solid gainsboro",
                             "font-size": "115%",
                             "padding": "5px",
                             "width": "100%",
                         },
                         width_policy="fixed")

        return rms_div

    def reset_dialog_handler(self, result):
        """
        Reset fit parameter values.
        """
        if result:
            self.fitting_parameters_ui.reset_ui()

    @staticmethod
    def update_info(info_div, f):
        """
        Listener to update the info panel with the RMS of a fit.

        Parameters
        ----------
        info_div : bm.Div
            Div parameter containing the RMS information.
        f : ???
            ???
        """
        info_div.update(text=f'RMS: <b>{f.rms:.4f}</b>')

    def update_regions(self):
        """ Update fitting regions """
        self.fit.model.regions = self.band_model.build_regions()


class TraceAperturesVisualizer(Fit1DVisualizer):
    """
    Custom visualizer for traceApertures().
    """

    def __init__(self, data_source, fitting_parameters, config,
                 domains=None, filename_info=None, help_text=None,
                 modal_button_label="Trace apertures",
                 modal_message="Tracing apertures...",
                 primitive_name=None, reinit_extras=None, reinit_params=None,
                 tab_name_fmt='{}', template="fit1d.html", title=None,
                 xlabel='x', ylabel='y', **kwargs):

        super(Fit1DVisualizer, self).__init__(config=config,
                                              filename_info=filename_info,
                                              help_text=help_text,
                                              primitive_name=primitive_name,
                                              template=template,
                                              title=title)

        self.layout = None
        self.last_changed = None
        self.reinit_extras = {} if reinit_extras is None else reinit_extras
        self.reinit_params = [] if reinit_params is None else reinit_params
        self.widgets = {}
        self.error_alert = self.create_error_alert()

        # Save parameters in case we want to reset them
        self._reinit_extras = {} if reinit_extras is None \
            else {key: val.default for key, val in self.reinit_extras.items()}
        self._reinit_pars = {} if reinit_params is None \
            else {key: val for key, val in config.items() if key in reinit_params}

        self.function_name = 'chebyshev'
        self.function = self.create_function_div(
            text='Tracing Parameters')

        self.reinit_panel = self.create_tracing_panel(
            modal_button_label=modal_button_label,
            modal_message=modal_message,
            reinit_extras=reinit_extras,
            reinit_params=reinit_params)

        # Grab input coordinates or calculate if we were given a callable
        self.reconstruct_points_fn = self.data_source_factory(
            data_source, reinit_extras=reinit_extras, reinit_params=reinit_params)

        data = self.reconstruct_points_fn(config, self.extras)

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
            if all_weights is None or len(all_weights) == 0:
                all_weights = None

        # Some sanity checks now
        if isinstance(fitting_parameters, list):
            if not (len(fitting_parameters) == len(allx) == len(ally)):
                raise ValueError("Different numbers of models and coordinates")
            self.nfits = len(fitting_parameters)
        else:
            if len(allx) != len(ally):
                raise ValueError("Different (x, y) array sizes")
            self.nfits = 1

        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})

        self.tabs = bm.Tabs(css_classes=['tabs'],
                            height_policy="max",
                            name="tabs",
                            tabs=[])

        self.tabs.sizing_mode = 'scale_width'
        self.fits = []
        if self.nfits > 1:
            if domains is None:
                domains = [None] * len(fitting_parameters)
            if all_weights is None:
                all_weights = [None] * len(fitting_parameters)
            for i, (fitting_parms, domain, x, y, weights) in \
                    enumerate(zip(fitting_parameters, domains, allx, ally, all_weights), start=1):
                # tui = TraceAperturesTab(self, fitting_parms, domain, x, y, weights, index=i, **kwargs)
                tui = TraceAperturesTab(self, fitting_parms, domain, x, y, weights, **kwargs)
                tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(i))
                self.tabs.tabs.append(tab)
                self.fits.append(tui.fit)
        else:

            # ToDo: Review if there is a better way of handling this.
            if all_weights is None:
                all_weights = [None]

            # ToDo: the domains variable contains a list. I changed it to
            #  domains[0] and the code worked.
            tui = TraceAperturesTab(self, fitting_parameters[0], domains[0], allx[0], ally[0], all_weights[0], **kwargs)
            tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(1))
            self.tabs.tabs.append(tab)
            self.fits.append(tui.fit)

        self.add_callback_to_sliders()

    def add_callback_to_sliders(self):
        """
        Adds a callback function to record which slider was last updated.
        """
        for key, val in self.widgets.items():
            val.on_change('value', self.register_last_changed(key))

    @staticmethod
    def create_error_alert():
        """
        Creates a Pre element to hold a callback function that displays an
        Alert when an error occurs.
        """

        callback = bm.CustomJS(
            args={}, code="if (cb_obj.text !== '') { "
                          "    alert(cb_obj.text); "
                          "    cb_obj.text = '';"
                          "}")

        holder = bm.PreText(text='', css_classes=['hidden'], visible=False)
        holder.js_on_change('text', callback)
        return holder

    def data_source_factory(self, data_source, reinit_extras=None,
                            reinit_params=None):
        """
        If our input data_source is an array, wraps it inside a callable
        function which is called later by the `...` method.

        Parameters
        ----------
        data_source : array or function
            An array with 2 or more dimensions or a function that returns this
            array.
        reinit_extras : dict, optional
            Extra parameters to reinitialize the data.
        reinit_params : dict, optional
            Parameters used to reinitialize the data.

        Returns
        -------
        function
        """
        # TODO revisit the raging debate on `callable` for Python 3
        if callable(data_source):
            def _data_source(_config, _extras):
                """
                Wraps the callable so we can handle the error properly.
                """
                try:
                    data = data_source(_config, _extras)
                except IndexError:

                    self.error_alert.text = "Could not perform tracing with " \
                                            "current parameter. Rolling back " \
                                            "to previous working configuration."

                    self.reset_tracing_panel(param=self.last_changed)
                    _extras[self.last_changed] = \
                        self._reinit_extras[self.last_changed]
                    data = data_source(_config, _extras)
                else:
                    # Store successful pars
                    self._reinit_extras = deepcopy(_extras)

                return data

        else:
            def _data_source(_config, _extras):
                """Simply passes the input data forward."""
                if reinit_extras:
                    raise ValueError("Saw reinit_extras but "
                                     "data_source is not a callable")
                if reinit_params:
                    raise ValueError("Saw reinit_params but "
                                     "data_source is not a callable")
                if len(data_source) >= 3:
                    return data_source[0], data_source[1], data_source[2]
                else:
                    return data_source[0], data_source[1]

        return _data_source

    @staticmethod
    def create_function_div(text=""):
        """
        Creates a DIV element containing some small help text.

        Parameters
        ----------
        text : str
            Text displayed above function name.

        Returns
        -------
        bokeh.models.Div : help text.
        """
        div = bm.Div(css_classes=['panel-header'],
                     text=text,
                     id="function_div",
                     width=212,
                     width_policy="fixed",
                     style={"font-size": "large",
                            "font-weight": "bold",
                            "margin-top": "-10px"})

        return div

    def create_tracing_panel(self, modal_button_label="Reconstruct points",
                             modal_message=None, reinit_params=None,
                             reinit_extras=None):
        """
        Creates the Tracing (leftmost) Panel. This function had some code not
        used by TraceAperturesVisualizer, but this code helps to keep
        compatibility in case we want to merge it into Fit1DVisualizer.

        Parameters
        ----------
        modal_button_label : str
            Text on the button which performs tracing.
        modal_message : str
            Displays message as a Modal Div element while performing tracing on
            the background.
        reinit_params : list
            Parameters for re-tracing.
        reinit_extras : dict
            Extra parameters for re-tracing.
        """
        # No panel required if we are not re-creating data
        if reinit_params is None and reinit_extras is None:
            return

        reinit_widgets = self.make_widgets_from_config(reinit_params,
                                                       reinit_extras,
                                                       modal_message is None,
                                                       slider_width=128)

        # This should really go in the parent class, like submit_button
        if modal_message:

            # Performs tracing
            self.reinit_button = bm.Button(
                align='start',
                button_type='primary',
                height=35,
                label=modal_button_label,
                width=202)

            def trace_apertures_handler(result):
                if result:
                    self.reconstruct_points()

            self.make_ok_cancel_dialog(
                btn=self.reinit_button,
                message="Perform the tracing with the current parameters?",
                callback=trace_apertures_handler)

            self.make_modal(self.reinit_button, modal_message)

            # Reset tracing parameter
            reset_tracing_button = bm.Button(
                align='start',
                button_type='danger',
                height=35,
                id='reset-tracing-pars',
                label="Reset",
                width=202)

            def reset_dialog_handler(result):
                if result:
                    self.reset_tracing_panel()

            self.make_ok_cancel_dialog(
                btn=reset_tracing_button,
                message='Do you want to reset the tracing parameters?',
                callback=reset_dialog_handler)

            # Add to the Web UI
            reinit_widgets.append(self.reinit_button)
            reinit_widgets.append(reset_tracing_button)
            reinit_widgets.append(self.error_alert)

            reinit_panel = column(self.function, *reinit_widgets,
                                  id="left_panel")
        else:
            reinit_panel = column(self.function,
                                  id="left_panel")

        return reinit_panel

    def reset_tracing_panel(self, param=None):
        """
        Reset all the parameters in the Tracing Panel (leftmost column).
        If a param is provided, it resets only this parameter in particular.

        Parameters
        ----------
        param : string
            Parameter name
        """
        for key, val in self.reinit_extras.items():

            if param is None:
                reset_value = val.default
            elif key == param:
                reset_value = self._reinit_extras[key]
            else:
                continue

            old = self.widgets[key].value

            # Update Slider Value
            self.widgets[key].update(value=reset_value)

            # Update Text Field via callback function
            for callback in self.widgets[key]._callbacks['value_throttled']:
                callback(attrib='value_throttled', old=old, new=reset_value)

        for key, val in self.config.items():

            if key not in self.reinit_params:
                continue

            if param is None:
                reset_value = val
            elif key == param:
                reset_value = self._reinit_pars[key]
            else:
                continue

            old = self.widgets[key].value

            # Update Slider Value
            self.widgets[key].update(value=reset_value)

            # Update Text Field via callback function
            for callback in self.widgets[key]._callbacks['value']:
                callback('value', old=old, new=reset_value)

    def register_last_changed(self, key):
        """
        Creates a fallback function called when we update a slider's value.

        Parameters
        ----------
        key : string
            Key associated to the slider widget.

        Returns
        -------
        function : callback function that stores the last modified slider.
        """

        # noinspection PyUnusedLocal
        def _callback(attr, old, new):
            self.last_changed = key

        return _callback

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

        # Edit elements
        self.submit_button.align = "end"
        self.submit_button.height = 35
        self.submit_button.height_policy = "fixed"
        self.submit_button.margin = (0, 5, -35, 5)
        self.submit_button.width = 212
        self.submit_button.width_policy = "fixed"

        self.reinit_panel.css_classes = ["data_source"]
        self.reinit_panel.height_policy = "max"
        self.reinit_panel.width = 212
        self.reinit_panel.width_policy = "fixed"

        # Put all together --- Data provider on the Left
        top_row = row(Spacer(width=250),
                      column(self.get_filename_div(), self.submit_button),
                      css_classes=['top-row'], sizing_mode="scale_width")

        bottom_row = row(self.reinit_panel, self.tabs,
                         css_classes=['bottom-row'], id="bottom_row", spacing=10)

        all_content = column(top_row, bottom_row,
                             id="top_level_layout",
                             spacing=15,
                             sizing_mode="stretch_both")

        # doc.template_variables["primitive_long_help"] = (help.DEFAULT_HELP
        #                                                  + DETAILED_HELP
        #                                                  + help.PLOT_TOOLS_HELP_SUBTEXT
        #                                                  + help.REGION_EDITING_HELP_SUBTEXT)

        doc.add_root(all_content)


def interactive_trace_apertures(ext, config, fit1d_params):
    """
    Run traceApertures() interactively.

    Parameters
    ----------
    ext : AstroData
        Single extension extracted from an AstroData object.
    config : :class:`geminidr.code.spect.traceAperturesConfig`
        Configuration object containing the parameters from traceApertures().
    fit1d_params : dict
        Dictionary containing initial parameters for fitting a model.

    Returns
    -------
    Table : new aperture table.
    """
    ap_table = ext.APERTURE
    fit_par_list = [fit1d_params] * len(ap_table)

    domain_list = [[ap[kw] for kw in ("domain_start", "domain_end")]
                   for ap in ap_table]

    # Create parameters to add to the UI
    reinit_params = ["max_missed", "max_shift", "nsum", "step"]

    if (2 - ext.dispersion_axis()) == 1:
        xlabel = "x / columns [px]"
        ylabel = "y / rows [px]"
    else:
        xlabel = "y / rows [px]"
        ylabel = "x / columns [px]"

    def data_provider(conf, extra):
        """
        Callback function used to recreate the data for fitting.

        Parameters
        ----------
        conf : :class:`geminidr.core.parameters_spect.traceAperturesConfig`
            Standard configuration object.
        extra : dict
            Dictionary containing parameters not defined in the configuration
            object. Not used in this case.
        """
        return trace_apertures_data_provider(ext, conf)

    # noinspection PyTypeChecker
    visualizer = TraceAperturesVisualizer(
        data_provider,
        config=config,
        domains=domain_list,
        filename_info=ext.filename,
        fitting_parameters=fit_par_list,
        help_text=(help.DEFAULT_HELP
                   + help.TRACE_APERTURES
                   + help.PLOT_TOOLS_WITH_SELECT_HELP_SUBTEXT
                   + help.REGION_EDITING_HELP_SUBTEXT),
        primitive_name="traceApertures",
        reinit_params=reinit_params,
        tab_name_fmt="Aperture {}",
        title="Interactive Trace Apertures",
        xlabel=xlabel,
        ylabel=ylabel,
    )

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


# noinspection PyUnusedLocal
def trace_apertures_data_provider(ext, conf):
    """
    Function used by the interactive fitter to generate the a list with
    pairs of [x, y] data containing the knots used for tracing.

    Parameters
    ----------
    ext : AstroData
        Single extension of data containing an .APERTURE table.
    conf : :class:`geminidr.gmos.spect.traceAperturesConfig`
        Dictionary containing default traceApertures() parameters.

    Returns
    -------
    list : pairs of (x, y) for each aperture center, where x is the
    spectral position of the knots, and y is the spacial position of the
    knots.
    """
    all_tracing_knots = []
    dispaxis = 2 - ext.dispersion_axis()  # python sense

    # Convert configuration object into dictionary for easy access to its values
    conf_as_dict = {key: val for key, val in conf.items()}

    for i, loc in enumerate(ext.APERTURE['c0'].data):
        c0 = int(loc + 0.5)
        spectrum = ext.data[c0] if dispaxis == 1 else ext.data[:, c0]
        start = np.argmax(at.boxcar(spectrum, size=20))

        # The coordinates are always returned as (x-coords, y-coords)
        ref_coords, in_coords = tracing.trace_lines(
            ext,
            axis=dispaxis,
            cwidth=5,
            initial=[loc],
            initial_tolerance=None,
            max_missed=conf_as_dict['max_missed'],
            max_shift=conf_as_dict['max_shift'],
            nsum=conf_as_dict['nsum'],
            rwidth=None,
            start=start,
            step=conf_as_dict['step'],
        )

        in_coords = np.ma.masked_array(in_coords)

        # ToDo: This should not be required
        in_coords.mask = np.zeros_like(in_coords)

        spectral_tracing_knots = in_coords[1 - dispaxis]
        spatial_tracing_knots = in_coords[dispaxis]

        all_tracing_knots.append(
            [spectral_tracing_knots, spatial_tracing_knots])

    return all_tracing_knots
