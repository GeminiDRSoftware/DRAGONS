import math
from functools import partial, cmp_to_key

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, CheckboxGroup, ColumnDataSource,
                          Div, LabelSet, NumeralTickFormatter, Select, Slider,
                          Spacer, Span, Spinner, TextInput, Whisker)
from bokeh import models as bm
from bokeh.plotting import figure
from bokeh import events

from geminidr.interactive import server
from geminidr.interactive.controls import Controller
from geminidr.interactive.fit.fit1d import Fit1DPanel
from geminidr.interactive.fit.help import PLOT_TOOLS_HELP_SUBTEXT
from geminidr.interactive.interactive import (PrimitiveVisualizer, RegionEditor,
                                              GIRegionModel, connect_region_model)
from geminidr.interactive.interactive_config import interactive_conf
from geminidr.interactive.interactive_config import show_add_aperture_button
from geminidr.interactive.server import interactive_fitter
from gempy.library.tracing import (find_apertures, find_apertures_peaks,
                                   get_limits, pinpoint_peaks)
from gempy.utils import logutils

USER_MASK_NAME = 'rejected (user)'
BAND_MASK_NAME = 'excluded'

log = logutils.get_logger(__name__)

DETAILED_HELP = """
Lorem ipsum
"""

class BinView:
    def __init__(self, model, width=600, height=400, xpoint='x', ypoint='y',
                 xline='xlinspace', yline='model',
                 xlabel=None, ylabel=None,
                 ):
        fig = figure(plot_width=width, plot_height=height, min_width=400,
                    title='Bin plot', x_axis_label=xlabel, y_axis_label=ylabel,
                    output_backend="webgl", x_range=None, y_range=None,
                    min_border_left=80)
        fig.height_policy = 'fixed'
        fig.width_policy = 'fit'

        data = ColumnDataSource({'x': model['x'], 'y': model['y']})
        fig.scatter(x=xpoint, y=ypoint, source=data, size=5)

        regions_tuples = [slice(start, stop) for (start, stop) in model['regions']]
        region_model = GIRegionModel()
        region_model.load_from_tuples(regions_tuples)
        region_editor = RegionEditor(region_model)

        connect_region_model(fig, region_model)

        #  TODO: To this we'll add a Region Editor as well
        col = column(fig, region_editor.get_widget())
        col.sizing_mode = 'scale_width'
        self.view = col

class EditBinsVisualizer(PrimitiveVisualizer):
    def __init__(self, model, filename_info=''):
        """
        Create a view for editing bins with the given
        :class:`????`

        Parameters
        ----------
        model : dict
            X and Y data plus an initial list of regions
        """
        super().__init__(title='Edit/Set/Select Dispersion Bins',
                         primitive_name='makeSlitIllum',
                         filename_info=filename_info)
        self.model = model
        self.help_text = DETAILED_HELP

    def submit_button_handler(self):
        """
        Submit button handler.

        The parent version checks for bad/poor fits, but that's not an issue
        here, so we just exit by disabling the submit button, which triggers
        some callbacks.
        """
        self.submit_button.disabled = True

    def visualize(self, doc):
        super().visualize(doc)

        for btn in (self.submit_button, self.abort_button):
            btn.align = 'end'
            btn.height = 35
            btn.height_policy = "fixed"
            btn.margin = (0, 5, -20, 5)
            btn.width = 212
            btn.width_policy = "fixed"

        layout_ls = [row(self.abort_button, self.submit_button,
                         align="end", css_classes=['top-row'])]
        # Now, create columns for the plot&region editor, and controls
        # Then append them as a row to layout_ls

        bin_view = BinView(model=self.model)
        layout_ls.append(row(bin_view.view))

        layout = column(*layout_ls)

        doc.add_root(layout)

class BinPanel:
    def __init__(self, data_model, domain=None,
                 x=None, y=None, weights=None, idx=0, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True, plot_ratios=True,
                 enable_user_masking=True, enable_regions=True, central_plot=True, extra_masks=None):
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
        weights : None or :class:`~numpy.ndarray`
            weights of individual points
        xlabel : str
            label for X axis
        ylabel : str
            label for Y axis
        plot_width : int
            width of plot area in pixels
        plot_height : int
            height of plot area in pixels
        plot_residuals : bool
            True if we want the lower plot showing the differential between the data and the fit
        plot_ratios : bool
            True if we want the lower plot showing the ratio between the data and the fit
        enable_user_masking : bool
            True to enable fine-grained data masking by the user using bokeh selections
        enable_regions : bool
            True if we want to allow user-defind regions as a means of masking the data
        extra_masks : dict of boolean arrays
            points to display but not use in the fit
        """
        # Just to get the doc later
        self.visualizer = visualizer
        self.index = idx

        self.width = plot_width
        self.height = plot_height
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.enable_regions = enable_regions
        self.enable_user_masking = enable_user_masking
        self.xpoint = 'x'
        self.ypoint = 'y'
        self.p_main = None
        self.data_model = data_model

        # prep params to clean up sigma related inputs for the interface
        # i.e. niter min of 1, etc.

        # Avoids having to check whether this is None all the time
        band_model = GIRegionModel(domain=domain)

        self.model.add_listener(self.model_change_handler)

        reset_button = bm.Button(label="Reset", align='center',
                                 button_type='warning', width_policy='min')
        self.reset_dialog = self.visualizer.make_ok_cancel_dialog(
            reset_button, 'Reset will change all inputs for this tab back '
            'to their original values.  Proceed?', self.reset_dialog_handler)

        controller_div = Div(margin=(20, 0, 0, 0), width=220,
                             style={"color": "gray", "padding": "5px"})
        controls = column(reset_button, controller_div,
                          width=220)

        fig_column = self.build_figures(domain=domain, controller_div=controller_div,
                                        plot_residuals=plot_residuals,
                                        plot_ratios=plot_ratios,
                                        extra_masks=extra_masks)

        # Initializing regions here ensures the listeners are notified of the region(s)
        region_tuples = [slice(start, stop) for (start, stop) in model['regions']]
        band_model.load_from_tuples(region_tuples)

        # TODO refactor? this is dupe from band_model_handler
        # hacking it in here so I can account for the initial
        # state of the band model (which used to be always empty)
        mask = [BAND_MASK_NAME if not band_model.contains(x) and m == 'good' else m
                for x, m in zip(self.model.x, self.model.mask)]
        self.model.data.data['mask'] = mask
        self.model.perform_fit()

        if enable_regions:
            region_editor = RegionEditor(band_model)
            fig_column.append(region_editor.get_widget())
        col = column(*fig_column)
        col.sizing_mode = 'scale_width'

        col_order = [col, controls] if central_plot else [controls, col]
        self.component = row(*col_order, css_classes=["tab-content"],
                             spacing=10)

    def build_figures(self, domain=None, controller_div=None,
                      plot_residuals=True, plot_ratios=True,
                      extra_masks=None):
        """
        Construct the figures containing the various plots needed for this
        Visualizer.

        Parameters
        ----------
        domain : 2-tuple/None
            the domain over which the model is defined
        controller_div : Div
            Div object accessible by Controller for updating help text
        plot_ratios : bool
            make a ratios plot?
        plot_residuals : bool
            make a residuals plot?
        extra_masks : dict/list/None
            names of additional masks to inform the user about

        Returns
        -------
        fig_column : list
            list of bokeh objects with attached listeners
        """


        fig = figure(width=self.width, data_model = self.data_model, height=self.height, min_width=400,
                    title='Bin plot', xlabel=self.xlabel, ylabel=self.ylabel,
                    output_backend="webgl", x_range=None, y_range=None,
                    min_border_left=80, xmodel=self.model,
                                      enable_user_masking=self.enable_user_masking)
        fig.height_policy = 'fixed'
        fig.width_policy = 'fit'

        data = ColumnDataSource({'x': self.data_model['x'], 'y': self.data_model['y']})
      #  fig.scatter(x=xpoint, y=ypoint, source=data, size=5)

        if self.enable_regions:
            connect_region_model(fig, self.model.band_model)

        if self.enable_user_masking:
            mask_handlers = (self.mask_button_handler,
                             self.unmask_button_handler)
        else:
            mask_handlers = None

        Controller(fig, None, self.model.band_model if self.enable_regions else None, controller_div,
                   mask_handlers=mask_handlers, domain=domain, helpintrotext=
                   "While the mouse is over the upper plot, "
                   "choose from the following commands:")

        # self.add_custom_cursor_behavior(p_main)
        fig_column = [fig]

        self.p_main = fig

        # Do a custom padding for the ranges
        self.reset_view()

        return fig_column

    def reset_view(self):
        """
        This calculates the x and y ranges for the figure with some custom padding.

        This is used when we initially build the figure, but also as a listener for
        whenever the data changes.
        """
        if not hasattr(self, 'p_main') or self.p_main is None:
            # This may be a subclass, p_main is not being stored so nothing to reset
            return

        x_range = None
        y_range = None
        try:
            xdata = self.model.data.data[self.xpoint]
            ydata = self.model.data.data[self.ypoint]
        except (AttributeError, KeyError):
            pass
        else:
            x_min, x_max = min(xdata), max(xdata)
            if x_min != x_max:
                x_pad = (x_max - x_min) * 0.1
                self.p_main.x_range.update(start=x_min - x_pad, end=x_max + x_pad * 2)
            y_min, y_max = min(ydata), max(ydata)
            if y_min != y_max:
                y_pad = (y_max - y_min) * 0.1
                self.p_main.y_range.update(start=y_min - y_pad, end=y_max + y_pad)
        if x_range is not None:
            self.p_main.x_range = x_range
        if y_range is not None:
            self.p_main.y_range = y_range

    def reset_dialog_handler(self, result):
        """
        Reset fit parameter values.

        Parameters
        ----------
        result : bool
            This is the user response to an ok/cancel confirmation dialog.  If False, we do not reset.
        """

    def update_regions(self):
        """ Update fitting regions """
        self.model.regions = self.model.band_model.build_regions()

    def model_change_handler(self, model):
        """
        If the model changes, this gets called to evaluate the fit and save the results.

        Parameters
        ----------
        model : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model that changed.
        """
        model.evaluation.data['model'] = model.evaluate(model.evaluation.data['xlinspace'])

    def mask_button_handler(self, x, y, mult):
        """
        Handler for the mask button.

        When the mask button is clicked, this method
        will find the selected data points and set the
        user mask for them.

        Parameters
        ----------
        x : float
            The pointer x coordinate
        y : float
            The pointer y coordinate
        mult : float
            The ratio for the X-axis vs Y-axis, so we can calculate "pixel distance"
        """
        indices = self.model.data.selected.indices
        if not indices:
            self._point_mask_handler(x, y, mult, 'mask')
        else:
            self.model.data.selected.update(indices=[])
            mask = self.model.mask.copy()
            for i in indices:
                mask[i] = USER_MASK_NAME
            self.model.data.data['mask'] = mask
            self.model.perform_fit()

    def unmask_button_handler(self, x, y, mult):
        """
        Handler for the unmask button.

        When the unmask button is clicked, this method
        will find the selected data points and unset the
        user mask for them.

        Parameters
        ----------
        x : float
            The pointer x coordinate
        y : float
            The pointer y coordinate
        mult : float
            The ratio for the X-axis vs Y-axis, so we can calculate "pixel distance"
        """
        indices = self.model.data.selected.indices
        if not indices:
            self._point_mask_handler(x, y, mult, 'unmask')
        else:
            x_data = self.model.x
            self.model.data.selected.update(indices=[])
            mask = self.model.mask.copy()
            for i in indices:
                if mask[i] == USER_MASK_NAME:
                    mask[i] = ('good' if self.model.band_model.contains(x_data[i])
                               else BAND_MASK_NAME)
            self.model.data.data['mask'] = mask
            self.model.perform_fit()

    def _point_mask_handler(self, x, y, mult, action):
        """
        Handler for the mask button.

        When the mask button is clicked, this method
        will find the selected data points and set the
        user mask for them.

        Parameters
        ----------
        x : float
            X mouse position in pixels inside the canvas.
        y : float
            Y mouse position in pixels inside the canvas.
        mult : float
            The ratio of the x axis vs the y axis, for calculating pixel distance
        action : str
            The type of masking action being done
        """
        dist = None
        sel = None
        xarr = self.model.data.data[self.xpoint]
        yarr = self.model.data.data[self.ypoint]
        mask = self.model.mask
        if action not in ('mask', 'unmask'):
            action = None
        for i, (xd, yd) in enumerate(zip(xarr, yarr)):
            if action is None or ((action == 'mask') ^ (mask[i] == USER_MASK_NAME)):
                if xd is not None and yd is not None:
                    ddist = (x - xd) ** 2 + ((y - yd) * mult) ** 2
                    if dist is None or ddist < dist:
                        dist = ddist
                        sel = i
        if sel is not None:
            # we have a closest point, toggle the user mask
            if mask[sel] == USER_MASK_NAME:
                mask[sel] = ('good' if self.model.band_model.contains(xarr[sel])
                             else BAND_MASK_NAME)
            else:
                mask[sel] = USER_MASK_NAME

        self.model.perform_fit()

    # TODO refactored this down from tracing, but it breaks
    # x/y tracking when the mouse moves in the figure for calculateSensitivity
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


class BinVisualizer(PrimitiveVisualizer):
    def __init__(self, model,
                 xlabel='x', ylabel='y',
                 title=None, primitive_name=None, filename_info=None,
                 template="fit1d.html", help_text=None,
                 ui_params=None, panel_class=BinPanel, pad_buttons=False,
                 **kwargs):
        """
        Parameters
        ----------
        xlabel : str
            String label for X axis
        ylabel : str
            String label for Y axis
        title : str
            Title for UI (Interactive <Title>)
        primitive_name : str
            Name of the primitive this tool is a visualizer for, for displaying in the UI
        template : str
            Name of the HTML template to use for the UI
        help_text : str
            HTML help text for popup help, or None to use the default
        ui_params : :class:`~geminidr.interactive.interactive.UIParams`
            Parameter set for user input
        panel_class : Class
            The class of Panel to use in each tab. This allows specific
            operability for each primitive since most of the functions that do
            the work are methods of this class.
        pad_buttons : bool
            If True, pad the abort/accept buttons so the tabs can flow under them
        """
        super().__init__(title=title, primitive_name=primitive_name, filename_info=filename_info,
                         template=template, help_text=help_text, ui_params=ui_params)
        self.layout = None
        self.model = model
        self.pad_buttons = pad_buttons

        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # Keep a list of panels for access later
        self.panels = list()

        # Make the panel with widgets to control the creation of (x, y) arrays

        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})

        self.tabs = bm.Tabs(css_classes=['tabs'],
                            height_policy="max",
                            width_policy="max",
                            tabs=[], name="tabs")
        self.tabs.sizing_mode = 'scale_width'

       # self._reinit_params = {k: v for k, v in ui_params.values.items()}

    # noinspection PyProtectedMember

    def visualize(self, doc):
        """
        Start the bokeh document using this visualizer.

        This call is responsible for filling in the bokeh document with
        the user interface.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            bokeh document to draw the UI in
        """
        super().visualize(doc)
        col = column(self.tabs, )
        col.sizing_mode = 'scale_width'
        col.width_policy = 'max'

        for btn in (self.submit_button, self.abort_button):
            btn.align = 'end'
            btn.height = 35
            btn.height_policy = "fixed"
            btn.margin = (0, 5, -20 if not self.pad_buttons else 0, 5)
            btn.width = 212
            btn.width_policy = "fixed"

        layout_ls = list()
        if self.filename_info:
            self.submit_button.align = 'end'
            layout_ls.append(row(Spacer(width=250),
                                 column(self.get_filename_div(), row(self.abort_button, self.submit_button)),
                                 Spacer(width=10),
                                 align="end", css_classes=['top-row']))
        else:
            layout_ls.append(row(self.abort_button, self.submit_button))

        bin_view = BinPanel()
        layout_ls.append(row(bin_view))
       # layout_ls.append(col)

        self.layout = column(*layout_ls, sizing_mode="stretch_width")
        doc.add_root(self.layout)

    def results(self):
        """
        Get the results of the interactive fit.

        This gets the list of `~gempy.library.fitting.fit_1D` fits of
        the data to be used by the caller.

        Returns
        -------
        list of `~gempy.library.fitting.fit_1D`
        """
        return [fit.fit for fit in self.fits]




