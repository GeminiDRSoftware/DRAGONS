import numpy as np

from bokeh.layouts import column, row
from bokeh.models import (Button, Div, Spacer, NumericInput)
from bokeh import models as bm
from bokeh.plotting import figure

from geminidr.interactive.controls import Controller
from geminidr.interactive.fit.fit1d import Fit1DRegionListener, InteractiveModel1D
from geminidr.interactive.interactive import (PrimitiveVisualizer, RegionEditor,
                                              GIRegionModel, connect_region_model)
from geminidr.interactive.fit.aperture import CustomWidget
from gempy.utils import logutils

log = logutils.get_logger(__name__)

DETAILED_HELP = """

<h2>Help</h2>

<p>Interface to inspect and edit dispersion bins.</p>
<p>Allows the user to generate a number of equally-spaced bins, and to modify
bin limits, either manually entering them, or using a point-and-click interface.</p>
"""

def tuples_to_slices(tuples):
    """
    Helping function to translate into slices a list of bin limit expressed
    as tuples (start, end)
    """
    return [slice(start, end) for (start, end) in tuples]

def bin_figure(width=None, height=None, xpoint='x', ypoint='y',
                 xlabel=None, ylabel=None, model=None):
    """
    Function to produce bokeh objects for the bin editing plot.
    Listeners are not added here.

    Parameters
    ----------
    width : int
        width of the plots
    height : int
        height of the main plot (ratios/residuals are half-height)
    xpoint, ypoint : str
        column names in model.data containing x and y data for points
    xlabel, ylabel : str
        label for axes of main plot
    model : InteractiveModel1D
        object containing the fit information

    Returns
    -------
    Figure
        The plotting figure
    """

    tools = "pan,wheel_zoom,box_zoom,reset"

    p_main = figure(plot_width=width, plot_height=height, min_width=400,
                    title='Illumination bins', x_axis_label=xlabel, y_axis_label=ylabel,
                    tools=tools,
                    output_backend="webgl", x_range=None, y_range=None,
                    min_border_left=80)
    p_main.height_policy = 'fixed'
    p_main.width_policy = 'fit'
    p_main.scatter(x=xpoint, y=ypoint, source=model.data,
                   size=5, legend_field='mask',
                   **model.mask_rendering_kwargs())

    return p_main

class NumericInputLine(CustomWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_value = self.value
    def build(self):
        self.numeric_input = NumericInput(value=self.value if self.value else 0, width=64, **self.kwargs)
        self.numeric_input.on_change("value", self.handler)
        return row([Div(text=self.title, align='center'),
                    Spacer(width_policy='max'),
                    self.numeric_input])

    def reset(self):
        self.numeric_input.value = self.orig_value

class BinEditor(RegionEditor):
    """
    Specialized RegionEditor. Just for cosmetic changes (changes the title)
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.text_input.title = "Bin limits (i.e. 101:500,511:900,951: Press 'Enter' to apply):"

class BinModel1D(InteractiveModel1D):
    """
    Specialized InteractiveModle1D that removes the fitting functionality, as
    it's not needed for the bin editor.
    """
    def perform_fit(self, *args):
        """
        Dummy function. Needs to be here to be compliant with the InteractiveModel1D
        interface, but we don't want to perform any calculation.
        """
        ...

    def evaluate(self, x):
        """
        Returns the `x` parameter itself. This model does not perform fitting evaluation.
        """
        return x

class BinResettingUI:
    def __init__(self, vis, model, bin_parameters):
        """
        Class to manage the set of UI controls to generate equally-sized bins over the whole dispersion
        range, and to reset to original values.

        Parameters
        ----------
        vis : :class:`~geminidr.interactive.fit.bineditor.BinVisualizer`
            The visualizer related to these inputs
        bin_parameters: dict
            Initial number of bins and generated regions
        fit : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model information for doing the 1-D fit
        """
        self.vis = vis
        self.model = model
        self.original_parameters = bin_parameters
        self.num_of_bins = self.original_parameters['nbins']

        def _generate_handler(result):
            if result:
                generate_button.disabled = True
                def fn():
                    self.generate_model_regions(self.num_of_bins)
                    generate_button.disabled = False
                vis.do_later(fn)

        self.number_bins = NumericInputLine("Number of bins", self,
                                 mode='int', attr="num_of_bins",
                                 placeholder="e.g. 12")

        generate_button = Button(label="Generate bins", button_type='primary',
                                      default_size=200)

        vis.make_ok_cancel_dialog(generate_button,
                                  'All bin limits will be recomputed and changes will be lost. Proceed?',
                                  _generate_handler)
        reset_button = bm.Button(label="Reset", align='center',
                                 button_type='warning', width_policy='min')
        self.reset_dialog = self.vis.make_ok_cancel_dialog(
            reset_button, 'Reset will change all inputs for this tab back '
            'to their original values.  Proceed?', self.reset_dialog_handler)

        self.controls_column = (
            self.number_bins.build(),
            generate_button,
            reset_button,
        )

    def get_bokeh_components(self):
        """
        Return the bokeh components to be added with all the input widgets.

        Returns
        -------
        list : :class:`~bokeh.models.layout.LayoutDOM`
            List of bokeh components to add to the UI
        """
        return self.controls_column

    def generate_model_regions(self, nbins):
        """
        Handle the 'Generate bins' button being clicked.

        This will generate a new set of bin limits and update the model, which
        in turn will update the interface.
        """
        bin_limits = np.linspace(0, self.original_parameters['height'], nbins + 1, dtype=int)
        bin_list = list(zip(bin_limits[:-1], bin_limits[1:]))
        self.model.load_from_tuples(tuples_to_slices(bin_list))

    def reset_model_regions(self):
        """
        Handle the 'Reset' button being clicked.

        This will update the model with the initial bin limit list, which in
        turn will udpate the interface.
        """
        self.number_bins.reset()
        self.model.load_from_tuples(tuples_to_slices(self.original_parameters['bin_list']))

    def reset_dialog_handler(self, result):
       """
        Reset bin limits values.

        Parameters
        ----------
        result : bool
            This is the user response to an ok/cancel confirmation dialog.  If False, we do not reset.
       """
       if result:
           self.reset_model_regions()

class BinPanel:
    def __init__(self, visualizer, regions, bin_parameters, domain=None,
                 x=None, y=None, weights=None, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, central_plot=True):
        """
        Panel for visualizing a 1-D fit, perhaps in a tab

        Parameters
        ----------
        visualizer : :class:`~geminidr.interactive.fit.fit1d.Fit1DVisualizer`
            visualizer to associate with
        regions : list of regions
            ...
        bin_parameters: dict
            Initial number of bins and generated regions
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
        central_plot : bool
            If True, the main plot will be on the left and the control column
            on the right. If False, the opposite.
        """
        # Just to get the doc later
        self.visualizer = visualizer

        # self.title = ""

        self.width = plot_width
        self.height = plot_height
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xpoint = 'x'
        self.ypoint = 'y'
        self.p_main = None

        # Avoids having to check whether this is None all the time
        band_model = GIRegionModel(domain=domain, support_adjacent=True)
        self.model = BinModel1D({}, domain, x, y, weights, band_model=band_model)
        self.model.add_listener(self.model_change_handler)

        self.bin_resetting_ui = BinResettingUI(visualizer, band_model, bin_parameters)
        controls_column = self.bin_resetting_ui.get_bokeh_components()
        # reset_button = bm.Button(label="Reset", align='center',
        #                          button_type='warning', width_policy='min')

        # self.reset_dialog = self.visualizer.make_ok_cancel_dialog(
        #     reset_button, 'Reset will change all inputs for this tab back '
        #     'to their original values.  Proceed?', self.reset_dialog_handler)

        controller_div = Div(margin=(20, 0, 0, 0), width=220,
                             style={"color": "gray", "padding": "5px"})
        controls = column(*controls_column, controller_div,
                          width=220)

        fig_column = self.build_figures(domain=domain, controller_div=controller_div)

        # Initializing regions here ensures the listeners are notified of the region(s)
        band_model.load_from_tuples(tuples_to_slices(regions))

        region_editor = BinEditor(band_model)
        fig_column.append(region_editor.get_widget())
        col = column(*fig_column)
        col.sizing_mode = 'scale_width'

        col_order = [col, controls] if central_plot else [controls, col]
        self.component = row(*col_order, css_classes=["tab-content"],
                             spacing=10)

    def build_figures(self, domain=None, controller_div=None):
        """
        Construct the figure containing the plot needed for this
        Visualizer.

        Parameters
        ----------
        domain : 2-tuple/None
            the domain over which the model is defined
        controller_div : Div
            Div object accessible by Controller for updating help text

        Returns
        -------
        fig_column : list
            list of bokeh objects with attached listeners
        """

        p_main = bin_figure(width=self.width, height=self.height,
                            xpoint=self.xpoint, ypoint=self.ypoint,
                            xlabel=self.xlabel, ylabel=self.ylabel, model=self.model)
        self.model.band_model.add_listener(Fit1DRegionListener(self.update_bin_limits))
        connect_region_model(p_main, self.model.band_model)

        Controller(p_main, None, self.model.band_model, controller_div,
                   mask_handlers=None, domain=domain, helpintrotext=
                   "While the mouse is over the upper plot, "
                   "choose from the following commands:")

        fig_column = [p_main]

        self.p_main = p_main

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

    def update_bin_limits(self):
        """ Update bin limits """
        self.model.regions = self.model.band_model.build_regions()

    def model_change_handler(self, model):
        """
        If the model changes, this gets called to save the results.

        Parameters
        ----------
        model : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model that changed.
        """
        # We're not evaluating fits, but we're reusing existing models so
        # we'll follow the interface used elsewhere.
        model.evaluation.data['model'] = model.evaluate(model.evaluation.data['xlinspace'])

class BinVisualizer(PrimitiveVisualizer):
    """
    Specialized visualizer for displaying and editing bin limits.

    Attributes:
        tabs: layout containing all the stuff required for an interactive 1D fit
        submit_button: the button signifying a successful end to the interactive session

        config: the Config object describing the parameters are their constraints
        widgets: a dict of (param_name, widget) elements that allow the properties
                 of the widgets to be set/accessed by the calling primitive. So far
                 I'm only including the widgets in the reinit_panel
    """
    def __init__(self, data_source, xlabel='Column', ylabel='Signal',
                 domain=None, title=None, primitive_name=None, filename_info=None,
                 template="fit1d.html", help_text=None,
                 ui_params=None, pad_buttons=False,
                 **kwargs):
        """
        Parameters
        ----------
        data_source : dict or dict-returning function
            input data or the function to calculate the input data. The dict
            must have keys of "x" and "y" indicating the input data values,
            with each being an array or a list of arrays. There are also
            optional "weights" (array of weights), "*_mask" (additional input masks),
            and "meta" (any additional data)
        xlabel : str
            String label for X axis
        ylabel : str
            String label for Y axis
        domain : list
            Domains for the input
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
        pad_buttons : bool
            If True, pad the abort/accept buttons so the tabs can flow under them
        """
        super().__init__(title=title, primitive_name=primitive_name, filename_info=filename_info,
                         template=template, help_text=help_text, ui_params=ui_params)
        self.layout = None
        self.pad_buttons = pad_buttons

        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # Keep a list of panels for access later
        self.panels = list()

        # Make the panel with widgets to control the creation of (x, y) arrays

        if callable(data_source):
            self.reconstruct_points_fn = data_source
            data = data_source(ui_params=ui_params)
        else:
            data = data_source
            self.reconstruct_points_fn = None
        self.returns_list = isinstance(data["x"], list)

        if data["x"].size != data["y"].size:
            raise ValueError("Different (x, y) array sizes")
        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})

        # TODO: We don't need tabs, but the basic Fit1DVisualizer used
        #       as the base here does and I want to keep the functionality
        #       until everything else works. Afterwards, figure out how
        #       to get rid of it.
        self.tabs = bm.Tabs(css_classes=['tabs'],
                            height_policy="max",
                            width_policy="max",
                            tabs=[], name="tabs")
        self.tabs.sizing_mode = 'scale_width'
        this_dict = data

        tui = BinPanel(self, domain=domain, **this_dict, **kwargs)
        self.model = tui.model
        tab = bm.Panel(child=tui.component, title='Bin editor')
        self.tabs.tabs.append(tab)
        self.fits.append(tui.model)
        self.panels.append(tui)

        self._reinit_params = {k: v for k, v in (ui_params.values.items() if ui_params else {})}

    def submit_button_handler(self):
        """
        Submit button handler.

        The parent version checks for bad/poor fits, but that's not an issue
        here, so we just exit by disabling the submit button, which triggers
        some callbacks.
        """
        self.submit_button.disabled = True

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
            layout_ls.append(row(self.abort_button, self.submit_button,
                             align="end", css_classes=['top-row']))

        layout_ls.append(col)
        self.layout = column(*layout_ls, sizing_mode="stretch_width")
        doc.add_root(self.layout)

    def reconstruct_points(self):
        """
        Reconstruct the initial points to work with.

        This is expected to be expensive.  The core inputs
        are separated out in the UI as they are too slow to
        be interactive.  When a user is ready and submits
        updated core config parameters, this is what gets
        executed.  The configuration is updated with the
        new values form the user.  The UI is disabled and the
        expensive function is wrapped in the bokeh Tornado
        event look so the modal dialog can display.
        """
        rollback_config = self.ui_params.values.copy()
        def fn():
            """Top-level code to update the Config with the values from the widgets"""
            config_update = {k: (v.value if getattr(v, "show_value", True) else None)
                if hasattr(v, "value") else bool(v.active) for k, v in self.widgets.items()}
            self.ui_params.update_values(**config_update)

        self.do_later(fn)

        if self.reconstruct_points_fn is not None:
            def rfn():
                data = None
                try:
                    data = self.reconstruct_points_fn(ui_params=self.ui_params)
                except Exception as e:
                    # something went wrong, let's revert the inputs
                    # handling immediately to specifically trap the reconstruct_points_fn call
                    self.ui_params.update_values(**rollback_config)
                    self.show_user_message("Unable to build data from inputs, reverting")
                if data is not None:
                    for i, fit in enumerate(self.fits):
                        if self.returns_list:
                            this_dict = {k: v[i] for k, v in data.items()}
                        else:
                            this_dict = data
                        fit.populate_bokeh_objects(this_dict["x"], this_dict["y"],
                                                   this_dict.get("weights"))
                        fit.perform_fit()

                for pnl in self.panels:
                    pnl.reset_view()

            self.do_later(rfn)

    def results(self):
        """
        Get the results of the interactive bin editing.

        This uses the region model to generate a list of bin limits, to be used
         by the caller.

        Returns
        -------
        String representation of the bin limits list.
        """
        return self.model.regions


