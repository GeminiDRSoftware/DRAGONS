from abc import ABC, abstractmethod

import numpy as np

from bokeh import models as bm, transform as bt
from bokeh.layouts import row, column
from bokeh.models import Div, Select, Range1d, Spacer
from bokeh.plotting import figure
from bokeh import events

from geminidr.interactive import interactive
from geminidr.interactive.controls import Controller, Handler
from geminidr.interactive.interactive import GIRegionModel, connect_figure_extras, GIRegionListener, \
    RegionEditor, do_later
from geminidr.interactive.interactive_config import interactive_conf
from gempy.library.astrotools import cartesian_regions_to_slices
from gempy.library.fitting import fit_1D


def build_fit_1D(fit1d_params, data, points, weights, domain):
    """
    Create a fit_1D from the given parameter dictionary and x/y/weights

    Parameters
    ----------
    fit1d_params : dict
        Dictionary of parameters for the fit_1D
    data : list
        X coordinates
    points : list
        Y values
    weights : list
        weights

    Returns
    -------
        :class:`~gempy.library.fitting.fit_1D` fitter
    """
    return fit_1D(data,
                  points=points,
                  weights=weights,
                  domain=domain,
                  **fit1d_params)


SIGMA_MASK_NAME = 'rejected (sigma)'
USER_MASK_NAME = 'rejected (user)'
BAND_MASK_NAME = 'excluded'


class InteractiveModel(ABC):
    MASK_TYPE = [BAND_MASK_NAME, USER_MASK_NAME, 'good', SIGMA_MASK_NAME]
    MARKERS = ['triangle', 'inverted_triangle', 'circle', 'square']
    PALETTE = ['lightsteelblue', 'lightskyblue', 'black', 'darksalmon']  # Category10[4]
    """
    Base class for all interactive models, containing:
        (a) the parameters of the model
        (b) the parameters that control the fitting (e.g., sigma-clipping)
        (c) the way the fitting is performed
        (d) the input and output coordinates, mask, and weights

    """

    def __init__(self):
        bokeh_data_color = interactive_conf().bokeh_data_color
        InteractiveModel.PALETTE[2] = bokeh_data_color

        self.listeners = []
        self.mask_listeners = []
        self.data = None

    def add_listener(self, listener):
        """
        Add a function to call when the model is updated

        Parameters
        ----------
        listener : function
            This should be a no-arg function and it will get called when the model is updated
        """
        if not callable(listener):
            raise ValueError("Listeners must be callables")
        self.listeners.append(listener)

    def notify_listeners(self):
        """
        Notify all the registered listeners of a change.

        This calls all our registered listener functions to let them know we have changed.
        """
        for listener in self.listeners:
            listener()

    def add_mask_listener(self, mask_listener):
        if not callable(mask_listener):
            raise ValueError("Mask Listener must be callable")
        self.mask_listeners.append(mask_listener)

    def notify_mask_listeners(self):
        for mask_listener in self.mask_listeners:
            mask_listener(self.data.data['mask'])

    @abstractmethod
    def perform_fit(self):
        """
        Perform the fit (Base method, override)
        """
        pass

    @abstractmethod
    def evaluate(self, x):
        """
        Evaluate X

        Parameters
        ----------
        x

        Returns
        -------

        """
        pass

    def mask_rendering_kwargs(self):
        """
        Get the marks and colors to use for the various point masks

        Returns
        -------
        dict : Returns a dict for bokeh that describes the markers and pallete
        """
        return {'marker': bt.factor_mark('mask', self.MARKERS, self.MASK_TYPE),
                'color': bt.factor_cmap('mask', self.PALETTE, self.MASK_TYPE)}


class InteractiveModel1D(InteractiveModel):
    """
    Subclass for 1D models
    """

    def __init__(self, fitting_parameters, domain, x=None, y=None, weights=None, mask=None,
                 section=None, listeners=[], band_model=None):
        """
        Create base class with given parameters as initial model inputs.

        Parameters
        ----------
        x : :class:`~numpy.ndarray`
            list of x coordinate values
        y : :class:`~numpy.ndarray`
            list of y coordinate values
        mask : array of str
            array of mask names for each point
        section
        """
        super().__init__()

        self.band_model = band_model
        if band_model:
            band_model.add_listener(Fit1DRegionListener(self.band_model_handler))

        self.fitting_parameters = fitting_parameters
        self.domain = domain
        self.fit = None
        self.listeners = listeners

        self.section = section
        self.data = bm.ColumnDataSource({'x': [], 'y': [], 'mask': []})
        if isinstance(self.domain, int):
            xlinspace = np.linspace(0, self.domain, 500)
        elif len(self.domain) == 1:
            xlinspace = np.linspace(0, *self.domain, 500)
        else:
            xlinspace = np.linspace(*self.domain, 500)
        weights = self.populate_bokeh_objects(x, y, weights=weights, mask=mask)
        self.weights = weights

        self.sigma_clip = "sigma" in fitting_parameters and fitting_parameters["sigma"]
        self.perform_fit()
        self.evaluation = bm.ColumnDataSource({'xlinspace': xlinspace,
                                               'model': self.evaluate(xlinspace)})

    def set_function(self, fn):
        """
        Set the fit function to use.

        This sets the function the `gempy.library.fitting.fit_1D` fitter will use
        to perform the data fit.  It's a helper method to pass the function down
        to the model.

        Parameters
        ----------
        fn : str
            Which fitter to use
        """
        self.fitting_parameters["function"] = fn

    @property
    def regions(self):
        """
        Get the regions of the fitter.

        Returns
        -------
        tuple of tuples : `regions` of the model we are wrapping
        """
        return self.fitting_parameters["regions"]

    @regions.setter
    def regions(self, regions):
        """
        Set the regions in this fitter.

        This sets the regions.

        Parameters
        ----------
        regions : tuple of tuples
            regions to use in the fit
        """
        self.fitting_parameters["regions"] = regions

    def populate_bokeh_objects(self, x, y, weights, mask=None):
        """
        Initializes bokeh objects like a coord structure with extra
        columns for ratios and residuals and setting up masking

        Parameters
        ----------
        x : array of double
            x coordinate values
        y : array of double
            y coordinate values
        mask : array of str
            named mask for coordinates
        """
        if mask is None:
            try:  # Handle y as masked array
                if any(y.mask):
                    init_mask = y.mask
                else:
                    init_mask = np.array(np.zeros_like(x, dtype=bool))
            except AttributeError:
                init_mask = np.array(np.zeros_like(x, dtype=bool))
            else:
                y = y.data
        else:
            init_mask = mask

        x = x[~init_mask]
        y = y[~init_mask]
        if weights is not None:
            weights = weights[~init_mask]

        # "section" is the valid section provided by the user,
        # i.e., points not in this region(s) are user-masked
        if self.section is None:
            mask = ['good'] * len(x)
        else:
            user_mask = np.array(np.ones_like(x, dtype=bool))
            for slice_ in self.section:
                user_mask[slice_.start < x < slice_.stop] = False
            mask = list(np.where(user_mask, USER_MASK_NAME, 'good'))

        # Might put the variance in here for errorbars, but it's not needed
        # at the moment

        # need to setup the mask
        for i in np.arange(len(x)):
            if self.band_model.contains(x[i]):
                # User mask takes preference
                if mask[i] != USER_MASK_NAME:
                    mask[i] = 'good'
            elif mask[i] != USER_MASK_NAME:
                mask[i] = BAND_MASK_NAME
        bokeh_data = {'x': x, 'y': y, 'mask': mask}
        for extra_column in ('residuals', 'ratio'):
            if extra_column in self.data.data:
                bokeh_data[extra_column] = np.zeros_like(y)
        self.data.data = bokeh_data

        self.notify_mask_listeners()

        return weights

    def band_model_handler(self):
        """
        Respond when the band model changes.

        When the band model has changed, we
        brute force a new band mask by checking
        each x coordinate against the band model
        for inclusion.  The band model handles
        the case where there are no bands by
        marking all points as included.
        """
        x_data = self.data.data['x']
        mask = self.data.data['mask'].copy()
        for i in np.arange(len(x_data)):
            if self.band_model.contains(x_data[i]):
                # User mask takes preference
                if mask[i] != USER_MASK_NAME:
                    mask[i] = 'good'
            elif mask[i] != USER_MASK_NAME:
                mask[i] = BAND_MASK_NAME
        self.data.data['mask'] = mask
        # Band operations can come in through the keypress URL
        # so we defer the fit back onto the Bokeh IO event loop

        do_later(self.perform_fit)

    @property
    def x(self):
        """
        maps x attribute internally to bokeh structures

        Returns
        -------
        array of double : x coordinates
        """
        return np.asarray(self.data.data['x'])

    @property
    def y(self):
        """
        maps y attribute internally to bokeh structures

        Returns
        -------
        array of double : y coordinates
        """
        return np.asarray(self.data.data['y'])

    @property
    def mask(self):
        """
        maps mask attribute internally to bokeh structures

        Returns
        -------
        list of str : mask values
        """
        return self.data.data['mask']

    @property
    def sigma(self):
        """
        Maps sigma attribute to :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.lsigma`
        and :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.hsigma`

        Returns
        -------
        double : average of the two sigmae
        """
        if self.lsigma == self.hsigma:
            return self.lsigma
        return 0.5 * (self.lsigma + self.hsigma)  # do something better?

    @sigma.setter
    def sigma(self, value):
        """
        Set sigma attr, effectively setting both :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.lsigma`
        and :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.hsigma`

        Parameters
        ----------
        value : float
            new value for sigma rejection
        """
        self.lsigma = self.hsigma = float(value)

    def perform_fit(self, *args):
        """
        Perform the fit.

        This performs the fit in the contained model, updates the mask,
        and recalculates the plots for residuals and ratio, if present.
        It then notifies all listeners that the data and model have
        changed so they can respond.
        """
        # Note that band_mask is now handled by passing a region string to fit_1D
        # but we still use the band_mask for highlighting the affected points

        goodpix = np.array([m != USER_MASK_NAME for m in self.data.data['mask']])

        if self.sigma_clip:
            fitparms = {x: y for x, y in self.fitting_parameters.items()
                        if x not in ['sigma']}
        else:
            fitparms = {x: y for x, y in self.fitting_parameters.items()
                        if x not in ['sigma_lower', 'sigma_upper', 'niter', 'sigma']}

        self.fit = build_fit_1D(fitparms, self.y[goodpix], points=self.x[goodpix],
                                domain=self.domain, weights=None if self.weights is None else self.weights[goodpix])
        self.fit_mask = np.zeros_like(self.x, dtype=bool)
        if self.sigma_clip:
            # Now pull in the sigma mask
            self.fit_mask[goodpix] = self.fit.mask

        self.update_mask()
        if 'residuals' in self.data.data:
            self.data.data['residuals'] = self.y - self.evaluate(self.x)
        if 'ratio' in self.data.data:
            self.data.data['ratio'] = self.y / self.evaluate(self.x)

        for ll in self.listeners:
            ll(self.fit)

    def update_mask(self):
        goodpix = np.array([m != USER_MASK_NAME for m in self.data.data['mask']])
        mask = self.data.data['mask'].copy()
        fit_mask = np.zeros_like(self.x, dtype=bool)
        if self.sigma_clip:
            # Now pull in the sigma mask
            fit_mask[goodpix] = self.fit.mask
        for i in range(fit_mask.size):
            if fit_mask[i] and mask[i] == 'good':
                mask[i] = SIGMA_MASK_NAME
        self.data.data['mask'] = mask

    def evaluate(self, x):
        return self.fit.evaluate(x)


class FittingParametersUI:
    def __init__(self, vis, fit, fitting_parameters):
        self.vis = vis
        self.fit = fit
        self.saved_sigma_clip = self.fit.sigma_clip
        self.fitting_parameters = fitting_parameters
        self.fitting_parameters_for_reset = {x: y for x, y in self.fitting_parameters.items()}

        if 'function' in vis.config._fields:
            fn = vis.config.function
            fn_allowed = [k for k in vis.config._fields['function'].allowed.keys()]

            # Dropdown for selecting fit_1D function
            self.function = Select(title="Fitting Function:", value=fn,
                                   options=fn_allowed, width=200)

            def fn_select_change(attr, old, new):
                self.fit.set_function(new)
                self.fit.perform_fit()

            self.function.on_change('value', fn_select_change)
        else:
            # If the function is fixed
            self.function = bm.Div(
            text=f"Fit Function: <b>{fitting_parameters['function'].capitalize()}</b>",
            min_width=100, max_width=202, sizing_mode='stretch_width',
            style={"color": "black", "font-size": "115%", "margin-top": "5px"},
            width_policy='max')

        self.description = self.build_description()

        self.order_slider = interactive.build_text_slider(
            "Order", fitting_parameters["order"], None, None, None,
            fitting_parameters, "order", fit.perform_fit, throttled=True,
            config=vis.config, slider_width=128)
        self.sigma_upper_slider = interactive.build_text_slider(
            "Sigma (Upper)", fitting_parameters["sigma_upper"], None, None,
            None, fitting_parameters, "sigma_upper", self.sigma_slider_handler,
            throttled=True, config=vis.config, slider_width=128)
        self.sigma_lower_slider = interactive.build_text_slider(
            "Sigma (Lower)", fitting_parameters["sigma_lower"], None, None,
            None, fitting_parameters, "sigma_lower", self.sigma_slider_handler,
            throttled=True, config=vis.config, slider_width=128)
        self.niter_slider = interactive.build_text_slider(
            "Max iterations", fitting_parameters["niter"], None, 1, None,
            fitting_parameters, "niter", fit.perform_fit, throttled=True,
            config=vis.config, slider_width=128)
        if "grow" in fitting_parameters:  # not all have them
            self.grow_slider = interactive.build_text_slider(
                "Grow", fitting_parameters["grow"], None, None, None,
                fitting_parameters, "grow", fit.perform_fit, throttled=True,
                config=vis.config, slider_width=128)

        self.sigma_button = bm.CheckboxGroup(labels=['Sigma clip'], active=[0] if self.fit.sigma_clip else [])
        self.sigma_button.on_change('active', self.sigma_button_handler)

        self.enable_disable_sigma_inputs()

        self.controls_column = self.build_column()

    def enable_disable_sigma_inputs(self):
        # enable/disable sliders
        disabled = not self.fit.sigma_clip
        for c in self.niter_slider.children:
            c.disabled = disabled
        for c in self.sigma_upper_slider.children:
            c.disabled = disabled
        for c in self.sigma_lower_slider.children:
            c.disabled = disabled
        if hasattr(self, "grow_slider"):
            for c in self.grow_slider.children:
                c.disabled = disabled

    def build_column(self):
        """
        Builds a list with the components that belong to the Fit 1D Parameters
        column. The element's order inside the list represent the top-to-bottom
        order of the elements in the column.

        Return
        ------
        list : elements displayed in the column.
        """

        rejection_title = bm.Div(
            text="Rejection Parameters",
            min_width=100,
            max_width=202,
            sizing_mode='stretch_width',
            style={"color": "black", "font-size": "115%", "margin-top": "10px"},
            width_policy='max',
        )

        if self.function:
            column_list = [self.function, self.order_slider, rejection_title,
                           self.sigma_button, self.niter_slider,
                           self.sigma_lower_slider, self.sigma_upper_slider]
        else:
            column_title = bm.Div(
                text=f"Fit Function: <b>{self.vis.function_name.capitalize()}</b>",
                min_width=100,
                max_width=202,
                sizing_mode='stretch_width',
                style={"color": "black", "font-size": "115%", "margin-top": "5px"},
                width_policy='max',
            )
            column_list = [column_title, self.order_slider, rejection_title,
                           self.sigma_button, self.niter_slider,
                           self.sigma_lower_slider,
                           self.sigma_upper_slider]
        if hasattr(self, "grow_slider"):
            column_list.append(self.grow_slider)

        return column_list

    def build_description(self, text=""):
        """
        Adds a description text to the fitting function parameters panel.

        Parameters
        ----------
        text : str
            Text that shows up in the description.

        Return
        ------
        bokeh.models.Div : Div component containing the short description.
        """
        return bm.Div(
            text=text,
            min_width=100,
            max_width=202,
            sizing_mode='stretch_width',
            style={"color": "black", "font-size": "115%", "margin-top": "10px"},
            width_policy='min',
        )

    def reset_ui(self):
        self.fitting_parameters = {x: y for x, y in self.fitting_parameters_for_reset.items()}
        for key in ("order", "sigma_upper", "sigma_lower", "niter", "grow"):
            try:
                slider = self.getattr(f"{key}_slider")
            except AttributeError:
                pass
            else:
                slider.children[0].value = self.fitting_parameters[key]
                slider.children[1].value = self.fitting_parameters[key]
        self.sigma_button.active = [0] if self.saved_sigma_clip else []
        self.fit.perform_fit()

    def get_bokeh_components(self):
        return self.controls_column

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
        self.enable_disable_sigma_inputs()
        self.fit.perform_fit()

    def sigma_slider_handler(self, val):
        """
        Handle the sigma clipping being adjusted.

        This will trigger a fit since the result may
        change.
        """
        # If we're not sigma-clipping, we don't need to refit the model if sigma changes
        if self.fit.sigma_clip:
            self.fit.perform_fit()


class InfoPanel:
    def __init__(self):
        self.rms = 0.0
        self.band_count = 0
        self.user_count = 0
        self.fit_count = 0
        self.component = Div(text='')
        self.update_panel()

    def update_panel(self):
        rms = '<b>RMS:</b> {rms:.4f}<br/>'.format(rms=self.rms)
        band = '<b>Band Masked:</b> {band_count}<br/>'.format(band_count=self.band_count) if self.band_count else ''
        user = '<b>User Masked:</b> {user_count}<br/>'.format(user_count=self.user_count) if self.user_count else ''
        fit = '<b>Fit Masked:</b> {fit_count}<br/>'.format(fit_count=self.fit_count) if self.fit_count else ''

        self.component.update(text=rms + band + user + fit)

    def update(self, f):
        self.rms = f.rms
        self.update_panel()

    def update_mask(self, mask):
        self.band_count = mask.count(BAND_MASK_NAME)
        self.user_count = mask.count(USER_MASK_NAME)
        self.fit_count = mask.count(SIGMA_MASK_NAME)
        self.update_panel()


class Fit1DPanel:
    def __init__(self, visualizer, fitting_parameters, domain, x, y, idx=0,
                 weights=None, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True, plot_ratios=True,
                 enable_user_masking=True, enable_regions=True, central_plot=True):
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
        """
        # Just to get the doc later
        self.visualizer = visualizer
        self.index = idx
        self.info_panel = InfoPanel()
        self.info_div = self.info_panel.component

        # Make a listener to update the info panel with the RMS on a fit
        listeners = [lambda f: self.info_panel.update(f), ]

        # prep params to clean up sigma related inputs for the interface
        # i.e. niter min of 1, etc.
        prep_fit1d_params_for_fit1d(fitting_parameters)

        # Avoids having to check whether this is None all the time
        band_model = GIRegionModel(domain=domain)

        self.fitting_parameters = fitting_parameters
        self.model = InteractiveModel1D(self.fitting_parameters, domain, x, y, weights,
                                        listeners=listeners, band_model=band_model)

        # also listen for updates to the masks
        self.model.add_mask_listener(self.info_panel.update_mask)

        model = self.model
        self.fitting_parameters_ui = FittingParametersUI(visualizer, model, self.fitting_parameters)

        controls_ls = list()

        controls_column = self.fitting_parameters_ui.get_bokeh_components()

        reset_button = bm.Button(label="Reset", align='center', button_type='warning', width_policy='min')

        self.reset_dialog = self.visualizer.make_ok_cancel_dialog(reset_button,
                                                                  'Reset will change all inputs for this tab back '
                                                                  'to their original values.  Proceed?',
                                                                  self.reset_dialog_handler)

        controller_div = Div(margin=(20, 0, 0, 0),
                             width=220,
                             style={
                                 "color": "gray",
                                 "padding": "5px",
                             })

        controls_ls.extend(controls_column)

        controls_ls.append(reset_button)
        controls_ls.append(controller_div)

        controls = column(*controls_ls, width=220)

        # Now the figures
        x_range = None
        y_range = None
        try:
            if self.model.data and 'x' in self.model.data.data and len(self.model.data.data['x']) >= 2:
                x_min = min(self.model.data.data['x'])
                x_max = max(self.model.data.data['x'])
                x_pad = (x_max - x_min) * 0.1
                x_range = Range1d(x_min - x_pad, x_max + x_pad * 2)
            if self.model.data and 'y' in self.model.data.data and len(self.model.data.data['y']) >= 2:
                y_min = min(self.model.data.data['y'])
                y_max = max(self.model.data.data['y'])
                y_pad = (y_max - y_min) * 0.1
                y_range = Range1d(y_min - y_pad, y_max + y_pad)
        except:
            pass  # ok, we don't *need* ranges...
        if enable_user_masking:
            tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap"
        else:
            tools = "pan,wheel_zoom,box_zoom,reset"
        p_main = figure(plot_width=plot_width, plot_height=plot_height,
                        min_width=400,
                        title='Fit', x_axis_label=xlabel, y_axis_label=ylabel,
                        tools=tools,
                        output_backend="webgl", x_range=x_range, y_range=y_range)
        p_main.height_policy = 'fixed'
        p_main.width_policy = 'fit'

        if enable_regions:
            band_model.add_listener(Fit1DRegionListener(self.update_regions))

            connect_figure_extras(p_main, band_model)

        if enable_user_masking:
            mask_handlers = (self.mask_button_handler,
                             self.unmask_button_handler)
        else:
            mask_handlers = None

        Controller(p_main, None, band_model, controller_div, mask_handlers=mask_handlers,
                   domain=domain)
        # self.add_custom_cursor_behavior(p_main)
        fig_column = [p_main, self.info_div]

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
            p_resid.sizing_mode = 'stretch_width'
            connect_figure_extras(p_resid, band_model)
            # Initalizing this will cause the residuals to be calculated
            self.model.data.data['residuals'] = np.zeros_like(self.model.x)
            p_resid.scatter(x='x', y='residuals', source=self.model.data,
                            size=5, legend_field='mask', **self.model.mask_rendering_kwargs())
        if plot_ratios:
            p_ratios = figure(plot_width=plot_width, plot_height=plot_height // 2,
                              min_width=400,
                              title='Fit Ratios',
                              x_axis_label=xlabel, y_axis_label='Ratio',
                              tools="pan,box_zoom,reset",
                              output_backend="webgl", x_range=p_main.x_range, y_range=None)
            p_ratios.height_policy = 'fixed'
            p_ratios.width_policy = 'fit'
            p_ratios.sizing_mode = 'stretch_width'
            connect_figure_extras(p_ratios, band_model)
            # Initalizing this will cause the residuals to be calculated
            self.model.data.data['ratio'] = np.zeros_like(self.model.x)
            p_ratios.scatter(x='x', y='ratio', source=self.model.data,
                             size=5, legend_field='mask', **self.model.mask_rendering_kwargs())
        if plot_residuals and plot_ratios:
            tabs = bm.Tabs(tabs=[], sizing_mode="scale_width")
            tabs.tabs.append(bm.Panel(child=p_resid, title='Residuals'))
            tabs.tabs.append(bm.Panel(child=p_ratios, title='Ratios'))
            fig_column.append(tabs)
        elif plot_residuals:
            fig_column.append(p_resid)
        elif plot_ratios:
            fig_column.append(p_ratios)

        # Initializing regions here ensures the listeners are notified of the region(s)
        if "regions" in fitting_parameters and fitting_parameters["regions"] is not None:
            region_tuples = cartesian_regions_to_slices(fitting_parameters["regions"])
            band_model.load_from_tuples(region_tuples)

        self.scatter = p_main.scatter(x='x', y='y', source=self.model.data,
                                      size=5, legend_field='mask',
                                      **self.model.mask_rendering_kwargs())
        self.model.add_listener(self.model_change_handler)

        # TODO refactor? this is dupe from band_model_handler
        # hacking it in here so I can account for the initial
        # state of the band model (which used to be always empty)
        mask = [BAND_MASK_NAME if not band_model.contains(x) and m == 'good' else m
                for x, m in zip(self.model.x, self.model.mask)]
        model.data.data['mask'] = mask
        self.model.perform_fit()

        self.line = p_main.line(x='xlinspace',
                                y='model',
                                source=self.model.evaluation,
                                line_width=3,
                                color='crimson')

        if band_model:
            region_editor = RegionEditor(band_model)
            fig_column.append(region_editor.get_widget())
        col = column(*fig_column)
        col.sizing_mode = 'scale_width'

        if central_plot:
            self.component = row(col, controls,
                                 css_classes=["tab-content"],
                                 spacing=10)
        else:
            self.component = row(controls, col,
                                 css_classes=["tab-content"],
                                 spacing=10)

    def reset_dialog_handler(self, result):
        """
        Reset fit parameter values.
        """
        if result:
            self.fitting_parameters_ui.reset_ui()

    def update_regions(self):
        """ Update fitting regions """
        self.model.regions = self.model.band_model.build_regions()

    def model_change_handler(self, fit):
        """
        If the `~fit` changes, this gets called to evaluate the fit and save the results.
        """
        self.model.evaluation.data['model'] = self.model.evaluate(self.model.evaluation.data['xlinspace'])

    def mask_button_handler(self, x, y, mult):
        """
        Handler for the mask button.

        When the mask button is clicked, this method
        will find the selected data points and set the
        user mask for them.

        Parameters
        ----------
        stuff : any
            This is ignored, but the button passes it
        """
        indices = self.model.data.selected.indices
        if not indices:
            self._point_mask_handler(x, y, mult, 'mask')
        else:
            self.model.data.selected.update(indices=[])
            mask = self.model.data.data['mask'].copy()
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
        stuff : any
            This is ignored, but the button passes it
        """
        indices = self.model.data.selected.indices
        x_data = self.model.x
        if not indices:
            self._point_mask_handler(x, y, mult, 'unmask')
        else:
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
        """
        dist = None
        sel = None
        xarr, yarr = self.model.x, self.model.y
        mask = self.model.mask
        if action not in ('mask', 'unmask'):
            action = None
        for i in range(len(xarr)):
            if action is None or ((action == 'mask') ^ (mask[i] == USER_MASK_NAME)):
                xd = xarr[i]
                yd = yarr[i]
                if xd is not None and yd is not None:
                    ddist = (x - xd) ** 2 + ((y - yd) * mult) ** 2
                    if dist is None or ddist < dist:
                        dist = ddist
                        sel = i
        if sel is not None:
            # we have a clos_maskest point, toggle the user mask
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


class Fit1DRegionListener(GIRegionListener):
    """
    Wrapper class so we can just detect when a bands are finished. We don't want
    to do an expensive recalc as a user is dragging a band around. It creates a
    band listener that just updates on `finished`.

    Parameters
    ----------
    fn : function
        function to call when band is finished.
    """

    def __init__(self, fn):
        self.fn = fn

    def adjust_region(self, region_id, start, stop):
        pass

    def delete_region(self, region_id):
        self.fn()

    def finish_regions(self):
        self.fn()


class Fit1DVisualizer(interactive.PrimitiveVisualizer):
    """
    The generic class for interactive fitting of one or more 1D functions

    Attributes:
        reinit_panel: layout containing widgets that control the parameters
                      affecting the initialization of the (x,y) array(s)
        reinit_button: the button to reconstruct the (x,y) array(s)
        tabs: layout containing all the stuff required for an interactive 1D fit
        submit_button: the button signifying a successful end to the interactive session

        config: the Config object describing the parameters are their constraints
        widgets: a dict of (param_name, widget) elements that allow the properties
                 of the widgets to be set/accessed by the calling primitive. So far
                 I'm only including the widgets in the reinit_panel
        fits: list of InteractiveModel instances, one per (x,y) array
    """

    def __init__(self, data_source, fitting_parameters, config,
                 reinit_params=None, reinit_extras=None,
                 modal_message=None,
                 modal_button_label=None,
                 tab_name_fmt='{}',
                 xlabel='x', ylabel='y',
                 domains=None, title=None, primitive_name=None, filename_info=None,
                 template="fit1d.html", help_text=None, recalc_inputs_above=False,
                 **kwargs):
        """
        Parameters
        ----------
        data_source : array or function
            input data or the function to calculate the input data.  The input data
            should be [x, y] or [x, y, weights] or [[x, y], [x, y],.. or [[x, y, weights], [x, y, weights]..
            or, if a function, it accepts (config, extras) where extras is a dict of values based on reinit_extras
            and returns [[x, y], [x, y].. or [[x, y, weights], [x, y, weights], ...
        fitting_parameters : list of :class:`~geminidr.interactive.fit.fit1d.FittingParameters` or :class:`~geminidr.interactive.fit.fit1d.FittingParameters`
            Description of parameters to use for `fit_1d`
        config : Config instance describing primitive parameters and limitations
        reinit_params : list of str
            list of parameter names in config related to reinitializing fit arrays.  These cause the `data_source`
            function to be run to get the updated coordinates/weights.  Should not be passed if `data_source` is
            not a function.
        reinit_extras :
            Extra parameters to show on the left side that can affect the output of `data_source` but
            are not part of the primitive configuration.  Should not be passed if `data_source` is not a function.
        modal_message : str
            If set, datapoint calculation is expected to be expensive and a 'recalculate' button will be shown
            below the reinit inputs rather than doing it live.
        modal_button_label : str
            If set and if modal_message was set, this will be used for the label on the recalculate button.  It is
            not required.
        tab_name_fmt : str
            Format string for naming the tabs
        xlabel : str
            String label for X axis
        ylabel : str
            String label for Y axis
        domains : list
            List of domains for the inputs
        title : str
            Title for UI (Interactive <Title>)
        help_text : str
            HTML help text for popup help, or None to use the default
        """
        super().__init__(config=config, title=title, primitive_name=primitive_name, filename_info=filename_info,
                         template=template, help_text=help_text)
        self.layout = None
        self.recalc_inputs_above = recalc_inputs_above

        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # If we have a widget driving the modal dialog via it's enable/disable state,
        # store it in this so the recalc knows to re-enable the widget
        self.modal_widget = None

        # Make the panel with widgets to control the creation of (x, y) arrays

        if reinit_params is not None or reinit_extras is not None:
            # Create left panel
            reinit_widgets = self.make_widgets_from_config(reinit_params, reinit_extras, modal_message is None)

            # This should really go in the parent class, like submit_button
            if modal_message:
                if len(reinit_widgets) > 1:
                    self.reinit_button = bm.Button(label=modal_button_label if modal_button_label else "Reconstruct points")
                    self.reinit_button.on_click(self.reconstruct_points)
                    self.make_modal(self.reinit_button, modal_message)
                    reinit_widgets.append(self.reinit_button)
                    self.modal_widget = self.reinit_button
                else:
                    def kickoff_modal(attr, old, new):
                        self.reconstruct_points()
                    reinit_widgets[0].children[1].on_change('value', kickoff_modal)
                    self.make_modal(reinit_widgets[0], modal_message)
                    self.modal_widget = reinit_widgets[0]

            if recalc_inputs_above:
                self.reinit_panel = row(*reinit_widgets)
            else:
                self.reinit_panel = column(*reinit_widgets)
        else:
            # left panel with just the function selector (Chebyshev, etc.)
            self.reinit_panel = None  # column()

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
            if not (len(fitting_parameters) == len(allx) == len(ally)):
                raise ValueError("Different numbers of models and coordinates")
            self.nfits = len(fitting_parameters)
        else:
            if allx.size != ally.size:
                raise ValueError("Different (x, y) array sizes")
            self.nfits = 1

        self.reinit_extras = [] if reinit_extras is None else reinit_extras

        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})

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
                self.fits.append(tui.model)
        else:

            # ToDo: Review if there is a better way of handling this.
            if all_weights is None:
                all_weights = [None]

            # ToDo: the domains variable contains a list. I changed it to
            #  domains[0] and the code worked.
            tui = Fit1DPanel(self, fitting_parameters[0], domains[0], allx[0], ally[0], all_weights[0], **kwargs)
            tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(1))
            self.tabs.tabs.append(tab)
            self.fits.append(tui.model)

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

        self.submit_button.align = 'end'
        self.submit_button.height = 35
        self.submit_button.height_policy = "fixed"
        self.submit_button.margin = (0, 5, -30, 5)
        self.submit_button.width = 212
        self.submit_button.width_policy = "fixed"

        layout_ls = list()
        if self.filename_info:
            self.submit_button.align = 'end'
            layout_ls.append(row(Spacer(width=250),
                                 column(self.get_filename_div(), self.submit_button),
                                 Spacer(width=10),
                                 align="end", css_classes=['top-row']))
        else:
            layout_ls.append(self.submit_button,
                             align="end", css_classes=['top-row'])

        if self.reinit_panel is None:
            layout_ls.append(col)
        elif len(self.reinit_panel.children) <= 1 or self.recalc_inputs_above:
            layout_ls.append(row(self.reinit_panel))
            layout_ls.append(Spacer(height=10))
            layout_ls.append(col)
        else:
            layout_ls.append(row(self.reinit_panel, col))
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
        if self.modal_widget:
            self.modal_widget.disabled = True

        def fn():
            """Top-level code to update the Config with the values from the widgets"""
            config_update = {k: v.value for k, v in self.widgets.items()}
            for extra in self.reinit_extras:
                del config_update[extra]
            for k, v in config_update.items():
                print(f'{k} = {v}')
            self.config.update(**config_update)

        self.do_later(fn)

        if self.reconstruct_points_fn is not None:
            def rfn():
                all_coords = self.reconstruct_points_fn(self.config, self.extras)
                for fit, coords in zip(self.fits, all_coords):
                    if len(coords) > 2:
                        fit.weights = coords[2]
                    else:
                        fit.weights = None
                    fit.weights = fit.populate_bokeh_objects(coords[0], coords[1], fit.weights, mask=None)
                    fit.perform_fit()
                if self.modal_widget:
                    self.modal_widget.disabled = False

            self.do_later(rfn)

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


def prep_fit1d_params_for_fit1d(fit1d_params):
    """
    In the UI, which relies on `fit1d_params`, we constrain
    `niter` to 1 at the low end and separately have a `sigma`
    boolean checkbox.

    To support the `sigma` checkbox, here we remap the inputs
    based on the value of `niter`.  If `niter` is 0, this
    tells us no `sigma` rejection is desired.  So, in that
    case, we set `sigma` to False.  We then set `niter` to
    1 for the UI to work as desired.  Were `niter` passed
    in as `1` originally, it would remain `1` but `sigma`
    would be set to `True`.

    The UI will disable all the sigma related inputs when
    `sigma` is set to False.  It will also exclude them
    from the parameters sent to `fit_1d`.

    Parameters
    ----------
    :fit1d_params: dict
        Dictionary of parameters for the UI and the `fit_1d` fitter, modified in place
    """
    # If niter is 0, set sigma to None and niter to 1
    if 'niter' in fit1d_params and fit1d_params['niter'] == 0:
        # we use a min of 1 for niter, then remove the sigmas
        # to clue the UI in that we are not sigma clipping.
        # If we disable sigma clipping, niter will be disabled
        # in the UI.  Allowing a niter selection of 0 with
        # sigma clipping turned on is counterintuitive for
        # the user.
        fit1d_params['niter'] = 1
        fit1d_params['sigma'] = False
    else:
        fit1d_params['sigma'] = True
