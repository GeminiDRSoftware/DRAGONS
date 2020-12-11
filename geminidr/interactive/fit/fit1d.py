from abc import ABC, abstractmethod


import numpy as np

from bokeh import models as bm, transform as bt
from bokeh.layouts import row, column
from bokeh.models import Div, Select
from bokeh.plotting import figure
from bokeh.palettes import Category10

from geminidr.interactive import interactive
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIBandModel, GIApertureModel, connect_figure_extras, GIBandListener
from gempy.library.fitting import fit_1D


class FittingParameters(object):
    def __init__(self, *, function='chebyshev', order=None, axis=0, sigma_lower=3.0, sigma_upper=3.0,
                 niter=0, grow=False, regions=None):
        """
        Describe a set of parameters for running fits of data with.

        This makes it easy to bundle a set of parameters used to perform a fit.
        These are used with the `~gempy.library.fitting.fit_1D` fitter to do
        fits of the data.

        Parameters
        ----------
        function : str
            Name of the function to use for fitting (i.e. 'chebyshev')
        order : int
            Order of the fit
        axis : int
            Axis for data
        sigma_lower : float
            Lower sigma
        sigma_upper : float
            Upper sigma
        niter : int
            Number of iterations
        grow : int
            Grow window
        regions : list of tuple start/stop pairs
            This is a list of range start/stop pairs to pass down
        """
        self.function = function
        self.order = order
        self.axis = axis
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.niter = niter
        self.grow = grow
        self.regions = regions

    def build_fit_1D(self, data, weights):
        return fit_1D(data, weights=weights,
                      function=self.function,
                      order=self.order, axis=self.axis,
                      sigma_lower=self.sigma_lower,
                      sigma_upper=self.sigma_upper,
                      niter=self.niter,
                      # TODO grow is being rejected by fit_1D with the sigma_clip() in astropy.stats.sigma_clipping
                      # grow=self.grow,
                      regions=self.regions,
                      # plot=debug
                      )


class InteractiveModel(ABC):
    MASK_TYPE = ['band', 'user', 'good', 'fit']
    MARKERS = ['circle', 'circle', 'triangle', 'square']
    PALETTE = Category10[4]
    """
    Base class for all interactive models, containing:
        (a) the parameters of the model
        (b) the parameters that control the fitting (e.g., sigma-clipping)
        (c) the way the fitting is performed
        (d) the input and output coordinates, mask, and weights

    There will be 3 masks defined (all booleans):
        user_mask: points masked by the user
        fit_mask:  points rejected by the fit
        band_mask: points rejected by not being in a selection band (only if bands exist)
    """
    def __init__(self, model):
        self.model = model
        self.listeners = []
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

    def update_mask(self):
        """
        Update the internal mask on the data using the various boolean masks.

        This will consolidate the `~geminidr.interactive.fit.fit1d.InteractiveModel.band_mask`,
        `~geminidr.interactive.fit.fit1d.InteractiveModel.user_mask`, and
        `~geminidr.interactive.fit.fit1d.InteractiveModel.fit_mask` into a unified data mask.
        Order of preference is `user`, `band`, `fit`
        """
        # Update the "mask" column to change the glyphs
        new_mask = ['good'] * len(self.data.data['mask'])
        for i, (bdm, um, fm) in enumerate(zip(self.band_mask, self.user_mask, self.fit_mask)):
            if fm:
                new_mask[i] = 'fit'
            if bdm:
                new_mask[i] = 'band'
            if um:
                new_mask[i] = 'user'
        self.data.data['mask'] = new_mask


class InteractiveModel1D(InteractiveModel):
    """
    Subclass for 1D models
    """
    def __init__(self, fitting_parameters, domain, x=None, y=None, mask=None, var=None,
                 grow=0, sigma=3, lsigma=None, hsigma=None, maxiter=3,
                 section=None):
        """
        Create base class with given parameters as initial model inputs.

        Parameters
        ----------
        model : `astropy.modeling.models.Chebyshev1D` or ?
            Model behind the 1-D fit.  Chebyshev or Spline.  This gets wrapped
            in a helper model like :class:`~geminidr.interactive.fit1d.InteractiveChebyshev1D`
            or :class:`~geminidr.interactive.fit1d.InteractiveSpline1D`
        x : :class:`~numpy.ndarray`
            list of x coordinate values
        y : :class:`~numpy.ndarray`
            list of y coordinate values
        mask : array of str
            array of mask names for each point
        var
        grow
            grow for fit
        sigma
            sigma clip for fit
        lsigma
        hsigma
        maxiter
            max iterations to do on fit
        section
        """
        model = InteractiveNewFit1D(fitting_parameters, domain)
        super().__init__(model)
        self.section = section
        self.data = bm.ColumnDataSource({'x': [], 'y': [], 'mask': []})
        xlinspace = np.linspace(*self.domain, 100)
        self.populate_bokeh_objects(x, y, mask, var)

        # Our single slider isn't well set up for different low/hi sigma
        # We can worry about how we want to deal with that later
        if sigma:
            self.sigma = sigma
            self.sigma_clip = True
        else:
            self.sigma = 3
            self.sigma_clip = False
        self.lsigma = self.hsigma = self.sigma  # placeholder
        self.grow = grow
        self.maxiter = maxiter
        self.var = None

        model.perform_fit(self)
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
        self.model.set_function(fn)

    def populate_bokeh_objects(self, x, y, mask=None, var=None):
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
        var
        """
        if mask is None:
            try:  # Handle y as masked array
                if any(y.mask):
                    init_mask = y.mask
                else:
                    init_mask = np.zeros_like(x, dtype=bool)
                # init_mask = y.mask or np.zeros_like(x, dtype=bool)
            except AttributeError:
                init_mask = np.zeros_like(x, dtype=bool)
            else:
                y = y.data
        else:
            init_mask = mask
        self.var = var
        if var is not None:
            init_mask |= (self.var <= 0)

        x = x[~init_mask]
        y = y[~init_mask]

        self.fit_mask = np.zeros_like(x, dtype=bool)
        # "section" is the valid section provided by the user,
        # i.e., points not in this region(s) are user-masked
        if self.section is None:
            self.user_mask = np.zeros_like(self.fit_mask)
        else:
            self.user_mask = np.ones_like(self.fit_mask)
            for slice_ in self.section:
                self.user_mask[slice_.start < x < slice_.stop] = False

        self.band_mask = np.zeros_like(self.fit_mask)

        # Might put the variance in here for errorbars, but it's not needed
        # at the moment
        bokeh_data = {'x': x, 'y': y, 'mask': ['good'] * len(x)}
        for extra_column in ('residuals', 'ratio'):
            if extra_column in self.data.data:
                bokeh_data[extra_column] = np.zeros_like(y)
        self.data.data = bokeh_data
        self.update_mask()

    @property
    def x(self):
        """
        maps x attribute internally to bokeh structures

        Returns
        -------
        array of double : x coordinates
        """
        return self.data.data['x']

    @property
    def y(self):
        """
        maps y attribute internally to bokeh structures

        Returns
        -------
        array of double : y coordinates
        """
        return self.data.data['y']

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

    @property
    def order(self):
        """
        Maps the order attribute to the underlying model

        Returns
        -------
        int : order value from model
        """
        return self.model.order

    @order.setter
    def order(self, value):
        """
        Maps sets to the order attribute to the contained model

        Parameters
        ----------
        value : int
            new vlaue for order
        """
        self.model.order = value

    @property
    def domain(self):
        """
        Maps requests for the domain to the contained model

        Returns
        -------

        """
        return self.model.domain

    def perform_fit(self, *args):
        """
        Perform the fit.

        This performs the fit in the contained model, updates the mask,
        and recalculates the plots for residuals and ratio, if present.
        It then notifies all listeners that the data and model have
        changed so they can respond.
        """
        self.model.perform_fit(self)
        self.update_mask()
        if 'residuals' in self.data.data:
            self.data.data['residuals'] = self.y - self.evaluate(self.x)
        if 'ratio' in self.data.data:
            self.data.data['ratio'] = self.y / self.evaluate(self.x)
        self.notify_listeners()

    def evaluate(self, x):
        return self.model(x)


class InteractiveNewFit1D:
    def __init__(self, fitting_parameters, domain):
        """
        Create `~InteractiveNewFit1D` wrapper around the new fit1d model.

        The models don't like being modified, so this wrapper class handles
        that for us.  We can just keep a reference to this and, when needed,
        it will build a new `~fitting.fit_1D` instance and replace it's
        previous copy.

        Parameters
        ----------
        model : :class:`~fitting.fit_1D`
            :class:`~fitting.fit_1D` instance to wrap
        """
        assert isinstance(fitting_parameters, FittingParameters)
        self.fitting_parameters = fitting_parameters
        self.domain = domain
        self.fit = None

    def __call__(self, x):
        return self.fit.evaluate(x)

    def set_function(self, fn):
        self.fitting_parameters.function = fn

    @property
    def order(self):
        """
        Get the order of the fitter.

        Returns
        -------
        int : `order` of the model we are wrapping
        """
        return self.fitting_parameters.order

    @order.setter
    def order(self, order):
        """
        Set the order in this fitter.

        This sets the order, cleaning it if necessary into an `int`.  It also
        recreates the model in the same type as it currently is.

        Parameters
        ----------
        order : int
            order to use in the fit
        """
        self.fitting_parameters.order = int(order)  # fix if TextInput

    def perform_fit(self, parent):
        """
        Perform the fit, update self.model, parent.fit_mask

        The upper layer is a :class:`~InteractiveModel1D` that calls into
        this method. It passes itself down as `parent` to give access to
        various fields and allow this fit to be saved back up to it.

        Parameters
        ----------
        parent : :class:`~InteractiveModel1D`
            wrapper model passes itself when it calls into this method
        """
        # Note that band_mask is now handled by passing a region string to fit_1D
        # but we still use the band_mask for highlighting the affected points

        # TODO switch back if we use the region string...
        #goodpix = ~parent.user_mask
        goodpix = ~(parent.user_mask | parent.band_mask)
        if parent.var is None:
            weights = None
        else:
            weights = np.divide(1.0, parent.var, out=np.zeros_like(self.x),
                                where=parent.var > 0)[goodpix]

        if parent.sigma_clip:
            self.fit = self.fitting_parameters.build_fit_1D(parent.y[goodpix], weights=weights)
            parent.fit_mask = np.zeros_like(parent.x, dtype=bool)

            # Now pull in the sigma mask
            parent.fit_mask[goodpix] = self.fit.mask
        else:
            fitter = self.fitting_parameters.build_fit_1D(parent.y[goodpix], weights=weights)
            self.fit = fitter()
            parent.fit_mask = np.zeros_like(parent.x, dtype=bool)


class Fit1DPanel:
    def __init__(self, visualizer, fitting_parameters, domain, x, y, min_order=1, max_order=10, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True, grow_slider=True):
        """
        Panel for visualizing a 1-D fit, perhaps in a tab

        Parameters
        ----------
        visualizer : :class:`~geminidr.interactive.fit.Fit1DVisualizer`
            visualizer to associate with
        fitting_parameters : :class:`~geminidr.interactive.fit.FittingParameters`
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
        grow_slider : bool
            True if we want the slider for modifying growth radius
        """
        # Just to get the doc later
        self.visualizer = visualizer

        # Probably do something better here with factory function/class
        self.fitting_parameters = fitting_parameters
        self.fit = InteractiveModel1D(fitting_parameters, domain, x, y)

        fit = self.fit
        order_slider = interactive.build_text_slider("Order", fit.order, 1, min_order, max_order,
                                                     fit, "order", fit.perform_fit, throttled=True)
        sigma_upper_slider = interactive.build_text_slider("Sigma (Upper)", fitting_parameters.sigma_upper, 0.01, 1, 10,
                                                           fitting_parameters, "sigma_upper", self.sigma_slider_handler,
                                                           throttled=True)
        sigma_lower_slider = interactive.build_text_slider("Sigma (Lower)", fitting_parameters.sigma_lower, 0.01, 1, 10,
                                                           fitting_parameters, "sigma_lower", self.sigma_slider_handler,
                                                           throttled=True)
        sigma_button = bm.CheckboxGroup(labels=['Sigma clip'], active=[0] if self.fit.sigma_clip else [])
        sigma_button.on_change('active', self.sigma_button_handler)
        controls_column = [order_slider, row(sigma_upper_slider, sigma_button)]
        controls_column.append(sigma_lower_slider)
        # if grow_slider:
        #     controls_column.append(interactive.build_text_slider("Growth radius",
        #                                                          fitting_parameters.grow, 1, 0, 10,
        #                                                          fitting_parameters, "grow",
        #                                                          fit.perform_fit))
        controls_column.append(interactive.build_text_slider("Max iterations", fitting_parameters.niter,
                                                             1, 0, 10,
                                                             fitting_parameters, "niter",
                                                             fit.perform_fit))

        mask_button = bm.Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = bm.Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        controller_div = Div()

        controls = column(*controls_column,
                          row(mask_button, unmask_button), controller_div)

        # Now the figures
        p_main = figure(plot_width=plot_width, plot_height=plot_height,
                        title='Fit', x_axis_label=xlabel, y_axis_label=ylabel,
                        tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                        output_backend="webgl", x_range=None, y_range=None)
        p_main.height_policy = 'fixed'
        p_main.width_policy = 'fit'
        self.band_model = GIBandModel()

        class Fit1DBandListener(GIBandListener):
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

            def adjust_band(self, band_id, start, stop):
                pass

            def delete_band(self, band_id):
                self.fn()

            def finish_bands(self):
                self.fn()

        self.band_model.add_listener(Fit1DBandListener(self.band_model_handler))
        connect_figure_extras(p_main, None, self.band_model)
        Controller(p_main, None, self.band_model, controller_div)
        fig_column = [p_main]

        if plot_residuals:
            p_resid = figure(plot_width=plot_width, plot_height=plot_height // 2,
                             title='Fit Residuals',
                             x_axis_label=xlabel, y_axis_label='delta'+ylabel,
                             tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                             output_backend="webgl", x_range=None, y_range=None)
            p_resid.height_policy = 'fixed'
            p_resid.width_policy = 'fit'
            connect_figure_extras(p_resid, None, self.band_model)
            fig_column.append(p_resid)
            # Initalizing this will cause the residuals to be calculated
            self.fit.data.data['residuals'] = np.zeros_like(self.fit.x)
            p_resid.scatter(x='x', y='residuals', source=self.fit.data,
                            size=5, **self.fit.mask_rendering_kwargs())

        self.scatter = p_main.scatter(x='x', y='y', source=self.fit.data,
                                      size=5, **self.fit.mask_rendering_kwargs())
        self.fit.add_listener(self.model_change_handler)
        self.fit.perform_fit()
        self.line = p_main.line(x='xlinspace', y='model', source=self.fit.evaluation, line_width=1, color='red')

        col = column(*fig_column)
        col.sizing_mode = 'scale_width'
        self.component = row(controls, col)

    def model_change_handler(self):
        """
        If the `~fit` changes, this gets called to evaluate the fit and save the results.
        """
        self.fit.evaluation.data['model'] = self.fit.evaluate(self.fit.evaluation.data['xlinspace'])

    def sigma_slider_handler(self, val):
        """
        Handle the sigma clipping being adjusted.

        This will trigger a fit since the result may
        change.
        """
        # If we're not sigma-clipping, we don't need to refit the model if sigma changes
        if self.fit.sigma_clip:
            self.fit.perform_fit()

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
        self.fit.perform_fit()

    def mask_button_handler(self, stuff):
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
        indices = self.fit.data.selected.indices
        self.fit.data.selected.update(indices=[])
        for i in indices:
            self.fit.user_mask[i] = 1
        self.fit.perform_fit()

    def unmask_button_handler(self, stuff):
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
        indices = self.fit.data.selected.indices
        self.fit.data.selected.update(indices=[])
        for i in indices:
            self.fit.user_mask[i] = 0
        self.fit.perform_fit()

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
        x_data = self.fit.data.data['x']
        for i in np.arange(len(x_data)):
            if self.band_model.contains(x_data[i]):
                self.fit.band_mask[i] = 0
            else:
                self.fit.band_mask[i] = 1
        # Band operations can come in through the keypress URL
        # so we defer the fit back onto the Bokeh IO event loop

        # TODO figure out if we are using this or band_mask
        # self.fitting_parameters.regions = self.band_model.build_regions()
        self.visualizer.do_later(self.fit.perform_fit)


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

    def __init__(self, allx, ally, fitting_parameters, config, log=None,
                 reinit_params=None, reinit_extras=None, reinit_live=False,
                 order_param=None,
                 min_order=1, max_order=10, tab_name_fmt='{}',
                 xlabel='x', ylabel='y', reconstruct_points=None,
                 domains=None, **kwargs):
        """
        Parameters
        ----------
        allx: 1D array/list of N 1D arrays of "x" coordinates
        ally: 1D array/list of N 1D arrays of "y" coordinates
        models: Model/list of N Model instances
        config: Config instance describing parameters and limitations
        reinit_params: list of parameters related to reinitializing fit arrays
        """
        super().__init__(log=log, config=config)

        self.reconstruct_points_fn = reconstruct_points


        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # Some sanity checks now
        if isinstance(fitting_parameters, list):
            if not(len(fitting_parameters) == len(allx) == len(ally)):
                raise ValueError("Different numbers of models and coordinates")
            for x, y in zip(allx, ally):
                if len(x) != len(y):
                    raise ValueError("Different (x, y) array sizes")
            self.nfits = len(fitting_parameters)
            fn = fitting_parameters[0].function
        else:
            if allx.size != ally.size:
                raise ValueError("Different (x, y) array sizes")
            self.nfits = 1
            fn = fitting_parameters.function

        # Make the panel with widgets to control the creation of (x, y) arrays
        # Dropdown for selecting fit_1D function
        self.function = Select(title="Option:", value=fn,
                               options=['chebyshev', 'legendre', 'polynomial', 'spline1',
                                        'spline2', 'spline3', 'spline4', 'spline5'])

        def fn_select_change(attr, old, new):
            def refit():
                for fit in self.fits:
                    fit.set_function(new)
                    fit.perform_fit()
            self.do_later(refit)
        self.function.on_change('value', fn_select_change)

        if reinit_params is not None or reinit_extras is not None:
            # Create left panel
            reinit_widgets = self.make_widgets_from_config(reinit_params, reinit_extras, reinit_live)

            # This should really go in the parent class, like submit_button
            if not reinit_live:
                self.reinit_button = bm.Button(label="Reconstruct points")
                self.reinit_button.on_click(self.reconstruct_points)
                self.make_modal(self.reinit_button, "<b>Recalculating Points</b><br/>This may take 20 seconds")
                reinit_widgets.append(self.reinit_button)

            self.reinit_panel = column(self.function, *reinit_widgets)
        else:
            # left panel with just the function selector (Chebyshev, etc.)
            self.reinit_panel = column(self.function)

        self.reinit_extras = [] if reinit_extras is None else reinit_extras

        field = self.config._fields[order_param]
        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})
        if hasattr(field, 'min') and field.min:
            kwargs['min_order'] = field.min
        else:
            kwargs['min_order'] = 1
        if hasattr(field, 'max') and field.max:
            kwargs['max_order'] = field.max
        else:
            kwargs['max_order'] = field.default * 2
        self.tabs = bm.Tabs(tabs=[], name="tabs")
        self.tabs.sizing_mode = 'scale_width'
        self.fits = []
        if self.nfits > 1:
            if domains is None:
                domains = [None] * len(fitting_parameters)
            for i, (fitting_parms, domain, x, y) in enumerate(zip(fitting_parameters, domains, allx, ally), start=1):
                tui = Fit1DPanel(self, fitting_parms, domain, x, y, **kwargs)
                tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(i))
                self.tabs.tabs.append(tab)
                self.fits.append(tui.fit)
        else:
            tui = Fit1DPanel(fitting_parameters, allx, ally, **kwargs)
            tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(1))
            self.tabs.tabs.append(tab)
            self.fits.append(tui.fit)

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
        col = column(self.tabs,)
        col.sizing_mode = 'scale_width'
        layout = column(row(self.reinit_panel, col), self.submit_button)
        doc.add_root(layout)

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
        if hasattr(self, 'reinit_button'):
            self.reinit_button.disabled = True

        def fn():
            """Top-level code to update the Config with the values from the widgets"""
            config_update = {k: v.value for k, v in self.widgets.items()}
            for extra in self.reinit_extras:
                del config_update[extra[0]]
            for k, v in config_update.items():
                print(f'{k} = {v}')
            self.config.update(**config_update)
        self.do_later(fn)

        if self.reconstruct_points_fn is not None:
            def rfn():
                all_coords = self.reconstruct_points_fn(self.config, self.extras)
                for fit, coords in zip(self.fits, all_coords):
                    fit.populate_bokeh_objects(coords[0], coords[1], mask=None)
                    fit.perform_fit()
                if hasattr(self, 'reinit_button'):
                    self.reinit_button.disabled = False
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
        return [fit.model.fit for fit in self.fits]