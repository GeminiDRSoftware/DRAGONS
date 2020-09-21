from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from bokeh import models as bm, transform as bt
from bokeh.layouts import row, column
from bokeh.models import Div
from bokeh.plotting import figure
from bokeh.palettes import Category10

from geminidr.interactive import interactive
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import GIBandModel, GIApertureModel, connect_figure_extras, GIBandListener
from gempy.library import tracing, astrotools as at


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
    def __init__(self, model, x=None, y=None, mask=None, var=None,
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
        if isinstance(model, models.Chebyshev1D):
            model = InteractiveChebyshev1D(model)
        else:
            model = InteractiveSpline1D(model)
        super().__init__(model)
        self.section = section
        self.data = bm.ColumnDataSource({'x': [], 'y': [], 'mask': []})
        xlinspace = np.linspace(*self.domain, 100)
        self.evaluation = bm.ColumnDataSource({'xlinspace': xlinspace,
                                               'model': self.evaluate(xlinspace)})
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
                init_mask = y.mask or np.zeros_like(x, dtype=bool)
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

    def perform_fit(self):
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


class InteractiveChebyshev1D:
    def __init__(self, model):
        assert isinstance(model, models.Chebyshev1D)
        self.model = model

    def __call__(self, x):
        return self.model(x)

    @property
    def order(self):
        return self.model.degree

    @order.setter
    def order(self, order):
        degree = int(order)  # because of TextInput issues
        new_model = self.model.__class__(degree=degree, domain=self.model.domain)
        for i in range(degree + 1):
            setattr(new_model, f'c{i}', getattr(self.model, f'c{i}', 0))
        self.model = new_model

    @property
    def domain(self):
        # Defined explicitly here since a spline doesn't have a domain
        return self.model.domain

    def perform_fit(self, parent):
        """
        Perform the fit, update self.model, parent.fit_mask
        """
        goodpix = ~(parent.user_mask | parent.band_mask)
        if parent.var is None:
            weights = None
        else:
            weights = np.divide(1.0, parent.var, out=np.zeros_like(self.x),
                                where=parent.var > 0)[goodpix]

        if parent.sigma_clip:
            fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                       sigma_clip, sigma=parent.sigma,
                                                       niter=parent.maxiter)
            try:
                new_model, mask = fit_it(self.model, parent.x[goodpix], parent.y[goodpix], weights=weights)
            except (IndexError, np.linalg.linalg.LinAlgError):
                print("Problem with fitting")
            else:
                ngood = np.sum(~mask)
                if ngood > self.order:
                    rms = np.sqrt(np.sum((new_model(parent.x[goodpix][~mask]) - parent.y[goodpix][~mask])**2) / ngood)
                    # I'm not sure how to report the rms
                    self.model = new_model
                    parent.fit_mask = np.zeros_like(parent.x, dtype=bool)
                    parent.fit_mask[goodpix] = mask
                else:
                    print("Too few remaining points to constrain the fit")
        else:
            fit_it = fitting.LinearLSQFitter()
            try:
                new_model = fit_it(self.model, parent.x[goodpix], parent.y[goodpix], weights=weights)
            except (IndexError, np.linalg.linalg.LinAlgError):
                print("Problem with fitting")
            else:
                self.model = new_model
                parent.fit_mask = np.zeros_like(parent.x, dtype=bool)


class InteractiveSpline1D:
    pass
# ----------------------------------------------------------------------------


class Fit1DPanel:
    def __init__(self, visualizer, model, x, y, min_order=1, max_order=10, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True, grow_slider=True):
        """
        Panel for visualizing a 1-D fit, perhaps in a tab

        Parameters
        ----------
        visualizer : :class:`~geminidr.interactive.fit.Fit1DVisualizer`
            visualizer to associate with
        model : :class:`~geminidr.interactive.fit.Fit1DModel`
            model for this UI to present
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
        self.fit = InteractiveModel1D(model, x, y)

        fit = self.fit
        order_slider = interactive.build_text_slider("Order", fit.order, 1, min_order, max_order,
                                                     fit, "order", fit.perform_fit)
        sigma_slider = interactive.build_text_slider("Sigma", fit.sigma, 0.01, 1, 10,
                                                     fit, "sigma", self.sigma_slider_handler)
        sigma_button = bm.CheckboxGroup(labels=['Sigma clip'], active=[0] if self.fit.sigma_clip else [])
        sigma_button.on_change('active', self.sigma_button_handler)
        controls_column = [order_slider, row(sigma_slider, sigma_button)]
        if grow_slider:
            controls_column.append(interactive.build_text_slider("Growth radius", fit.grow, 1, 0, 10,
                                                        fit, "grow", fit.perform_fit))
        controls_column.append(interactive.build_text_slider("Max iterations", fit.maxiter, 1, 0, 10,
                                                    fit, "maxiter", fit.perform_fit))

        # Only need this if there are multiple tabs
        apply_button = bm.Button(label="Apply model universally")
        apply_button.on_click(self.apply_button_handler)

        mask_button = bm.Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = bm.Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        controller_div = Div()

        controls = column(*controls_column,
                          apply_button, row(mask_button, unmask_button), controller_div)

        # Now the figures
        p_main = figure(plot_width=plot_width, plot_height=plot_height,
                        title='Fit', x_axis_label=xlabel, y_axis_label=ylabel,
                        tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                        output_backend="webgl", x_range=None, y_range=None)
        aperture_model = GIApertureModel()
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
                pass

            def finish_bands(self):
                self.fn()

        self.band_model.add_listener(Fit1DBandListener(self.band_model_handler))
        connect_figure_extras(p_main, aperture_model, self.band_model)
        Controller(p_main, aperture_model, self.band_model, controller_div)
        fig_column = [p_main]

        if plot_residuals:
            p_resid = figure(plot_width=plot_width, plot_height=plot_height // 2,
                             title='Fit Residuals',
                             x_axis_label=xlabel, y_axis_label='delta'+ylabel,
                             tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                             output_backend="webgl", x_range=None, y_range=None)
            connect_figure_extras(p_resid, aperture_model, self.band_model)
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

        self.component = row(controls, column(*fig_column))

    def model_change_handler(self):
        self.fit.evaluation.data['model'] = self.fit.evaluate(self.fit.evaluation.data['xlinspace'])

    def sigma_slider_handler(self):
        # If we're not sigma-clipping, we don't need to refit the model if sigma changes
        if self.fit.sigma_clip:
            self.fit.perform_fit()

    def sigma_button_handler(self, attr, old, new):
        self.fit.sigma_clip = bool(new)
        self.fit.perform_fit()

    def apply_button_handler(self, stuff):
        return

    def mask_button_handler(self, stuff):
        indices = self.fit.data.selected.indices
        self.fit.data.selected.update(indices=[])
        for i in indices:
            self.fit.user_mask[i] = 1
        self.fit.perform_fit()

    def unmask_button_handler(self, stuff):
        indices = self.fit.data.selected.indices
        self.fit.data.selected.update(indices=[])
        for i in indices:
            self.fit.user_mask[i] = 0
        self.fit.perform_fit()

    def band_model_handler(self):
        x_data = self.fit.data.data['x']
        for i in np.arange(len(x_data)):
            if self.band_model.contains(x_data[i]):
                self.fit.band_mask[i] = 0
            else:
                self.fit.band_mask[i] = 1
        # Band operations can come in through the keypress URL
        # so we defer the fit back onto the Bokeh IO event loop
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

    def __init__(self, allx, ally, models, config, log=None,
                 reinit_params=None, order_param=None,
                 min_order=1, max_order=10, tab_name_fmt='{}',
                 xlabel='x', ylabel='y', **kwargs):
        """
        Parameters
        ----------
        allx: 1D array/list of N 1D arrays of "x" coordinates
        ally: 1D array/list of N 1D arrays of "y" coordinates
        models: Model/list of N Model instances
        config: Config instance describing parameters and limitations
        reinit_params: list of parameters related to reinitializing fit arrays
        """
        super().__init__(log=log)

        self.config = copy(config)
        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # Some sanity checks now
        if isinstance(models, list):
            if not(len(models) == len(allx) == len(ally)):
                raise ValueError("Different numbers of models and coordinates")
            for x, y in zip(allx, ally):
                if len(x) != len(y):
                    raise ValueError("Different (x, y) array sizes")
            self.nmodels = len(models)
        else:
            if allx.size != ally.size:
                raise ValueError("Different (x, y) array sizes")
            self.nmodels = 1

        # Make the panel with widgets to control the creation of (x, y) arrays
        if reinit_params is not None:
            # Create left panel
            reinit_widgets = self.make_widgets_from_config(reinit_params)

            # This should really go in the parent class, like submit_button
            self.reinit_button = bm.Button(label="Reconstruct points")
            self.reinit_button.on_click(self.reconstruct_points)
            self.make_modal(self.reinit_button, "<b>Recalculating Points</b><br/>This may take 20 seconds")
            reinit_widgets.append(self.reinit_button)

            self.reinit_panel = column(*reinit_widgets)
        else:
            self.reinit_panel = None

        field = self.config._fields[order_param]
        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})
        if field.min:
            kwargs['min_order'] = field.min
        if field.max:
            kwargs['max_order'] = field.max
        self.tabs = bm.Tabs(tabs=[], name="tabs")
        self.fits = []
        if self.nmodels > 1:
            for i, (model, x, y) in enumerate(zip(models, allx, ally), start=1):
                tui = Fit1DPanel(self, model, x, y, **kwargs)
                tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(i))
                self.tabs.tabs.append(tab)
                self.fits.append(tui.fit)
        else:
            tui = Fit1DPanel(models, allx, ally, **kwargs)
            tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(1))
            self.tabs.tabs.append(tab)
            self.fits.append(tui.fit)

    def visualize(self, doc):
        super().visualize(doc)
        layout = row(self.reinit_panel, column(self.tabs, self.submit_button))
        doc.add_root(layout)

    def reconstruct_points(self):
        self.reinit_button.disabled = True

        def fn():
            """Top-level code to update the Config with the values from the widgets"""
            config_update = {k: v.value for k, v in self.widgets.items()}
            for k, v in config_update.items():
                print(f'{k} = {v}')
            self.config.update(**config_update)
        self.do_later(fn)


class TraceApertures1DVisualizer(Fit1DVisualizer):
    def __init__(self, allx, ally, models, config, ext=None, locations=None,
                 **kwargs):
        self.ext = ext
        self.locations = locations
        super().__init__(allx, ally, models, config, **kwargs)

    def reconstruct_points(self):
        # super() to update the Config with the widget values
        # In this primitive, init_mask is always empty
        super().reconstruct_points()

        def fn():
            all_coords = trace_apertures_reconstruct_points(self.ext, self.locations, self.config)
            for fit, coords in zip(self.fits, all_coords):
                fit.populate_bokeh_objects(coords[0], coords[1], mask=None)
                fit.perform_fit()
            self.reinit_button.disabled = False
        self.do_later(fn)


def trace_apertures_reconstruct_points(ext, locations, config):
    dispaxis = 2 - ext.dispersion_axis()
    all_coords = []
    for loc in locations:
        c0 = int(loc + 0.5)
        spectrum = ext.data[c0] if dispaxis == 1 else ext.data[:, c0]
        start = np.argmax(at.boxcar(spectrum, size=3))

        # The coordinates are always returned as (x-coords, y-coords)
        ref_coords, in_coords = tracing.trace_lines(ext, axis=dispaxis,
                                                    start=start, initial=[loc],
                                                    rwidth=None, cwidth=5, step=config.step,
                                                    nsum=config.nsum, max_missed=config.max_missed,
                                                    initial_tolerance=None,
                                                    max_shift=config.max_shift)
        # Store as spectral coordinate first (i.e., x in the y(x) fit)
        if dispaxis == 0:
            all_coords.append(in_coords[::-1])
        else:
            all_coords.append(in_coords)
    return all_coords