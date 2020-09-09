from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from bokeh import models as bm, transform as bt
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.palettes import Category10

from geminidr.interactive import interactive, server
from gempy.library import tracing, astrotools as at

# ------------------ class/subclasses to store the models -----------------
class InteractiveModel(ABC):
    MASK_TYPE = ['user', 'good', 'fit']
    MARKERS = ['circle', 'triangle', 'plus']
    PALETTE = Category10[3]
    """
    Base class for all interactive models, containing:
        (a) the parameters of the model
        (b) the parameters that control the fitting (e.g., sigma-clipping)
        (c) the way the fitting is performed
        (d) the input and output coordinates, mask, and weights

    There will be 2 masks defined (all booleans):
        user_mask: points masked by the user
        fit_mask:  points rejected by the fit
    """
    def __init__(self, model):
        self.model = model
        self.listeners = []

    def add_listener(self, listener):
        if not callable(listener):
            raise ValueError("Listeners must be callables")
        self.listeners.append(listener)

    def notify_listeners(self):
        for listener in self.listeners:
            listener()

    @abstractmethod
    def perform_fit(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        pass

#    @property
#    def view(self):
#        # Return a CDSView of what we're interested in
#        return bm.CDSView(source=self.data,
#                          filters=[bm.BooleanFilter(~self.init_mask)])

    def mask_rendering_kwargs(self):
        return {'marker': bt.factor_mark('mask', self.MARKERS, self.MASK_TYPE),
                'color': bt.factor_cmap('mask', self.PALETTE, self.MASK_TYPE)}

    def update_mask(self):
        # Update the "mask" column to change the glyphs
        new_mask = ['good'] * len(self.data.data['mask'])
        for i, (um, fm) in enumerate(zip(self.user_mask, self.fit_mask)):
            if fm:
                new_mask[i] = 'fit'
            if um:
                new_mask[i] = 'user'
        self.data.data['mask'] = new_mask

class InteractiveModel1D(InteractiveModel):
    """Subclass for 1D models"""
    def __init__(self, model, x=None, y=None, mask=None, var=None,
                 grow=0, sigma=3, lsigma=None, hsigma=None, maxiter=3,
                 section=None):
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

    def populate_bokeh_objects(self, x, y, mask=None, var=None):
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
                self.user_mask[slice_.start < x < slice._stop] = False

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
        return self.data.data['x']

    @property
    def y(self):
        return self.data.data['y']

    @property
    def sigma(self):
        if self.lsigma == self.hsigma:
            return self.lsigma
        return 0.5 * (self.lsigma + self.hsigma)  # do something better?

    @sigma.setter
    def sigma(self, value):
        self.lsigma = self.hsigma = float(value)

    @property
    def order(self):
        return self.model.order

    @order.setter
    def order(self, value):
        self.model.order = value

    @property
    def domain(self):
        return self.model.domain

    def perform_fit(self):
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
        goodpix = ~parent.user_mask
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
    def __init__(self, model, x, y, min_order=1, max_order=10, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True):
        """A class that handles a 1d model and its visualization (maybe in a tab)"""

        # Probably do something better here with factory function/class
        self.fit = InteractiveModel1D(model, x, y)

        fit = self.fit
        order_slider = interactive.build_text_slider("Order", fit.order, 1, min_order, max_order,
                                                     fit, "order", fit.perform_fit)
        sigma_slider = interactive.build_text_slider("Sigma", fit.sigma, 0.01, 1, 10,
                                                     fit, "sigma", self.sigma_slider_handler)
        sigma_button = bm.CheckboxGroup(labels=['Sigma clip'], active=[0] if self.fit.sigma_clip else [])
        sigma_button.on_change('active', self.sigma_button_handler)
        grow_slider = interactive.build_text_slider("Growth radius", fit.grow, 1, 0, 10,
                                                    fit, "grow", fit.perform_fit)
        iter_slider = interactive.build_text_slider("Max iterations", fit.maxiter, 1, 0, 10,
                                                    fit, "maxiter", fit.perform_fit)

        # Only need this if there are multiple tabs
        apply_button = bm.Button(label="Apply model universally")
        apply_button.on_click(self.apply_button_handler)

        mask_button = bm.Button(label="Mask")
        mask_button.on_click(self.mask_button_handler)

        unmask_button = bm.Button(label="Unmask")
        unmask_button.on_click(self.unmask_button_handler)

        controls = column(order_slider, row(sigma_slider, sigma_button), grow_slider, iter_slider,
                          apply_button, row(mask_button, unmask_button))

        # Now the figures
        p_main = figure(plot_width=plot_width, plot_height=plot_height,
                            title='Fit', x_axis_label=xlabel, y_axis_label=ylabel,
                            tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                            output_backend="webgl", x_range=None, y_range=None)
        fig_column = [p_main]

        if plot_residuals:
            p_resid = figure(plot_width=plot_width, plot_height=plot_height // 2,
                                 title='Fit Residuals',
                                 x_axis_label=xlabel, y_axis_label='delta'+ylabel,
                                 tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap",
                                 output_backend="webgl", x_range=None, y_range=None)
            fig_column.append(p_resid)
            # Initalizing this will cause the residuals to be calculated
            self.fit.data.data['residuals'] = np.zeros_like(self.fit.x)
            p_resid.scatter(x='x', y='residuals', source=self.fit.data, #view=self.fit.view,
                                          size=5, **self.fit.mask_rendering_kwargs())

        self.scatter = p_main.scatter(x='x', y='y', source=self.fit.data, #view=self.fit.view,
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
        return
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.addmask(indices)

    def unmask_button_handler(self, stuff):
        return
        indices = self.scatter.source.selected.indices
        self.scatter.clear_selection()
        self.model.coords.unmask(indices)






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
                 xlabel='x', ylabel='y'):
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
        kwargs = {'xlabel': xlabel, 'ylabel': ylabel}
        if field.min:
            kwargs['min_order'] = field.min
        if field.max:
            kwargs['max_order'] = field.max
        self.tabs = bm.Tabs(tabs=[], name="tabs")
        self.fits = []
        if self.nmodels > 1:
            for i, (model, x, y) in enumerate(zip(models, allx, ally), start=1):
                tui = Fit1DPanel(model, x, y, **kwargs)
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
        """Top-level code to update the Config with the values from the widgets"""
        config_update = {k: v.value for k, v in self.widgets.items()}
        for k, v in config_update.items():
            print(f'{k} = {v}')
        self.config.update(**config_update)


def interactive_fitter(visualizer):
    server.set_visualizer(visualizer)
    server.start_server()
    server.set_visualizer(None)
    return visualizer.user_satisfied


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
        all_coords = trace_apertures_reconstruct_points(self.ext, self.locations, self.config)
        for fit, coords in zip(self.fits, all_coords):
            fit.populate_bokeh_objects(coords[0], coords[1], mask=None)
            fit.perform_fit()
        self.reinit_button.disabled = False

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