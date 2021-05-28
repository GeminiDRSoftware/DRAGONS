import numpy as np

from bokeh import models as bm
from bokeh.layouts import row, column
from bokeh.plotting import figure

from geminidr.interactive.controls import Controller
from gempy.library.matching import match_sources

from .fit1d import (Fit1DPanel, Fit1DVisualizer, fit1d_figure,
                    USER_MASK_NAME)


class WavelengthSolutionPanel(Fit1DPanel):
    def __init__(self, visualizer, fitting_parameters, domain, x, y,
                 weights=None, other_data=None, **kwargs):
        # No need to compute wavelengths here as the model_change_handler() does it
        self.spectrum = bm.ColumnDataSource({'wavelengths': np.zeros_like(other_data["spectrum"]),
                                             'spectrum': other_data["spectrum"]})
        super().__init__(visualizer, fitting_parameters, domain, x, y,
                         weights=weights, **kwargs)
        self.model.other_data = other_data

    def build_figures(self, domain=None, controller_div=None,
                      plot_residuals=True, plot_ratios=True):

        self.xpoint = 'fitted'
        self.ypoint = 'nonlinear'
        p_main, p_supp = fit1d_figure(width=self.width, height=self.height,
                                      xpoint='fitted', ypoint='nonlinear',
                                      xline='model', yline='nonlinear',
                                      xlabel=self.xlabel, ylabel=self.ylabel,
                                      model=self.model, plot_ratios=False,
                                      enable_user_masking=True)

        mask_handlers = (self.mask_button_handler, self.unmask_button_handler)
        Controller(p_main, None, self.model.band_model, controller_div,
                   mask_handlers=mask_handlers, domain=domain)

        p_spectrum = figure(plot_width=self.width, plot_height=self.height,
                            min_width=400, title='Spectrum',
                            x_axis_label=self.xlabel, y_axis_label="Signal",
                            tools = "pan,wheel_zoom,box_zoom,reset",
                            output_backend="webgl",
                            x_range=p_main.x_range, y_range=None,
                            min_border_left=80)
        p_spectrum.height_policy = 'fixed'
        p_spectrum.width_policy = 'fit'
        p_spectrum.sizing_mode = 'stretch_width'
        p_spectrum.step(x='wavelengths', y='spectrum', source=self.spectrum,
                        line_width=1, color="blue", mode="center")
        p_spectrum.text(x='fitted', y='heights', text='lines',
                        source=self.model.data, angle=0.5 * np.pi,
                        text_color=self.model.mask_rendering_kwargs()['color'],
                        text_baseline='middle')
        spectrum_plot = p_spectrum

        add_new_line_button = bm.Button(label="Add new line")
        self.line_chooser = row(bm.Div(text="New line wavelength"),
                                add_new_line_button)

        identify_button = bm.Button(label="Identify lines")
        identify_button.on_click(self.identify_lines)

        identify_panel = row(self.line_chooser, identify_button)

        return [spectrum_plot, identify_panel, p_main, p_supp]

    @property
    def linear_model(self):
        """Return only the linear part of a model. It doesn't work for
        splines, which is why it's not in the InteractiveModel1D class"""
        model = self.model.fit._models
        return model.__class__(degree=1, c0=model.c0, c1=model.c1,
                               domain=model.domain)

    # I could put the extra stuff in a second listener but the name of this
    # is generic, so let's just super() it and then do the extra stuff
    def model_change_handler(self, fit):
        """
        If the `~fit` changes, this gets called to evaluate the fit and save the results.
        """
        super().model_change_handler(fit)
        x, y = self.model.x, self.model.y
        linear_model = self.linear_model

        self.model.data.data['fitted'] = self.model.evaluate(x)
        self.model.data.data['nonlinear'] = y - linear_model(x)
        self.model.data.data['heights'] = [self.spectrum.data['spectrum'][int(xx + 0.5)] + 0.02 * self.spectrum.data['spectrum'].max() for xx in x]
        self.model.data.data['lines'] = [str(np.round(yy, decimals=6)) for yy in y]

        self.model.evaluation.data['nonlinear'] = self.model.evaluation.data['model'] - linear_model(self.model.evaluation.data['xlinspace'])

        domain = self.model.domain
        self.spectrum.data['wavelengths'] = self.model.evaluate(
            np.arange(domain[0], domain[1]+1))

    def add_identified_line(self, peak, wavelength):
        """
        Add a new line to the ColumnDataSource and performs a new fit

        Parameters
        ----------
        peak : float
            pixel locations of peak
        wavelength : float
            wavelengths
        """
        new_data = {'x': [peak], 'y': [wavelength], 'mask': ['good'],
                    'fitted': [0], 'nonlinear': [0], 'heights': [0],
                    'residuals': [0],
                    'lines': [str(np.round(wavelength, decimals=6))],
                   }
        self.model.data.stream(new_data)
        self.model.perform_fit()

    def identify_lines(self):
        """
        Called when the user clicks the "Identify Lines" button. This:
        1) Removes any masked points (user-masked or sigma-clipped) from the fit data
        2) Gets all the already-identified peaks that aren't in the fit
        3) Calculates the wavelengths of these peaks, based on the fit
        4) Matches those to unmatched lines in the linelist based on some criteria
        5) Adds new matches to the list
        6) Performs a new fit, triggering a plot update
        """
        print("IDENTIFY LINES")
        dw = self.linear_model.c1
        matching_distance = abs(self.model.other_data["fwidth"] * dw)
        all_lines = self.model.other_data["linelist"].wavelengths(
            in_vacuo=self.visualizer.config.in_vacuo, units="nm")

        good_data = {}
        for k, v in self.model.data.data.items():
            good_data[k] = [vv for vv, mask in zip(v, self.model.data.data['mask'])
                            if mask == 'good']

        matches = match_sources(all_lines, good_data['y'], radius=0.01 * abs(dw))
        unmatched_lines = [l for l, m in zip(all_lines, matches) if m == -1]

        new_peaks = np.setdiff1d(self.model.other_data["peaks"],
                                 good_data['x'], assume_unique=True)
        new_waves = self.model.evaluate(new_peaks)

        matches = match_sources(new_waves, unmatched_lines, radius=matching_distance)
        for peak, m in zip(new_peaks, matches):
            if m != -1:
                good_data['x'].append(peak)
                good_data['y'].append(unmatched_lines[m])
                good_data['mask'].append('good')
                good_data['fitted'].append(0)
                good_data['nonlinear'].append(0)
                good_data['heights'].append(0)
                good_data['residuals'].append(0)
                good_data['lines'].append(str(np.round(unmatched_lines[m], decimals=6)))
                print("NEW LINE", peak, unmatched_lines[m])

        self.model.data.data = good_data
        self.model.perform_fit()


class WavelengthSolutionVisualizer(Fit1DVisualizer):
    """
    A Visualizer specific to determineWavelengthSolution

    This differs from the parent class in the following ways:
    1) __init__()
        (a) each tab is a WavelengthSolutionPanel, not a Fit1DPanel
        (b) the data_source returns a fourth column, a dict containing
            additional information, that gets put in an "other_data" attribute
            and its "spectrum" element is passed to the InteractiveModel1D
            object in the Panel

    2) reconstruct_points()
        (a) this has to deal with the same issue as 1(b)

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
        super(Fit1DVisualizer, self).__init__(
            config=config, title=title, primitive_name=primitive_name,
            filename_info=filename_info, template=template, help_text=help_text)
        self.layout = None
        self.recalc_inputs_above = recalc_inputs_above

        other_data = []
        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # Make the panel with widgets to control the creation of (x, y) arrays

        if reinit_params is not None or reinit_extras is not None:
            # Create left panel
            reinit_widgets = self.make_widgets_from_config(reinit_params, reinit_extras, modal_message is None)

            # This should really go in the parent class, like submit_button
            if modal_message:
                self.reinit_button = bm.Button(
                    label=modal_button_label if modal_button_label else "Reconstruct points")
                self.reinit_button.on_click(self.reconstruct_points)
                self.make_modal(self.reinit_button, modal_message)
                reinit_widgets.append(self.reinit_button)

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
                all_weights.append(dat[2] if len(dat) > 2 else None)
                other_data.append(dat[3] if len(dat) > 3 else None)
        else:
            self.reconstruct_points_fn = None
            if reinit_params:
                raise ValueError("Saw reinit_params but data_source is not a callable")
            if reinit_extras:
                raise ValueError("Saw reinit_extras but data_source is not a callable")
            allx = data_source[0]
            ally = data_source[1]
            all_weights = data_source[2] if len(data_source) > 2 else [None]
            other_data = data_source[3] if len(data_source) > 3 else [None]

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
            for i, (fitting_parms, domain, x, y, weights, other) in \
                    enumerate(zip(fitting_parameters, domains, allx, ally, all_weights, other_data), start=1):
                tui = WavelengthSolutionPanel(
                    self, fitting_parms, domain, x, y, weights,
                    other_data=other_data, **kwargs)
                tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(i))
                self.tabs.tabs.append(tab)
                self.fits.append(tui.model)
        else:
            # ToDo: the domains variable contains a list. I changed it to
            #  domains[0] and the code worked.
            tui = WavelengthSolutionPanel(
                self, fitting_parameters[0], domains[0], allx[0], ally[0],
                all_weights[0], other_data=other_data[0], **kwargs)
            tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(1))
            self.tabs.tabs.append(tab)
            self.fits.append(tui.model)

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
                        if len(coords) > 3:
                            fit.other = coords[3]
                    else:
                        fit.weights = None
                    fit.weights = fit.populate_bokeh_objects(coords[0], coords[1], fit.weights, mask=None)
                    fit.perform_fit()
                if hasattr(self, 'reinit_button'):
                    self.reinit_button.disabled = False

            self.do_later(rfn)

    @property
    def other_data(self):
        return [fit.other_data for fit in self.fits]

    @property
    def image(self):
        image = []
        for model in self.fits:
            goodpix = np.array([m != USER_MASK_NAME for m in model.mask])
            image.append(model.y[goodpix])
        return image
