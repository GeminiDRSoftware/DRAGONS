import numpy as np
from scipy.interpolate import interp1d
from bisect import bisect

from bokeh import models as bm
from bokeh.layouts import row, column
from bokeh.plotting import figure

from geminidr.interactive.controls import Controller, Handler
from gempy.library.matching import match_sources
from gempy.library.tracing import cwt_ricker, pinpoint_peaks

from .fit1d import (Fit1DPanel, Fit1DVisualizer, InfoPanel,
                    fit1d_figure, USER_MASK_NAME)


def wavestr(line):
    """Convert a line wavelength to a string, rounding the internal
    representation of the floating-point value"""
    return str(np.round(line, decimals=6))


def beep():
    """Make a beep noise"""
    print("\a", end="")


def disable_when_identifying(fn):
    """A decorator that prevents methods from being executed when the
    WavelengthSolutionPanel is currently identifying an arc line"""
    def gn(self, *args, **kwargs):
        if self.currently_identifying:
            beep()
            return
        fn(self, *args, **kwargs)
    return gn


class WavelengthSolutionPanel(Fit1DPanel):
    def __init__(self, visualizer, fitting_parameters, domain, x, y,
                 weights=None, meta=None, **kwargs):
        # No need to compute wavelengths here as the model_change_handler() does it
        self.spectrum = bm.ColumnDataSource({'wavelengths': np.zeros_like(meta["spectrum"]),
                                             'spectrum': meta["spectrum"]})
        # This line is needed for the initial call to model_change_handler
        self.currently_identifying = False

        super().__init__(visualizer, fitting_parameters, domain, x, y,
                         weights=weights, **kwargs)

        # This has to go on the model (and not this Panel instance) since the
        # models are returned by the Visualizer, not the Panel instances
        self.model.meta = meta

        self.new_line_marker = bm.ColumnDataSource(
            {"x": [min(self.spectrum.data['wavelengths'])] * 2, "y": [0, 0]})
        self.p_spectrum.line("x", "y", source=self.new_line_marker,
                             color="red", name="new_line_marker",
                             line_width=3, visible=False)
        self.set_currently_identifying(False)

    def set_currently_identifying(self, peak):
        status = bool(peak)
        self.p_spectrum.select_one({"name": "new_line_marker"}).visible = status
        for c in self.new_line_div.children:
            c.disabled = not status
        self.identify_button.disabled = status
        self.currently_identifying = peak

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
        self.p_spectrum = p_spectrum
        delete_line_handler = Handler('d', "Delete arc line",
                                      self.delete_line)
        identify_line_handler = Handler('i', "Identify arc line",
                                        self.identify_line)
        Controller(p_spectrum, None, self.model.band_model, controller_div,
                   mask_handlers=None, domain=domain,
                   handlers=[delete_line_handler, identify_line_handler])

        self.identify_button = bm.Button(label="Identify lines", width=200,
                                         button_type="primary", width_policy="fixed")
        self.identify_button.on_click(self.identify_lines)

        self.new_line_prompt = bm.Div(text="Line", style={"font-size": "16px",},
                                      width_policy="min")
        self.new_line_dropdown = bm.Select(options=[], width=100,
                                           width_policy="fixed")
        self.new_line_textbox = bm.NumericInput(width=100, mode='float',
                                                width_policy="fixed")
        self.new_line_dropdown.on_change("value", self.set_new_line_textbox_value)
        new_line_ok_button = bm.Button(label="OK", width=120, width_policy="fixed",
                                       button_type="success")
        new_line_ok_button.on_click(self.add_new_line)
        new_line_cancel_button = bm.Button(label="Cancel", width=120, width_policy="fixed",
                                           button_type="danger")
        new_line_cancel_button.on_click(self.cancel_new_line)
        self.new_line_div = row(bm.Spacer(sizing_mode="stretch_width"),
                                self.new_line_prompt, self.new_line_dropdown,
                                self.new_line_textbox,
                                new_line_ok_button, new_line_cancel_button,
                                sizing_mode="stretch_both")

        identify_panel = row(self.identify_button, self.new_line_div)

        info_panel = InfoPanel()
        self.model.add_listener(info_panel.model_change_handler)

        return [p_spectrum, identify_panel, info_panel.component,
                p_main, p_supp]

    @staticmethod
    def linear_model(model):
        """Return only the linear part of a model. It doesn't work for
        splines, which is why it's not in the InteractiveModel1D class"""
        model = model.fit._models
        return model.__class__(degree=1, c0=model.c0, c1=model.c1,
                               domain=model.domain)

    def label_height(self, x):
        """
        Provide a location for a wavelength label identifying a line

        Parameters
        ----------
        x : float
            pixel location of line

        Returns
        -------
        float : an appropriate y value for writing a label
        """
        padding = 0.02 * self.spectrum.data['spectrum'].max()
        try:
            return [self.spectrum.data["spectrum"][int(xx + 0.5)] + padding for xx in x]
        except TypeError:
            return self.spectrum.data["spectrum"][int(x + 0.5)] + padding

    # I could put the extra stuff in a second listener but the name of this
    # is generic, so let's just super() it and then do the extra stuff
    def model_change_handler(self, model):
        """
        If the `~fit` changes, this gets called to evaluate the fit and save the results.
        """
        super().model_change_handler(model)
        x, y = model.x, model.y
        linear_model = self.linear_model(model)

        self.model.data.data['fitted'] = model.evaluate(x)
        self.model.data.data['nonlinear'] = y - linear_model(x)
        self.model.data.data['heights'] = self.label_height(x)
        self.model.data.data['lines'] = [wavestr(yy) for yy in y]

        self.model.evaluation.data['nonlinear'] = (
                model.evaluation.data['model'] -
                linear_model(model.evaluation.data['xlinspace']))

        domain = model.domain
        self.spectrum.data['wavelengths'] = model.evaluate(
            np.arange(domain[0], domain[1]+1))

        # If we recalculated the model while in the middle of identifying a
        # peak, keep the peak location but update the line choices
        if self.currently_identifying:
            peak = self.currently_identifying
            self.currently_identifying = False
            self.identify_line('i', 0, 0, peak=peak)

    def add_new_line(self, *args):
        """Handler for the 'OK' button in the line identifier"""
        if self.currently_identifying:
            try:
                wavelength = float(self.new_line_textbox.value)
            except TypeError:
                beep()
                return
            peak = self.currently_identifying
            try:
                self.add_line_to_data(peak, wavelength)
            except ValueError:
                return
            self.cancel_new_line(*args)

    def cancel_new_line(self, *args):
        """Handler for the 'Cancel' button in the line identifier"""
        self.new_line_prompt.text = ""
        self.new_line_dropdown.options = []
        self.set_currently_identifying(False)

    def add_line_to_data(self, peak, wavelength):
        """
        Add a new line to the ColumnDataSource and performs a new fit.

        Parameters
        ----------
        peak : float
            pixel location of peak
        wavelength : float
            wavelength in nm
        """
        print(f"Adding {wavelength} nm at pixel {peak}")
        if self.model.x.size > 1:
            lower_limit, upper_limit = get_closest(self.model.x, peak)
            if not np.isinf(lower_limit):
                lower_limit = self.model.y[list(self.model.x).index(lower_limit)]
            if not np.isinf(upper_limit):
                upper_limit = self.model.y[list(self.model.x).index(upper_limit)]
            lower_limit, upper_limit = sorted([lower_limit, upper_limit])
            if not (lower_limit < wavelength < upper_limit):
                self.visualizer.show_user_message(
                    f"The value {wavelength} nm does not preserve a monotonic"
                     "sequence of identified line wavelengths")
                raise ValueError
        # Dummy values should be close to true values to avoid plot resizing
        new_data = {'x': [peak], 'y': [wavelength], 'mask': ['good'],
                    'fitted': [wavelength], 'nonlinear': [0],
                    'heights': [self.label_height(peak)],
                    'residuals': [0],
                    'lines': [wavestr(wavelength)],
                   }
        self.model.data.stream(new_data)
        self.model.perform_fit()

    @disable_when_identifying
    def identify_line(self, key, x, y, peak=None):
        """
        Identifies a peak near the cursor location and allows the user to
        provide a wavelength.
        """
        if peak is None:
            x1, x2 = self.p_spectrum.x_range.start, self.p_spectrum.x_range.end
            fwidth = self.model.meta["fwidth"]
            pixel = interp1d(self.spectrum.data["wavelengths"],
                             range(len(self.spectrum.data["wavelengths"])))(x)
            new_peaks = np.setdiff1d(self.model.meta["peaks"],
                                     self.model.x, assume_unique=True)
            index = np.argmin(abs(new_peaks - pixel))

            # If we've clicked "close" to a real peak (based on viewport size),
            # then select that
            if abs(self.model.evaluate(new_peaks[index]) - x) < 0.025 * (x2 - x1):
                peak = new_peaks[index]
                print(f"Retrieved peak from list at {peak}")
            else:
                # TODO: Check this behaves sensibly, and doesn't find
                # all tiny bumps
                pinpoint_data = cwt_ricker(self.spectrum.data["spectrum"],
                                           [0.42466 * fwidth])
                eps = np.finfo(np.float32).eps  # Minimum representative data
                pinpoint_data[np.nan_to_num(pinpoint_data) < eps] = eps
                try:
                    peak = pinpoint_peaks(pinpoint_data, None, [pixel])[0]
                    print(f"Found peak at pixel {peak}")
                except IndexError:  # no peak
                    print("Couldn't find a peak")
                    return
            est_wave = self.model.evaluate(peak)[0]
            if not (x1 < est_wave < x2):  # peak outside viewport
                return
        else:
            est_wave = self.model.evaluate(peak)[0]  # evaluate always returns array

        # Find all unidentified arc lines that this could be, maintaining
        # monotonicity
        all_lines = self.model.meta["linelist"].wavelengths(
            in_vacuo=self.visualizer.ui_params.in_vacuo, units="nm")
        lower_limit, upper_limit = get_closest(self.model.y, est_wave)
        possible_lines = [line for line in all_lines
                          if lower_limit < line < upper_limit]
        if possible_lines:
            selectable_lines = sorted(sorted(possible_lines,
                                             key=lambda x: abs(x - est_wave))[:5])
            select_index = np.argmin(abs(np.asarray(selectable_lines) - est_wave))
            self.new_line_dropdown.options = [wavestr(line) for line in selectable_lines]
            self.new_line_dropdown.value = wavestr(selectable_lines[select_index])
            self.new_line_dropdown.disabled = False
        else:
            self.new_line_dropdown.options = []
            self.new_line_dropdown.disabled = True
        self.new_line_prompt.text = f"Line at {peak:.1f} ({est_wave:.5f} nm)"
        lheight = 0.05 * (self.p_spectrum.y_range.end -
                          self.p_spectrum.y_range.start)
        self.new_line_marker.data = {"x": [est_wave] * 2,
                                     "y": [self.label_height(peak),
                                           self.label_height(peak) + lheight]}
        self.set_currently_identifying(peak)

    @disable_when_identifying
    def delete_line(self, key, x, y):
        """
        Delete (not mask) from the fit the line nearest the cursor. This
        operates only on the spectrum panel.
        """
        index = np.argmin(abs(self.model.data.data['fitted'] - x))
        new_data = {col: list(values)[:index] + list(values)[index+1:]
                    for col, values in self.model.data.data.items()}
        self.model.data.data = new_data
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
        linear_model = self.linear_model(self.model)
        dw = linear_model.c1 / np.diff(linear_model.domain)[0]
        matching_distance = abs(self.model.meta["fwidth"] * dw)
        all_lines = self.model.meta["linelist"].wavelengths(
            in_vacuo=self.visualizer.ui_params.in_vacuo, units="nm")

        good_data = {}
        for k, v in self.model.data.data.items():
            good_data[k] = [vv for vv, mask in zip(v, self.model.data.data['mask'])
                            if mask == 'good']

        matches = match_sources(all_lines, good_data['y'], radius=0.01 * abs(dw))
        unmatched_lines = [l for l, m in zip(all_lines, matches) if m == -1]

        new_peaks = np.setdiff1d(self.model.meta["peaks"],
                                 good_data['x'], assume_unique=True)
        new_waves = self.model.evaluate(new_peaks)

        matches = match_sources(new_waves, unmatched_lines, radius=matching_distance)
        for peak, m in zip(new_peaks, matches):
            if m != -1:
                good_data['x'].append(peak)
                good_data['y'].append(unmatched_lines[m])
                good_data['mask'].append('good')
                good_data['fitted'].append(unmatched_lines[m])
                good_data['nonlinear'].append(0)
                good_data['heights'].append(self.label_height(peak))
                good_data['residuals'].append(0)
                good_data['lines'].append(wavestr(unmatched_lines[m]))
                print("NEW LINE", peak, unmatched_lines[m])

        self.model.data.data = good_data
        self.model.perform_fit()

    def set_new_line_textbox_value(self, attrib, old, new):
        """Update the value of the textbox related to the new line ID"""
        if new != old:
            self.new_line_textbox.value = float(new)

class WavelengthSolutionVisualizer(Fit1DVisualizer):
    """
    A Visualizer specific to determineWavelengthSolution

    This differs from the parent class in the following ways:
    1) __init__()
        (a) each tab is a WavelengthSolutionPanel, not a Fit1DPanel
        (b) the data_source returns a fourth column, a dict containing
            additional information, that gets put in an "meta" attribute
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

    def __init__(self, data_source, fitting_parameters,
                 modal_message=None,
                 modal_button_label=None,
                 tab_name_fmt='{}',
                 xlabel='x', ylabel='y',
                 domains=None, title=None, primitive_name=None, filename_info=None,
                 template="fit1d.html", help_text=None, recalc_inputs_above=False,
                 ui_params=None,
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
        ui_params : :class:`~geminidr.interactive.interactive.UIParams`
            Parameter set for user input
        """
        super(Fit1DVisualizer, self).__init__(
            title=title, primitive_name=primitive_name, filename_info=filename_info,
            template=template, help_text=help_text, ui_params=ui_params)
        self.layout = None
        self.recalc_inputs_above = recalc_inputs_above

        meta = []
        # Make the widgets accessible from external code so we can update
        # their properties if the default setup isn't great
        self.widgets = {}

        # If we have a widget driving the modal dialog via it's enable/disable state,
        # store it in this so the recalc knows to re-enable the widget
        self.modal_widget = None

        # Make the panel with widgets to control the creation of (x, y) arrays

        # Create left panel
        reinit_widgets = self.make_widgets_from_parameters(ui_params, reinit_live=modal_message is None)
        if reinit_widgets:
            # This should really go in the parent class, like submit_button
            if modal_message:
                if len(reinit_widgets) > 1:
                    self.reinit_button = bm.Button(label=modal_button_label if modal_button_label
                                                   else "Reconstruct points")
                    self.reinit_button.on_click(self.reconstruct_points)
                    self.make_modal(self.reinit_button, modal_message)
                    reinit_widgets.append(self.reinit_button)
                    self.modal_widget = self.reinit_button
                elif len(reinit_widgets) == 1:
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
        self.widgets["in_vacuo"].disabled = True

        # Grab input coordinates or calculate if we were given a callable
        # TODO revisit the raging debate on `callable` for Python 3
        if callable(data_source):
            self.reconstruct_points_fn = data_source
            data = data_source(ui_params=ui_params)
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
                meta.append(dat[3] if len(dat) > 3 else None)
        else:
            self.reconstruct_points_fn = None
            allx = data_source[0]
            ally = data_source[1]
            all_weights = data_source[2] if len(data_source) > 2 else [None]
            meta = data_source[3] if len(data_source) > 3 else [None]

        # Some sanity checks now
        if isinstance(fitting_parameters, list):
            if not (len(fitting_parameters) == len(allx) == len(ally)):
                raise ValueError("Different numbers of models and coordinates")
            self.nfits = len(fitting_parameters)
        else:
            if allx.size != ally.size:
                raise ValueError("Different (x, y) array sizes")
            self.nfits = 1

        kwargs.update({'xlabel': xlabel, 'ylabel': ylabel})

        self.tabs = bm.Tabs(tabs=[], name="tabs")
        self.tabs.sizing_mode = 'scale_width'
        self.fits = []

        if self.nfits > 1:
            if domains is None:
                domains = [None] * len(fitting_parameters)
            for i, (fitting_parms, domain, x, y, weights, other) in \
                    enumerate(zip(fitting_parameters, domains, allx, ally, all_weights, meta), start=1):
                tui = WavelengthSolutionPanel(
                    self, fitting_parms, domain, x, y, weights,
                    meta=meta, **kwargs)
                tab = bm.Panel(child=tui.component, title=tab_name_fmt.format(i))
                self.tabs.tabs.append(tab)
                self.fits.append(tui.model)
        else:
            # ToDo: the domains variable contains a list. I changed it to
            #  domains[0] and the code worked.
            tui = WavelengthSolutionPanel(
                self, fitting_parameters[0], domains[0], allx[0], ally[0],
                all_weights[0], meta=meta[0], **kwargs)
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

        rollback_config = self.ui_params.values.copy()
        def fn():
            """Top-level code to update the Config with the values from the widgets"""
            config_update = {k: v.value for k, v in self.widgets.items()}
            self.ui_params.update_values(**config_update)

        self.do_later(fn)

        if self.reconstruct_points_fn is not None:
            def rfn():
                all_coords = None
                try:
                    all_coords = self.reconstruct_points_fn(ui_params=self.ui_params)
                except Exception as e:
                    # something went wrong, let's revert the inputs
                    # handling immediately to specifically trap the reconstruct_points_fn call
                    self.ui_params.update_values(**rollback_config)
                    self.show_user_message("Unable to build data from inputs, reverting")
                if all_coords is not None:
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

    @property
    def meta(self):
        return [fit.meta for fit in self.fits]

    @property
    def image(self):
        image = []
        for model in self.fits:
            goodpix = np.array([m != USER_MASK_NAME for m in model.mask])
            image.append(model.y[goodpix])
        return image


def get_closest(arr, value):
    """
    Return the array values closest to the request value, or +/-inf if
    the request value is beyond the range of the array

    Parameters
    ----------
    arr : sequence
        array of values
    value : numeric

    Returns
    -------
    2-tuple: largest value in array less than value (or -inf) and
             smallest value in array larger than value (or +inf)
    """
    arr_sorted = sorted(arr)
    index = bisect(arr_sorted, value)
    lower_limit = -np.inf if index == 0 else arr_sorted[index - 1]
    upper_limit = np.inf if index == len(arr_sorted) else arr_sorted[index]
    return lower_limit, upper_limit
