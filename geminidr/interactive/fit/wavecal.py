import uuid

import numpy as np
from bokeh.models import CustomJS
from scipy.interpolate import interp1d
from bisect import bisect

from bokeh import models as bm
from bokeh.layouts import row, column
from bokeh.plotting import figure

from geminidr.interactive.controls import Controller, Handler
from .help import DETERMINE_WAVELENGTH_SOLUTION_HELP_TEXT
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
    def __init__(self, visualizer, fitting_parameters, domain=None,
                 x=None, y=None, weights=None, absorption=False, meta=None, **kwargs):
        # No need to compute wavelengths here as the model_change_handler() does it
        self.absorption = absorption
        if absorption:
            self.spectrum = bm.ColumnDataSource({'wavelengths': np.zeros_like(meta["spectrum"]),
                                            'spectrum': -meta["spectrum"]})
        else:
            self.spectrum = bm.ColumnDataSource({'wavelengths': np.zeros_like(meta["spectrum"]),
                                             'spectrum': meta["spectrum"]})
        # Data for the reference plot
        self.show_refplot = False
        if not (meta.get("refplot_spec") is None or meta.get("refplot_linelist") is None):
            self.show_refplot = True
            self.refplot_spectrum = bm.ColumnDataSource({'wavelengths': meta["refplot_spec"][:,0],
                                                'refplot_spectrum': meta["refplot_spec"][:,1]})
            wlengths = meta["refplot_linelist"][:,0]
            self.refplot_linelist = bm.ColumnDataSource({
                                    'wavelengths': wlengths,
                                    'intensities': meta["refplot_linelist"][:,1],
                                    'labels': ["{:.2f}".format(w) for w in wlengths]})
            self.refplot_y_axis_label = meta["refplot_y_axis_label"]
            self.refplot_name = meta["refplot_name"]

        # This line is needed for the initial call to model_change_handler
        self.currently_identifying = False

        if len(x) == 0:
            kwargs["initial_fit"] = meta["fit"]

        super().__init__(visualizer, fitting_parameters, domain, x, y,
                         weights=weights, **kwargs)

        # This has to go on the model (and not this Panel instance) since the
        # models are returned by the Visualizer, not the Panel instances
        self.model.meta = meta
        self.model.allow_poor_fits = False

        self.new_line_marker = bm.ColumnDataSource(
            {"x": [min(self.spectrum.data['wavelengths'])] * 2, "y": [0, 0]})
        self.p_spectrum.line("x", "y", source=self.new_line_marker,
                             color="red", name="new_line_marker",
                             line_width=3, visible=False)
        self.set_currently_identifying(False)

    def set_currently_identifying(self, peak):
        status = bool(peak)
        self.p_spectrum.select_one({"name": "new_line_marker"}).visible = status

        def recursively_set_status(parent, disabled):
            for c in parent.children:
                if hasattr(c, "children"):
                    recursively_set_status(c, disabled)
                else:
                    c.disabled = disabled

        recursively_set_status(self.new_line_div, not status)
        self.identify_button.disabled = status
        self.currently_identifying = peak

    def build_figures(self, domain=None, controller_div=None,
                      plot_residuals=True, plot_ratios=True,
                      extra_masks=True):

        self.xpoint = 'fitted'
        self.ypoint = 'nonlinear'
        p_main, p_supp = fit1d_figure(width=self.width, height=self.height,
                                      xpoint=self.xpoint, ypoint=self.ypoint,
                                      xline='model', yline='nonlinear',
                                      xlabel=self.xlabel, ylabel=self.ylabel,
                                      model=self.model, plot_ratios=False,
                                      enable_user_masking=True)
        mask_handlers = (self.mask_button_handler, self.unmask_button_handler)
        Controller(p_main, None, self.model.band_model if self.enable_regions else None, controller_div,
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
                        text_baseline='middle', text_align='right',
                        text_font_size='10pt')
        delete_line_handler = Handler('d', "Delete arc line",
                                      self.delete_line)
        identify_line_handler = Handler('i', "Identify arc line",
                                        self.identify_line)
        c = Controller(p_spectrum, None, self.model.band_model if self.enable_regions else None, controller_div,
                   mask_handlers=None, domain=domain,
                   handlers=[delete_line_handler, identify_line_handler])
        c.helpmaskingtext += ("After selecting a line to identify, select its "
                              "wavelength (in nm) from the drop-down menu or "
                              "enter a value in the text box, then click 'OK'.")
        c.set_help_text()
        p_spectrum.y_range.on_change("start", lambda attr, old, new:
                                     self.update_label_heights())
        p_spectrum.y_range.on_change("end", lambda attr, old, new:
                                     self.update_label_heights())
        self.p_spectrum = p_spectrum

        self.identify_button = bm.Button(label="Identify lines", width=200,
                                         button_type="primary", width_policy="fit",
                                         height_policy="max")
        self.identify_button.on_click(self.identify_lines)

        self.clear_all_lines_button = bm.Button(label="Clear all lines", width=200,
                                         button_type="warning", width_policy="fit",
                                         height_policy="max")
        self.clear_all_lines_button.on_click(self.clear_all_lines)

        self.new_line_prompt = bm.Div(text="", style={"font-size": "16px",},
                                      width_policy="max")
        self.new_line_dropdown = bm.Select(options=[], width=100,
                                           width_policy="fixed")

        # Make a unique ID for our callback to find the HTML input widget
        # Note that even with the cb_obj reference, it's easier to find the real input widget this way
        focus_id = str(uuid.uuid4())
        self.new_line_textbox = bm.NumericInput(width=100, mode='float',
                                                width_policy="fixed", name=focus_id)

        # JS side listener to perform the focus.  We have to do it async via setTimeout
        # because bokeh triggers the disabled change before the html widget is ready for focus
        cb = CustomJS(code="""
                        if (cb_obj.disabled == false) {
                          setTimeout(function() {
                            document.getElementsByName('%s')[0].focus();
                          });
                        }
                      """ % focus_id)
        self.new_line_textbox.js_on_change('disabled', cb)

        self.new_line_dropdown.on_change("value", self.set_new_line_textbox_value)
        self.new_line_textbox.on_change("value", self.handle_line_wavelength)
        new_line_ok_button = bm.Button(label="OK", width=120, width_policy="fit",
                                       button_type="success")
        new_line_ok_button.on_click(self.add_new_line)
        new_line_cancel_button = bm.Button(label="Cancel", width=120, width_policy="fit",
                                           button_type="danger")
        new_line_cancel_button.on_click(self.cancel_new_line)
        #self.new_line_div = row(bm.Spacer(sizing_mode="stretch_width"),
        #                        self.new_line_prompt, self.new_line_dropdown,
        #                        self.new_line_textbox,
        #                        new_line_ok_button, new_line_cancel_button,
        #                        sizing_mode="stretch_both")
        self.new_line_div = row(column(row(self.new_line_prompt, self.new_line_dropdown,
                                self.new_line_textbox),
                                row(bm.Spacer(sizing_mode="stretch_width"), new_line_ok_button, new_line_cancel_button),
                                width_policy="max"))

        identify_panel = row(self.identify_button, self.clear_all_lines_button, self.new_line_div)

        info_panel = InfoPanel(self.enable_regions, self.enable_user_masking)
        self.model.add_listener(info_panel.model_change_handler)

        # ATRAN or other reference spectrum plot
        if self.show_refplot:
            p_refplot = figure(plot_width=self.width, plot_height=int(self.height // 1.5),
                                min_width=400, title=self.refplot_name,
                                x_axis_label="Wavelength (nm)", y_axis_label=self.refplot_y_axis_label,
                                tools = "pan,wheel_zoom,box_zoom,reset",
                                output_backend="webgl",
                                x_range=self.p_spectrum.x_range, y_range=None,
                                min_border_left=80)
            p_refplot.height_policy = 'fixed'
            p_refplot.width_policy = 'fit'
            p_refplot.sizing_mode = 'stretch_width'
            p_refplot.step(x='wavelengths', y='refplot_spectrum', source=self.refplot_spectrum,
                            line_width=1, color="gray", mode="center")
            p_refplot.text(name="labels",
                           x='wavelengths', y= 'heights', text='labels',
                            source=self.refplot_linelist, angle=0.5 * np.pi,
                            text_color='gray',
                            text_baseline='middle', text_align='right',
                            text_font_size='8pt')
            p_refplot.y_range.on_change("start", lambda attr, old, new:
                                         self.update_refplot_label_heights())
            p_refplot.y_range.on_change("end", lambda attr, old, new:
                                         self.update_refplot_label_heights())
            self.p_refplot = p_refplot

            return [p_refplot, p_spectrum, identify_panel, info_panel.component,
                p_main, p_supp]
        else:
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
        x : float/iterable
            pixel location of line(s)

        Returns
        -------
        float/list : appropriate y value(s) for writing a label
        """
        try:
            height = self.p_spectrum.y_range.end - self.p_spectrum.y_range.start
        except TypeError:  # range is None, plot being initialized
            # This is calculated on the basis that bokeh pads by 5% of the
            # data range on each side
            #height = (44 / 29 * self.spectrum.data['spectrum'].max() -
            #          1.1 * self.spectrum.data['spectrum'].min())
            height = 44 / 29 * np.nanmax(self.spectrum.data['spectrum'])
        if self.absorption:
            padding = -0.05 * height
        else:
            padding = 0.25 * height
        try:
            return [self.spectrum.data["spectrum"][int(xx + 0.5)] + padding for xx in x]
        except TypeError:
            return self.spectrum.data["spectrum"][int(x + 0.5)] + padding

    def refplot_label_height(self):
        """
        Provide a location for a wavelength label identifying a line
        in the reference spectrum plot

        Returns
        -------
        float/list : appropriate y value(s) for writing a label
        """
        try:
            height = self.p_refplot.y_range.end - self.p_refplot.y_range.start
        except TypeError:  # range is None, plot being initialized
            # This is calculated on the basis that bokeh pads by 5% of the
            # data range on each side
            height = 44 / 29 * np.nanmax(self.refplot_linelist.data["intensities"])
        if self.absorption:
            padding = -0.05 * height
        else:
            padding = 0.35 * height
        heights = self.refplot_linelist.data["intensities"] + padding
        return heights

    def update_label_heights(self):
        """Simple callback to move the labels if the spectrum panel is resized"""
        self.model.data.data['heights'] = self.label_height(self.model.x)
        if self.currently_identifying:
            lheight = 0.05 * (self.p_spectrum.y_range.end -
                              self.p_spectrum.y_range.start)
            if self.absorption:
                self.new_line_marker.data["y"][1] = self.new_line_marker.data["y"][0] - lheight
            else:
                self.new_line_marker.data["y"][1] = self.new_line_marker.data["y"][0] + lheight

    def update_refplot_label_heights(self):
        """Simple callback to move the labels if the reference spectrum panel is resized"""
        self.refplot_linelist.data['heights'] = self.refplot_label_height()

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
        if self.show_refplot:
            self.refplot_linelist.data['heights'] = self.refplot_label_height()
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
            self.cancel_new_line(*args)
            try:
                self.add_line_to_data(peak, wavelength)
            except ValueError:
                return

    def cancel_new_line(self, *args):
        """Handler for the 'Cancel' button in the line identifier"""
        self.new_line_prompt.text = ""
        self.new_line_dropdown.options = []
        self.new_line_textbox.value = None
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
            lower_limit, upper_limit = get_closest(
                self.model.y, self.model.evaluate(peak)[0])
            lower_limit, upper_limit = sorted([lower_limit, upper_limit])
            if not (lower_limit < wavelength < upper_limit):
                self.visualizer.show_user_message(
                    f"The value {wavelength} nm does not preserve a monotonic"
                     " sequence of identified line wavelengths")
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
            # This will fail if no line was deleted before user attempts to identify a line,
            # so do it only if there are new_peaks
            if len(new_peaks) > 0:
                index = np.argmin(abs(new_peaks - pixel))

            # If we've clicked "close" to a real peak (based on viewport size),
            # then select that
            if len(new_peaks) > 0 and \
                    (abs(self.model.evaluate(new_peaks[index]) - x) < 0.025 * (x2 - x1)):
                peak = new_peaks[index]
                print(f"Retrieved peak from list at {peak}")
            else:
                # TODO: Check this behaves sensibly, and doesn't find
                # all tiny bumps
                if self.absorption:
                    pinpoint_data = cwt_ricker(-self.spectrum.data["spectrum"],
                                           [0.42466 * fwidth])[0]
                else:
                    pinpoint_data = cwt_ricker(self.spectrum.data["spectrum"],
                                           [0.42466 * fwidth])[0]
                eps = np.finfo(np.float32).eps  # Minimum representative data
                pinpoint_data[np.nan_to_num(pinpoint_data) < eps] = eps
                try:
                    peak = pinpoint_peaks(pinpoint_data, [pixel], None)[0][0]
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
                                             key=lambda x: abs(x - est_wave))[:9])
            select_index = np.argmin(abs(np.asarray(selectable_lines) - est_wave))
            self.new_line_dropdown.options = [wavestr(line) for line in selectable_lines]
            self.new_line_dropdown.value = wavestr(selectable_lines[select_index])
            self.new_line_dropdown.disabled = False
        else:
            self.new_line_dropdown.options = []
            self.new_line_dropdown.disabled = True
        self.new_line_prompt.text = f"Line at {peak:.1f} ({est_wave:.5f} nm)"
        lheight = (self.p_spectrum.y_range.end -
                   self.p_spectrum.y_range.start) * (-0.05 if self.absorption else 0.05)
        # TODO: check what happens here in case of absorption -OS
        height = self.spectrum.data["spectrum"][int(peak + 0.5)]
        self.new_line_marker.data = {"x": [est_wave] * 2,
                                     "y": [height, height + lheight]}
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

    def clear_all_lines(self):
        """
        Called when the user clicks the "Clear all ines" button. Deletes
        all identified lines, performs a new fit.
        """
        self.model.data.data = {col: [] for col, values in self.model.data.data.items()}
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
            good_data[k] = [vv for vv, mask in zip(v, self.model.mask)
                            if mask == 'good']

        try:
            matches = match_sources(all_lines, good_data['y'], radius=0.01 * abs(dw))
        except ValueError:  # good_data['y'] is empty
            unmatched_lines = all_lines
        else:
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

    def handle_line_wavelength(self, attrib, old, new):
        """
        Handle user pressing Enter in the new line wavelength textbox.
        """
        if new is not None and wavestr(new) not in self.new_line_dropdown.options:
            self.add_new_line()

    def update_refplot_name(self, new_name):
        self.refplot_name = new_name
        self.p_refplot.title.text = new_name
        self.p_refplot.title.update()

class WavelengthSolutionVisualizer(Fit1DVisualizer):
    """
    A Visualizer specific to determineWavelengthSolution
    """
    def __init__(self, *args, absorption=None, **kwargs):
        self.num_atran_params = None
        super().__init__(*args, **kwargs, panel_class=WavelengthSolutionPanel,
                         help_text=DETERMINE_WAVELENGTH_SOLUTION_HELP_TEXT,
                         absorption=absorption)
        #self.widgets["in_vacuo"] = bm.RadioButtonGroup(
        #    labels=["Air", "Vacuum"], active=0)
        #self.reinit_panel.children[-3] = self.widgets["in_vacuo"]
        skip_lines = -3
        if self.num_atran_params is not None:
            skip_lines = skip_lines - self.num_atran_params - 1

        self.reinit_panel.children[skip_lines] = bm.Div(
            text="<b>Calibrating to wavelengths in {}</b>".format(
                "vacuo" if self.ui_params.in_vacuo else "air"), align="center")
        self.widgets["in_vacuo"].disabled = True
        del self.num_atran_params

        self.absorption = absorption

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

    def make_widgets_from_parameters(self, params,
                                     slider_width: int = 256, add_spacer=False,
                                     hide_textbox=None):
        linelist_reinit_params = None

        if params.reinit_params:
            for key in params.reinit_params:
                # The following is to add a special subset of UI widgets
                # for fine-tuning the generated ATRAN linelist
                if type(key) == dict and "atran_linelist_pars" in key:
                    linelist_reinit_params = key.get("atran_linelist_pars")
                    params.reinit_params.remove(key)
                    params.reinit_params = params.reinit_params+linelist_reinit_params

        lst = super().make_widgets_from_parameters(params)
        # If there are widgets for controlling ATRAN linelist, add
        # a title line above them:
        if linelist_reinit_params is not None:
            self.num_atran_params = len(linelist_reinit_params)
            section_title = bm.Div(
                text="Parameters for ATRAN linelist generation:",
                align="start", style={"font-weight":"bold"}, margin=(40,0,20,0))
            lst.insert((-self.num_atran_params), section_title)
        return lst

    def reconstruct_points_additional_work(self, data):
        """
        Reconstruct the initial points to work with.
        """
        super().reconstruct_points_additional_work(data)
        if data is not None:
            for i, fit in enumerate(self.fits):
                if self.returns_list:
                    this_dict = {k: v[i] for k, v in data.items()}
                else:
                    this_dict = data
                # spectrum update
                if self.absorption == True:
                    self.panels[i].spectrum.data['spectrum'] = -this_dict["meta"]["spectrum"]
                else:
                    self.panels[i].spectrum.data['spectrum'] = this_dict["meta"]["spectrum"]

                try:
                    self.panels[i].refplot_spectrum.data['wavelengths'] = this_dict["meta"]["refplot_spec"][:,0]
                    self.panels[i].refplot_spectrum.data['refplot_spectrum'] = this_dict["meta"]["refplot_spec"][:,1]

                    wlengths = this_dict["meta"]["refplot_linelist"][:,0]
                    self.panels[i].refplot_linelist.update(
                        data = {
                            'wavelengths': wlengths,
                            'intensities': this_dict["meta"]["refplot_linelist"][:,1],
                            'labels': ["{:.2f}".format(w) for w in wlengths]
                        }
                    )
                    self.panels[i].update_refplot_label_heights()
                    self.panels[i].update_refplot_name(this_dict["meta"]["refplot_name"])
                    self.panels[i].reset_view()
                except (AttributeError, KeyError):
                    pass


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
