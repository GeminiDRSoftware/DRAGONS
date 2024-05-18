"""Wavelength calibration interactive fitting panel and visualizer.

This module contains the WavelengthSolutionPanel and
WavelengthSolutionVisualizer classes, which are used to interactively fit a
wavelength solution to a spectrum.
"""
import logging
import uuid

from bisect import bisect

from functools import lru_cache

from bokeh import models as bm
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models import CustomJS

import numpy as np

from scipy.interpolate import interp1d

from geminidr.interactive.controls import Controller, Handler
from geminidr.interactive.styles import dragons_styles


from gempy.library.matching import match_sources
from gempy.library.tracing import cwt_ricker, pinpoint_peaks
from gempy.library.fitting import fit_1D

from .fit1d import (
    Fit1DPanel,
    Fit1DVisualizer,
    InfoPanel,
    fit1d_figure,
    USER_MASK_NAME,
)

from .help import DETERMINE_WAVELENGTH_SOLUTION_HELP_TEXT


@lru_cache(maxsize=1000)
def wavestr(line):
    """Convert a line wavelength to a string, rounding the internal
    representation of the floating-point value

    Parameters
    ----------
    line : float
        Wavelength of the line.

    Notes
    -----
    This function is lru cached to avoid unnecessary calls/conversions. It has
    a maximum cache size of 1000, which should be enough for most use cases.
    """
    return str(np.round(line, decimals=6))


def beep():
    """Make a beep noise"""
    print("\a", end="")


def disable_when_identifying(func):
    """A decorator that prevents methods from being executed when the
    WavelengthSolutionPanel is currently identifying an arc line.
    """

    def wrapper(self, *args, **kwargs):
        if self.currently_identifying:
            beep()
            return

        func(self, *args, **kwargs)

    return wrapper


class WavelengthSolutionPanel(Fit1DPanel):
    """Wavelenth solution fitting panel.

    This class is used to manage the interactive fitting of a peak to a
    specific wavelength in a spectrum. This is usually handled within a bokeh
    Tab, and this class manages the contents of that tab.
    """
    def __init__(self, visualizer, fitting_parameters, domain=None,
                 x=None, y=None, weights=None, absorption=False, meta=None, **kwargs):

        """Initializes the WavelengthSolutionPanel.

        Parameters
        ----------
        visualizer : WavelengthSolutionVisualizer
            The visualizer that is managing this panel.

        fitting_parameters : dict
            A dictionary of fitting parameters.

        domain : 2-tuple
            The domain of the fit.

        x : array-like
            The x values of the fit.

        y : array-like
            The y values of the fit.

        weights : array-like
            The weights of the fit.

        absorption :  boolean
            Whether we are fitting absorption lines or not.

        meta : dict
            A dictionary of metadata. This is the "all_input_data"

        kwargs : dict
            Any additional keyword arguments.

        Notes
        -----
        This class is a subclass of |Fit1DPanel| and so inherits all of its
        """

        # No need to compute wavelengths here as the model_change_handler() does it
        self.absorption = absorption
        if absorption:
            spectrum_data_dict = {
                "wavelengths": np.zeros_like(meta["spectrum"]),
                "spectrum": -meta["spectrum"],
            }
        else:
            spectrum_data_dict = {
                "wavelengths": np.zeros_like(meta["spectrum"]),
                "spectrum": meta["spectrum"],
            }

        kwargs["default_model"] = meta["init_models"][0]

        self.spectrum = bm.ColumnDataSource(spectrum_data_dict)
        self.spectrum_linelist = meta["linelist"]

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

        super().__init__(
            visualizer,
            fitting_parameters,
            domain,
            x,
            y,
            weights=weights,
            **kwargs,
        )

        # This has to go on the model (and not this TabPanel instance) since
        # the models are returned by the Visualizer, not the TabPanel
        # instances.
        self.model.meta = meta
        self.model.allow_poor_fits = False

        self.new_line_marker = bm.ColumnDataSource(
            {"x": [min(self.spectrum.data["wavelengths"])] * 2, "y": [0, 0]}
        )

        self.p_spectrum.line(
            "x",
            "y",
            source=self.new_line_marker,
            color="red",
            name="new_line_marker",
            line_width=3,
            visible=False,
        )

        self.set_currently_identifying(False)

    def set_currently_identifying(self, peak):
        """Specify whether the panel is currently identifying a peak, and set
        the panel state accordingly.

        Parameters
        ----------
        peak : bool or Number
            Whether the panel is currently identifying a peak.
        """
        status = bool(peak)

        spectrum = self.p_spectrum.select_one({"name": "new_line_marker"})
        spectrum.visible = status

        def recursively_set_status(parent, disabled):
            for child in parent.children:
                if hasattr(child, "children"):
                    recursively_set_status(child, disabled)

                else:
                    child.disabled = disabled

        recursively_set_status(self.new_line_row, not status)
        self.identify_button.disabled = status
        self.currently_identifying = peak

    def build_figures(
        self,
        domain=None,
        controller_div=None,
        plot_residuals=True,
        plot_ratios=False,
        extra_masks=False,
    ):
        """Build the figures for the panel.

        Parameters
        ----------
        domain : 2-tuple
            The domain of the fit. Default none.

        controller_div : bokeh.models.Div
            The div to hold the controller. Default None.

        plot_residuals : bool
            Whether to plot the residuals. This is only included to match the
            build_figures method in Fit1DPanel. Default True

        plot_ratios : bool
            Whether to plot the ratios. This is only included to match the
            build_figures method in Fit1DPanel. Default False

        extra_masks : bool
            Whether to plot the extra masks. This is only included to match the
            build_figures method in Fit1DPanel. Default False

        Notes
        -----
        This method is overridden to add the spectrum panel and the line
        identifier.

        The last three kwargs---plot_residuals, plot_ratios, and
        extra_masks---are not used in this function, and are only included to
        match the signature of Fit1DPanel.build_figures.
        """
        # Catch user attempting to use plot_residuals, plot_ratios, or
        # extra_masks, and print a warning.
        if not plot_residuals:
            logging.warning(
                "plot_residuals is always enabled in WavelengthSolutionPanel."
            )

        # Plot ratios is not used
        if plot_ratios:
            logging.warning(
                "plot_ratios is always disabled in WavelengthSolutionPanel."
            )

        if extra_masks:
            logging.warning(
                "extra_masks is always disabled in WavelengthSolutionPanel."
            )

        self.xpoint = "fitted"
        self.ypoint = "nonlinear"

        p_main, p_supp = fit1d_figure(
            width=self.width,
            height=self.height,
            xpoint=self.xpoint,
            ypoint=self.ypoint,
            xline="model",
            yline="nonlinear",
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            model=self.model,
            plot_ratios=False,
            enable_user_masking=True,
        )

        mask_handlers = (self.mask_button_handler, self.unmask_button_handler)

        Controller(
            p_main,
            None,
            self.model.band_model if self.enable_regions else None,
            controller_div,
            mask_handlers=mask_handlers,
            domain=domain,
        )

        p_spectrum = figure(
            height=self.height,
            min_width=400,
            title="Spectrum",
            x_axis_label=self.xlabel,
            y_axis_label="Signal",
            tools="pan,wheel_zoom,box_zoom,reset",
            output_backend="webgl",
            x_range=p_main.x_range,
            y_range=(np.nan, np.nan),
            min_border_left=80,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        p_spectrum.step(
            x="wavelengths",
            y="spectrum",
            source=self.spectrum,
            line_width=1,
            color="blue",
            mode="center",
        )

        # Offset is in pixels in screen space (up is negative, down is
        # positive) Technically, this is actually relative to the anchor point
        # for the text, which is the center of the text, so it's inverse the
        # common axes logic. And also not standard for, e.g., image coordinates
        # in most software.
        font_size = 10
        y_offset = -5
        text_align = 'left'

        if self.absorption:
            y_offset = -y_offset
            text_align = 'right'

        self._spectrum_line_labels = p_spectrum.text(
            x="fitted",
            y="heights",
            text="lines",
            y_offset=y_offset,
            source=self.model.data,
            angle=0.5 * np.pi,
            text_color=self.model.mask_rendering_kwargs()["color"],
            text_baseline="middle",
            text_align=text_align,
            text_font_size=f"{font_size}pt",
        )

        delete_line_handler = Handler("d", "Delete arc line", self.delete_line)

        identify_line_handler = Handler(
            "i", "Identify arc line", self.identify_line
        )

        controls = Controller(
            p_spectrum,
            None,
            self.model.band_model if self.enable_regions else None,
            controller_div,
            mask_handlers=None,
            domain=domain,
            handlers=[delete_line_handler, identify_line_handler],
        )

        controls.helpmaskingtext += (
            "After selecting a line to identify, select its "
            "wavelength (in nm) from the drop-down menu or "
            "enter a value in the text box, then click 'OK'."
        )

        controls.set_help_text()

        # The set_*_axes functions work on the stored reference itself.
        self.p_spectrum = p_spectrum
        self._spectrum_label_font_size = font_size

        self._set_spectrum_plot_axes()

        self.identify_button = bm.Button(
            label="Identify lines",
            width=200,
            max_width=250,
            button_type="primary",
            sizing_mode="stretch_both",
            stylesheets=dragons_styles(),
        )

        self.identify_button.on_click(self.identify_lines)

        self.clear_all_lines_button = bm.Button(
            label="Clear all lines",
            width=200,
            max_width=250,
            button_type="warning",
            width_policy="fit",
            height_policy="max",
            stylesheets=dragons_styles()
        )
        self.clear_all_lines_button.on_click(self.clear_all_lines)

        self.new_line_prompt = bm.Div(
            text="",
            styles={"font-size": "16px"},
            max_width=600,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        self.new_line_dropdown = bm.Select(
            options=[],
            width=100,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        # Make a unique ID for our callback to find the HTML input widget Note
        # that even with the cb_obj reference, it's easier to find the real
        # input widget this way
        focus_id = str(uuid.uuid4())

        self.new_line_textbox = bm.NumericInput(
            width=100,
            mode="float",
            sizing_mode="stretch_width",
            name=focus_id,
            stylesheets=dragons_styles(),
        )

        # JS side listener to perform the focus.  We have to do it async via
        # setTimeout because bokeh triggers the disabled change before the html
        # widget is ready for focus
        callback = CustomJS(
            code="""
                        if (cb_obj.disabled == false) {
                          setTimeout(function() {
                            document.getElementsByName('%s')[0].focus();
                          });
                        }
                      """
            % focus_id
        )

        self.new_line_textbox.js_on_change("disabled", callback)

        self.new_line_dropdown.on_change(
            "value", self.set_new_line_textbox_value
        )

        self.new_line_textbox.on_change("value", self.handle_line_wavelength)

        new_line_ok_button = bm.Button(
            label="OK",
            min_width=120,
            width_policy="fit",
            button_type="success",
            stylesheets=dragons_styles(),
        )

        new_line_ok_button.on_click(self.add_new_line)

        new_line_cancel_button = bm.Button(
            label="Cancel",
            min_width=120,
            width_policy="fit",
            button_type="danger",
            stylesheets=dragons_styles(),
        )

        new_line_cancel_button.on_click(self.cancel_new_line)

        line_selection_row = row(
            self.new_line_prompt,
            self.new_line_dropdown,
            self.new_line_textbox,
            stylesheets=dragons_styles(),
        )

        confirm_cancel_row = row(
            bm.Spacer(
                sizing_mode="stretch_width",
                stylesheets=dragons_styles(),
            ),
            new_line_ok_button,
            new_line_cancel_button,
            stylesheets=dragons_styles(),
        )

        self.new_line_row = row(
            column(
                line_selection_row,
                confirm_cancel_row,
                sizing_mode="stretch_width",
                stylesheets=dragons_styles(),
            ),
            stylesheets=dragons_styles(),
            sizing_mode="stretch_width",
        )

        identify_panel = row(
            self.identify_button,
            self.clear_all_lines_button,
            self.new_line_row,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        info_panel = InfoPanel(self.enable_regions, self.enable_user_masking)

        self.model.add_listener(info_panel.model_change_handler)

        # ATRAN or other reference spectrum plot
        if self.show_refplot:
            p_refplot = figure(width=self.width, height=int(self.height // 1.5),
                                min_width=400, title=self.refplot_name,
                                x_axis_label="Wavelength (nm)", y_axis_label=self.refplot_y_axis_label,
                                tools = "pan,wheel_zoom,box_zoom,reset",
                                output_backend="webgl",
                                x_range=self.p_spectrum.x_range,
                                y_range=(np.nan, np.nan),
                                min_border_left=80, stylesheets=dragons_styles())
            p_refplot.height_policy = 'fixed'
            p_refplot.width_policy = 'fit'
            p_refplot.sizing_mode = 'stretch_width'
            p_refplot.step(x='wavelengths', y='refplot_spectrum', source=self.refplot_spectrum,
                            line_width=1, color="gray", mode="center")

            # Offset is in pixels in screen space (up is negative, down is
            # positive) Technically, this is actually relative to the anchor
            # point for the text, which is the center of the text, so it's
            # inverse the common axes logic. And also not standard for, e.g.,
            # image coordinates in most software.
            y_offset = -5
            text_align = 'left'

            # Font size (pt)
            font_size = 8

            if self.absorption:
                y_offset = -y_offset
                text_align = 'right'

            self._replot_line_labels = p_refplot.text(name="labels",
                                                      x='wavelengths', y='intensities', text='labels',
                                                      y_offset=y_offset,
                                                      source=self.refplot_linelist, angle=0.5 * np.pi,
                                                      text_color='gray',
                                                      text_baseline='middle', text_align=text_align,
                                                      text_font_size=f'{font_size}pt')

            # The set_*_axes functions work on the stored reference itself.
            self.p_refplot = p_refplot
            self._refplot_label_font_size = font_size

            self._set_refplot_axes()

            return [p_refplot, p_spectrum, identify_panel, info_panel.component,
                p_main, p_supp]
        else:
            return [p_spectrum, identify_panel, info_panel.component,
                p_main, p_supp]

    def reset_spectrum_axes(self):
        """Reset the axes for the spectrum plot. If a reference plot is shown,
        reset the axes for that as well.
        """
        self._set_spectrum_plot_axes()

        if self.show_refplot:
            self._set_refplot_axes()

    def _set_spectrum_plot_axes(self):
        """Set the axes for the spectrum plot.

        Notes
        -----
        This should only be called if the spectrum plot is shown. It can be
        called before the plot has bounds, though.

        This also overwrites the original bounds of the plot.
        """
        p_spectrum = self.p_spectrum
        font_size = self._spectrum_label_font_size

        # Set the initial vertical range to include some padding for labels
        label_positions = self.label_height(self.model.x)

        # Some of the following code will fail if there are no labels for any
        # reason (e.g., no fit). In that case, we'll just set the range to the
        # min/max of the spectrum data, which we're collecting twice here (but
        # it's cached by the ufunc anyways).
        if not label_positions:
            label_positions = self.spectrum.data["spectrum"]

        min_line_intensity = np.amin(label_positions)
        min_spectrum_intensity = np.amin(self.spectrum.data["spectrum"])
        max_line_intensity = np.amax(label_positions)
        max_spectrum_intensity = np.amax(self.spectrum.data["spectrum"])
        y_low = min(min_line_intensity, min_spectrum_intensity)
        y_high = max(max_line_intensity, max_spectrum_intensity)
        spectrum_range = y_high - y_low
        max_assumed_chars = 10
        text_padding = max_assumed_chars * font_size
        margin_padding = 0.1 * spectrum_range

        # Scale text padding to plot units
        text_padding *= 4 * (y_high - y_low) / (3 * p_spectrum.height)

        if self.absorption:
            y_low -= text_padding + margin_padding
            y_high += margin_padding

        else:
            y_low -= margin_padding
            y_high += text_padding + margin_padding

        p_spectrum.y_range.start = y_low
        p_spectrum.y_range.end = y_high
        p_spectrum.y_range.reset_start = y_low
        p_spectrum.y_range.reset_end = y_high


    def _set_refplot_axes(self):
        """Set the axes for the reference plot.

        This should only be called if the reference plot is shown.

        Notes
        -----
        This should only be called if the spectrum plot is shown. It can be
        called before the plot has bounds, though.

        This also overwrites the original bounds of the plot.

        Raises
        ------
        AttributeError
            If the reference plot is not found.
        """
        if not hasattr(self, "p_refplot"):
            msg = (
                f"Reference plot not found in {self.__class__.__name__}, "
                f"expected a plot at self.p_refplot."
            )
            raise AttributeError(msg)

        p_refplot = self.p_refplot
        font_size = self._refplot_label_font_size

        # Set the initial vertical range to include some padding for labels
        linelist_intensity = self.refplot_linelist.data["intensities"]
        spectrum_intensity = self.refplot_spectrum.data["refplot_spectrum"]

        min_line_intensity = np.amin(linelist_intensity)
        min_spectrum_intensity = np.amin(spectrum_intensity)
        max_line_intensity = np.amax(linelist_intensity)
        max_spectrum_intensity = np.amax(spectrum_intensity)
        y_low = min(min_line_intensity, min_spectrum_intensity)
        y_high = max(max_line_intensity, max_spectrum_intensity)
        spectrum_range = y_high - y_low
        labels = self.refplot_linelist.data["labels"]
        text_padding = max(len(x) for x in labels) * font_size
        margin_padding = 0.1 * spectrum_range

        # Scale text padding to plot units
        text_padding *= 4 * (y_high - y_low) / (3 * p_refplot.height)

        if self.absorption:
            y_low -= text_padding + margin_padding
            y_high += margin_padding

        else:
            y_low -= margin_padding
            y_high += text_padding + margin_padding

        p_refplot.y_range.start = y_low
        p_refplot.y_range.end = y_high
        p_refplot.y_range.reset_start = y_low
        p_refplot.y_range.reset_end = y_high

    def reset_view(self):
        super().reset_view()
        self.reset_spectrum_axes()

    @staticmethod
    def linear_model(model):
        """Return only the linear part of a model. It doesn't work for
        splines, which is why it's not in the InteractiveModel1D class
        """
        if model.fit is None:
            model = model.default_model
        else:
            model = model.fit.model
        new_model = model.__class__(
            degree=1, c0=model.c0, c1=model.c1, domain=model.domain
        )

        return new_model

    @staticmethod
    def _get_plot_ranges(plot: figure):
        """Get the start and end of the x and y ranges of a plot."""
        x_range = plot.x_range
        y_range = plot.y_range

        ranges = {
            "x": (x_range.start, x_range.end),
            "y": (y_range.start, y_range.end),
        }

        return ranges

    def plot_range(self):
        return self._get_plot_ranges(self.p_spectrum)

    def refplot_range(self):
        return self._get_plot_ranges(self.p_refplot)

    def label_height(self, x):
        """Provide a location for a wavelength label identifying a line.

        This calculates the label heights relative to lines already present in
        the plot, and lets bokeh handle rescaling the y-axis as needed.

        Parameters
        ----------
        x : float/iterable
            pixel location of line(s)

        Returns
        -------
        float/list : appropriate y value(s) for writing a label
        """
        try:
            iter(x)
            single_value = False

        except TypeError:
            single_value = True
            x = [x]

        spectrum = self.spectrum.data["spectrum"]

        # Get points around the line.
        def get_nearby_points(p):
            distance = 5
            low = max(0, int(p - distance + 0.5))
            high = min(len(spectrum), int(p + distance + 0.5))
            return spectrum[low:high]

        extrema_func = np.amin if self.absorption else np.amax

        heights = [extrema_func(get_nearby_points(xx)) for xx in x]

        if single_value:
            return heights[0]

        return heights

    def refplot_label_height(self):
        """
        Provide a location for a wavelength label identifying a line
        in the reference spectrum plot.

        This calculates the label heights relative to lines already present in
        the plot, and lets bokeh handle rescaling the y-axis as needed.

        Returns
        -------
        float/list : appropriate y value(s) for writing a label
        """
        plot_range = self.refplot_range()
        height = abs(plot_range['y'][0] - plot_range['y'][1])

        if self.absorption:
            padding = -0.05 * height

        else:
            padding = 0.25 * height

        heights = self.refplot_linelist.data["intensities"] + padding
        return heights

    def update_label_heights(self):
        """Simple callback to move the labels if the spectrum panel is
        resized.
        """
        self.model.data.data["heights"] = self.label_height(self.model.x)

        if self.currently_identifying:
            start = self.p_spectrum.y_range.start
            end = self.p_spectrum.y_range.end
            lheight = 0.05 * (end - start)
            if self.absorption:
                lheight *= -1

            self.new_line_marker.data["y"][1] = (
                self.new_line_marker.data["y"][0] + lheight
            )

    # I could put the extra stuff in a second listener but the name of this
    # is generic, so let's just super() it and then do the extra stuff
    def model_change_handler(self, model):
        """If the `~fit` changes, this gets called to evaluate the fit and save
        the results.
        """
        super().model_change_handler(model)
        x, y = model.x, model.y
        linear_model = self.linear_model(model)

        model_data = self.model.data.data

        model_data["fitted"] = model.evaluate(x)
        model_data["nonlinear"] = y - linear_model(x)
        model_data["heights"] = self.label_height(x)
        model_data["lines"] = [wavestr(yy) for yy in y]

        if self.show_refplot:
            self.refplot_linelist.data['heights'] = self.refplot_label_height()

        eval_data = self.model.evaluation.data
        eval_data["nonlinear"] = (
            eval_data["model"] - linear_model(eval_data["xlinspace"])
        )

        domain = model.domain
        self.spectrum.data["wavelengths"] = model.evaluate(
            np.arange(domain[0], domain[1] + 1)
        )

        # If we recalculated the model while in the middle of identifying a
        # peak, keep the peak location but update the line choices
        if self.currently_identifying:
            peak = self.currently_identifying
            self.currently_identifying = False
            self.identify_line("i", 0, 0, peak=peak)

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
                logging.warning("Could not add line to data %s", wavelength)
                return

    def cancel_new_line(self, *args):
        """Handler for the 'Cancel' button in the line identifier"""
        logging.debug("cancel_new_line: %s", args)

        self.new_line_prompt.text = ""
        self.new_line_dropdown.options = []
        self.new_line_textbox.value = None
        self.set_currently_identifying(False)

    def add_line_to_data(self, peak, wavelength):
        """Add a new line to the ColumnDataSource and performs a new fit.

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
                self.model.y, self.model.evaluate(peak)[0]
            )

            lower_limit, upper_limit = sorted([lower_limit, upper_limit])

            if not lower_limit < wavelength < upper_limit:
                self.visualizer.show_user_message(
                    f"The value {wavelength} nm does not preserve a monotonic"
                    f" sequence of identified line wavelengths"
                )

                raise ValueError

        # Dummy values should be close to true values to avoid plot resizing
        new_data = {
            "x": [peak],
            "y": [wavelength],
            "mask": ["good"],
            "fitted": [wavelength],
            "nonlinear": [0],
            "heights": [self.label_height(peak)],
            "residuals": [0],
            "lines": [wavestr(wavelength)],
        }

        self.model.data.stream(new_data)
        self.model.perform_fit()

    @disable_when_identifying
    def identify_line(self, key, x, y, peak=None):
        """Identifies a peak near the cursor location and allows the user to
        provide a wavelength.
        """
        logging.debug("Identifying line: %s %s %s", key, x, y)

        if peak is None:
            x1, x2 = self.p_spectrum.x_range.start, self.p_spectrum.x_range.end
            fwidth = self.model.meta["fwidth"]

            interp_pixel = interp1d(
                self.spectrum.data["wavelengths"],
                range(len(self.spectrum.data["wavelengths"])),
            )

            pixel = interp_pixel(x)

            new_peaks = np.setdiff1d(
                self.model.meta["peaks"], self.model.x, assume_unique=True
            )

            # This will fail if no line was before user attempts to
            # identify a line, so do it only if there are new_peaks
            index = None
            if len(new_peaks):
                index = np.argmin(abs(new_peaks - pixel))

            # If we've clicked "close" to a real peak (based on viewport size),
            # then select that
            close_to_peak = self._close_to_peak(x, x1, x2, new_peaks, index)
            if close_to_peak:
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

            # Peak outside viewport.
            if not x1 < est_wave < x2:
                return

        else:
            # Evaluate always returns array.
            est_wave = self.model.evaluate(peak)[0]

        # Find all unidentified arc lines that this could be, maintaining
        # monotonicity
        all_lines = self.spectrum_linelist.wavelengths(
            in_vacuo=self.visualizer.ui_params.in_vacuo, units="nm"
        )

        lower_limit, upper_limit = get_closest(self.model.y, est_wave)

        possible_lines = [
            line for line in all_lines if lower_limit < line < upper_limit
        ]

        if possible_lines:
            selectable_lines = sorted(
                sorted(possible_lines, key=lambda x: abs(x - est_wave))[:9]
            )

            select_index = np.argmin(
                abs(np.asarray(selectable_lines) - est_wave)
            )

            self.new_line_dropdown.options = [
                wavestr(line) for line in selectable_lines
            ]

            self.new_line_dropdown.value = wavestr(
                selectable_lines[select_index]
            )

            self.new_line_dropdown.disabled = False

        else:
            self.new_line_dropdown.options = []
            self.new_line_dropdown.disabled = True

        self.new_line_prompt.text = f"Line at {peak:.1f} ({est_wave:.5f} nm)"
        start = self.p_spectrum.y_range.start
        end = self.p_spectrum.y_range.end

        lheight = (end - start) * (-0.05 if self.absorption else 0.05)
        # TODO: check what happens here in case of absorption -OS

        height = self.spectrum.data["spectrum"][int(peak + 0.5)]

        self.new_line_marker.data = {
            "x": [est_wave] * 2,
            "y": [height, height + lheight],
        }

        self.set_currently_identifying(peak)

    def _close_to_peak(self, x, x1, x2, new_peaks, index):
        """Checks if the click is close to a peak and if so returns True.

        Parameters
        ----------
        x : float
            The x-coordinate of the click.

        x1 : float
            The start of the x-axis range.

        x2 : float
            The end of the x-axis range.

        new_peaks : array-like
            The peaks that have not been identified yet.

        index : int
            The index of the peak that is closest to the click.

        Returns
        -------
        bool : True if the click is close to a peak, False otherwise.

        Notes
        -----
        If the index is None, then the click is not close to a peak. It's
        assumed this means no lines were found in the region.
        """
        # No index -> no peak
        if index is None:
            return False

        # Handle numpy arrays and lists.
        try:
            if not new_peaks.shape[0]:
                return False

        except AttributeError:
            if len(new_peaks) == 0:
                return False

        eval_peaks = self.model.evaluate(new_peaks[index])
        close_to_peak = abs(eval_peaks - x) < (0.025 * (x2 - x1))

        return close_to_peak

    @disable_when_identifying
    def delete_line(self, key, x, y):
        """Delete (not mask) from the fit the line nearest the cursor. This
        operates only on the spectrum panel.
        """
        logging.debug("delete_line: %s %s %s", key, x, y)

        index = np.argmin(abs(self.model.data.data["fitted"] - x))

        new_data = {
            col: list(values)[:index] + list(values)[index + 1:]
            for col, values in self.model.data.data.items()
        }

        self.model.data.data = new_data
        self.model.perform_fit()

    def clear_all_lines(self):
        """
        Called when the user clicks the "Clear all lines" button. Deletes
        all identified lines, performs a new fit.
        """
        self.model.data.data = {col: [] for col, values in self.model.data.data.items()}
        self.model.perform_fit()

    def identify_lines(self):
        """Called when the user clicks the "Identify Lines" button. This:
        1) Removes any masked points (user-masked or sigma-clipped) from the
           fit data
        2) Gets all the already-identified peaks that aren't in the fit
        3) Calculates the wavelengths of these peaks, based on the fit
        4) Matches those to unmatched lines in the linelist based on some
           criteria
        5) Adds new matches to the list
        6) Performs a new fit, triggering a plot update
        """
        linear_model = self.linear_model(self.model)
        dw = linear_model.c1 / np.diff(linear_model.domain)[0]
        matching_distance = abs(self.model.meta["fwidth"] * dw)

        all_lines = self.spectrum_linelist.wavelengths(
            in_vacuo=self.visualizer.ui_params.in_vacuo, units="nm"
        )

        good_data = {}
        for k, values in self.model.data.data.items():
            good_data[k] = [
                vv for vv, mask in zip(values, self.model.mask)
                if mask == "good"
            ]

        try:
            matches = match_sources(
                all_lines, good_data["y"], radius=0.01 * abs(dw)
            )

        except ValueError as err:  # good_data['y'] is empty
            logging.warning(
                "No lines identified, recieved ValueError: %s",
                err
            )

            unmatched_lines = all_lines

        else:
            unmatched_lines = [
                line for line, m in zip(all_lines, matches) if m == -1
            ]

        new_peaks = np.setdiff1d(
            self.model.meta["peaks"], good_data["x"], assume_unique=True
        )

        new_waves = self.model.evaluate(new_peaks)

        matches = match_sources(
            new_waves, unmatched_lines, radius=matching_distance
        )

        for peak, match in zip(new_peaks, matches):
            if match != -1:
                good_data["x"].append(peak)
                good_data["y"].append(unmatched_lines[match])
                good_data["mask"].append("good")
                good_data["fitted"].append(unmatched_lines[match])
                good_data["nonlinear"].append(0)
                good_data["heights"].append(self.label_height(peak))
                good_data["residuals"].append(0)
                good_data["lines"].append(wavestr(unmatched_lines[match]))
                print("NEW LINE", peak, unmatched_lines[match])

        self.model.data.data = good_data
        self.model.perform_fit()

    def set_new_line_textbox_value(self, attrib, old, new):
        """Update the value of the textbox related to the new line ID"""
        logging.debug("set_new_line_textbox_value: %s %s %s", attrib, old, new)

        if new != old:
            self.new_line_textbox.value = float(new)

    def handle_line_wavelength(self, attrib, old, new):
        """Handle user pressing Enter in the new line wavelength textbox."""
        logging.debug("handle_line_wavelength: %s %s %s", attrib, old, new)

        if new is None:
            return

        in_dropdown = wavestr(new) in self.new_line_dropdown.options
        if not in_dropdown:
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

        calibration_type = "vacuo" if self.ui_params.in_vacuo else "air"

        text = f"<b>Calibrating to wavelengths in {calibration_type}" f"</b>"

        self.reinit_panel.children[skip_lines] = bm.Div(
            text=text,
            align="center",
            stylesheets=dragons_styles(),
            sizing_mode="stretch_width",
        )

        self.widgets["in_vacuo"].disabled = True
        del self.num_atran_params

        self.absorption = absorption

    @property
    def meta(self):
        """The metadata for each fit as a list."""
        return [fit.meta for fit in self.fits]

    @property
    def image(self):
        """The image for each fit as a list."""
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
                if isinstance(key, dict) and "atran_linelist_pars" in key:
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
                align="start", styles={"font-weight":"bold"}, margin=(40,0,20,0))
            lst.insert((-self.num_atran_params), section_title)
        return lst

    def reconstruct_points_additional_work(self, data):
        """Reconstruct the initial points to work with."""
        super().reconstruct_points_additional_work(data)

        if data is not None:
            for i, _ in enumerate(self.fits):
                if self.returns_list:
                    this_dict = {k: v[i] for k, v in data.items()}

                else:
                    this_dict = data

                # spectrum update
                spectrum = self.panels[i].spectrum
                if self.absorption:
                    spectrum.data['spectrum'] = -this_dict["meta"]["spectrum"]
                else:
                    spectrum.data['spectrum'] = this_dict["meta"]["spectrum"]

                self.panels[i].spectrum_linelist = this_dict["meta"]["linelist"]
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
                    self.panels[i].update_refplot_name(this_dict["meta"]["refplot_name"])
                    self.panels[i].reset_view()
                except (AttributeError, KeyError):
                    pass

            # Reset panel axes
            for panel in self.panels:
                panel.reset_spectrum_axes()


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
