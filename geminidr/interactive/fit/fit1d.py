"""Module for 1-D fitting.

This module contains the classes and functions for performing 1-D fitting
of data.  This is used in the interactive module to allow the user to
interactively fit data and mask points that are not well fit.

The main class is :class:`~geminidr.interactive.fit.fit1d.Fit1DVisualizer`
which is a subclass of
:class:`~geminidr.interactive.interactive.PrimitiveVisualizer`.

This is also used across modules such as
:mod:`~geminidr.interactive.fit.wavecal` to manage specific fitting tasks.

The structure and nomenclature here is esoteric, so here's some help.

Visualizer:
    .panels = [list of Fit1DPanel instances]
        Fit1DPanel.model is the corresponding element in Visualizer.fits
    .fits = [list of InteractiveModel1D instances]
        InteractiveModel1D.fit is a fit_1D object
"""
from abc import ABC, abstractmethod
from collections import namedtuple
import inspect
import logging
import warnings

import numpy as np

from bokeh import models as bm, transform as bt
from bokeh.layouts import row, column
from bokeh.models import Div, Select, Spacer
from bokeh.plotting import figure
from bokeh import events

from geminidr.interactive import interactive
from geminidr.interactive.styles import dragons_styles
from geminidr.interactive.controls import Controller
from geminidr.interactive.interactive import (
    GIRegionModel,
    connect_region_model,
    GIRegionListener,
    RegionEditor,
    do_later,
    TabsTurboInjector,
    FitQuality,
)
from geminidr.interactive.interactive_config import interactive_conf
from gempy.library.astrotools import cartesian_regions_to_slices
from gempy.library.fitting import fit_1D


Glyph = namedtuple("Glyph", "name marker palette")


class InteractiveModel(ABC):
    """Base class for all interactive models, containing:
    (a) the parameters of the model
    (b) the parameters that control the fitting (e.g., sigma-clipping)
    (c) the way the fitting is performed
    (d) the input and output coordinates, mask, and weights
    """
    BandMasked = Glyph("excluded", "triangle", "lightsteelblue")
    UserMasked = Glyph("rejected (user)", "inverted_triangle", "lightskyblue")
    Good = Glyph("good", "circle", interactive_conf().bokeh_data_color)
    SigmaClipped = Glyph("rejected (sigma)", "square", "darksalmon")

    def __init__(self, mask_glyphs):
        """Initializes the model with default values."""

        self.listeners = []
        self.data = None
        self.quality = FitQuality.BAD  # no fit yet
        self.allow_poor_fits = True

        self.glyphs = [v for k, v in inspect.getmembers(self)
                       if isinstance(v, Glyph)]
        if mask_glyphs is None:
            self.extra_mask_names = []
        else:
            for k, v in mask_glyphs.items():
                self.glyphs.append(Glyph(k, *v))
            self.extra_mask_names = list(mask_glyphs.keys())

    def add_listener(self, listener):
        """Add a function to call when the model is updated

        Parameters
        ----------
        listener : function
            This should be a no-arg function and it will get called when the
            model is updated
        """
        if not callable(listener):
            raise TypeError("Listeners must be callables")

        self.listeners.append(listener)

    def notify_listeners(self):
        """Notify all the registered listeners of a change.

        This calls all our registered listener functions to let them know we
        have changed.
        """
        for listener in self.listeners:
            listener(self)

    @abstractmethod
    def perform_fit(self):
        """Perform the fit (abstract method, must override)."""

    @abstractmethod
    def evaluate(self, x):
        """Evaluate X. This is an abstract method that must be overridden.

        Parameters
        ----------
        x : :class:`~numpy.ndarray`
            X coordinate values
        """

    def mask_rendering_kwargs(self):
        """Get the marks and colors to use for the various point masks

        Returns
        -------
        dict : Returns a dict for bokeh that describes the markers and pallete
        """
        mask_types = [g.name for g in self.glyphs]
        return {
            "marker": bt.factor_mark("mask", [g.marker for g in self.glyphs], mask_types),
            "color": bt.factor_cmap("mask", [g.palette for g in self.glyphs], mask_types),
        }


class InteractiveModel1D(InteractiveModel):
    """InteractiveModel subclass for 1D models."""

    def __init__(
        self,
        fitting_parameters,
        domain,
        x=None,
        y=None,
        weights=None,
        mask=None,
        section=None,
        listeners=None,
        band_model=None,
        extra_masks=None,
        initial_fit=None,
        default_model=None,
        visualizer=None
    ):
        """Create base class with given parameters as initial model inputs.

        Parameters
        ----------
        x : :class:`~numpy.ndarray`
            list of x coordinate values

        y : :class:`~numpy.ndarray`
            list of y coordinate values

        mask : array of str
            array of mask names for each point

        section :
            Section based user mask

        listeners : list of functions
            Listeners for responding to updates in the model.

        band_model : :class:`~geminidr.interactive.interactive.BandModel
            Model for band selections

        extra_masks : dict of boolean arrays
            points to display but not use in fit

        initial_fit: fit_1D/None
            an initial fit to use if there are no data points

        default_model : callable
            function to evaluate model if self.fit is None
        """
        super().__init__(visualizer.mask_glyphs)

        if not listeners:
            listeners = []

        self.band_model = band_model
        if band_model:
            band_model.add_listener(
                Fit1DRegionListener(self.band_model_handler)
            )

        self.fitting_parameters = fitting_parameters
        self.domain = domain
        self.fit = initial_fit

        self.listeners = listeners
        self.default_model = default_model

        self.section = section
        self.data = bm.ColumnDataSource({"x": [], "y": [], "mask": []})

        if self.domain:
            xlinspace = np.linspace(*self.domain, 500)

        else:
            xlinspace = np.linspace(min(x), max(x), 500)
        self.populate_bokeh_objects(
            x, y, weights=weights, mask=mask, extra_masks=extra_masks
        )

        self.sigma_clip = (
            "sigma" in fitting_parameters and fitting_parameters["sigma"]
        )

        if len(x):
            self.perform_fit()
        self.evaluation = bm.ColumnDataSource(
            {"xlinspace": xlinspace, "model": self.evaluate(xlinspace)}
        )

    def set_function(self, function):
        """Set the fit function to use.

        This sets the function the `gempy.library.fitting.fit_1D` fitter will
        use to perform the data fit.  It's a helper method to pass the function
        down to the model.

        Parameters
        ----------
        fn : str
            Which fitter to use
        """
        self.fitting_parameters["function"] = function

    @property
    def regions(self):
        """Get the regions of the fitter.

        Returns
        -------
        tuple of tuples : `regions` of the model we are wrapping
        """
        return self.fitting_parameters["regions"]

    @regions.setter
    def regions(self, regions):
        """Set the regions in this fitter.

        This sets the regions.

        Parameters
        ----------
        regions : tuple of tuples
            regions to use in the fit
        """
        self.fitting_parameters["regions"] = regions

    def populate_bokeh_objects(
        self, x, y, weights, mask=None, extra_masks=None
    ):
        """Initializes bokeh objects like a coord structure with extra
        columns for ratios and residuals and setting up masking.

        Parameters
        ----------
        x : array of double
            x coordinate values

        y : array of double
            y coordinate values

        mask : array of str
            named mask for coordinates

        extra_masks : dict of boolean arrays
            points to display but not passed to the fit (for apertures in
            skyCorrect)
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

        # Check if the data has been fully masked; if so, raise a ValueError.
        if init_mask.size and init_mask.all():
            raise ValueError(
                "All data points are masked. Cannot perform a fit."
            )

        x = x[~init_mask]
        y = y[~init_mask]

        if weights is not None:
            weights = weights[~init_mask]

        # "section" is the valid section provided by the user,
        # i.e., points not in this region(s) are user-masked
        if self.section is None:
            mask = [self.Good.name] * len(x)

        else:
            user_mask = np.array(np.ones_like(x, dtype=bool))

            for slice_ in self.section:
                user_mask[slice_.start < x < slice_.stop] = False

            mask = list(np.where(user_mask, self.UserMasked.name, self.Good.name))

        # If we are showing masked points as user-masked, set that up here
        if extra_masks:
            for key, emask in extra_masks.items():
                mask_iter = zip(mask, emask[~init_mask])
                mask = [key if em else m for m, em in mask_iter]

        # Might put the variance in here for errorbars, but it's not needed
        # at the moment

        # need to setup the mask
        for i in np.arange(len(x)):
            if self.band_model is None or self.band_model.contains(x[i]):
                # User mask takes preference
                if mask[i] not in [self.UserMasked.name] + self.extra_mask_names:
                    mask[i] = self.Good.name

            elif mask[i] not in [self.UserMasked.name] + self.extra_mask_names:
                mask[i] = self.BandMasked.name

        bokeh_data = {"x": x, "y": y, "mask": mask}

        for extra_column in ("residuals", "ratio"):
            if extra_column in self.data.data:
                bokeh_data[extra_column] = np.zeros_like(y)

        self.data.data = bokeh_data

        self.weights = weights

    def band_model_handler(self):
        """Respond when the band model changes.

        When the band model has changed, we brute force a new band mask by
        checking each x coordinate against the band model for inclusion.  The
        band model handles the case where there are no bands by marking all
        points as included.
        """
        x_data = self.data.data["x"]
        mask = self.data.data["mask"].copy()

        for i in np.arange(len(x_data)):
            if self.band_model.contains(x_data[i]):
                # User mask takes preference
                if mask[i] not in [self.UserMasked.name] + self.extra_mask_names:
                    mask[i] = self.Good.name

            elif mask[i] not in [self.UserMasked.name] + self.extra_mask_names:
                mask[i] = self.BandMasked.name

        self.data.data["mask"] = mask
        # Band operations can come in through the keypress URL
        # so we defer the fit back onto the Bokeh IO event loop

        do_later(self.perform_fit)

    @property
    def x(self):
        """Maps x attribute internally to bokeh structures.

        Returns
        -------
        x_coords : `~numpy.ndarray`
            x coordinates as a numpy array of doubles.
        """
        return np.asarray(self.data.data["x"])

    @property
    def y(self):
        """Maps y attribute internally to bokeh structures.

        Returns
        -------
        y_coords : `~numpy.ndarray`
            y coordinates as a numpy array of doubles.
        """
        return np.asarray(self.data.data["y"])

    @property
    def mask(self):
        """Maps mask attribute internally to bokeh structures.

        Returns
        -------
        list of str : mask values
        """
        return self.data.data["mask"]

    @property
    def sigma(self):
        """Maps sigma attribute to
        :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.lsigma` and
        :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.hsigma`

        Returns
        -------
        sigma: float
            sigma value
        """
        if self.lsigma == self.hsigma:
            return self.lsigma

        return 0.5 * (self.lsigma + self.hsigma)  # do something better?

    @sigma.setter
    def sigma(self, value):
        """Set sigma attr, effectively setting both
        :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.lsigma` and
        :attr:`~geminidr.interactive.fit.fit1d.InteractiveModel1D.hsigma`.

        Parameters
        ----------
        value : float
            new value for sigma rejection
        """
        self.lsigma = self.hsigma = float(value)

    def perform_fit(self, *args):
        """Perform the fit.

        This performs the fit in the contained model, updates the mask,
        and recalculates the plots for residuals and ratio, if present.
        It then notifies all listeners that the data and model have
        changed so they can respond.
        """
        logging.debug("perform_fit: %s", args)

        # Note that band_mask is now handled by passing a region string to
        # fit_1D but we still use the band_mask for highlighting the affected
        # points.
        if self.sigma_clip:
            fitparms = {
                x: y
                for x, y in self.fitting_parameters.items()
                if x not in ["sigma"]
            }

        else:
            fitparms = {
                x: y
                for x, y in self.fitting_parameters.items()
                if x not in ["sigma_lower", "sigma_upper", "niter", "sigma"]
            }

        goodpix = np.array(
            [m not in [self.UserMasked.name] + self.extra_mask_names for m in self.mask]
        )

        self.quality = FitQuality.BAD

        if goodpix.sum():
            new_fit = fit_1D(
                self.y[goodpix],
                points=self.x[goodpix],
                domain=self.domain,
                weights=None
                if self.weights is None
                else self.weights[goodpix],
                **fitparms,
            )

            # For splines, "rank" is the number of spline pieces; for
            # Chebyshevs it's effectively the number of fitted points (max
            # order+1).
            rank = new_fit.fit_info["rank"]

            if rank > 0:
                if "params" in new_fit.fit_info:  # it's a polynomial
                    rank -= 1

                if rank >= fitparms["order"]:
                    self.quality = FitQuality.GOOD
                    self.fit = new_fit

                elif self.fit is None:
                    self.quality = FitQuality.BAD
                    self.fit = new_fit

                else:
                    # Modify the fit_1D object with a shift by ugly hacking
                    offset = np.mean(
                        self.y[goodpix] - self.evaluate(self.x[goodpix])
                    )
                    self.fit.offset_fit(offset)
                    self.fit.points = new_fit.points
                    self.fit.mask = new_fit.mask
                    self.quality = FitQuality.POOR  # else stay BAD

        if self.quality != FitQuality.BAD:  # don't update if it's BAD
            if "residuals" in self.data.data:
                self.data.data["residuals"] = self.y - self.evaluate(self.x)
            if "ratio" in self.data.data:
                with np.errstate(invalid="ignore", divide="ignore"):
                    self.data.data["ratio"] = self.y / self.evaluate(self.x)

            self.update_mask()

        self.notify_listeners()

    def update_mask(self):
        """Update the mask based on the current fit. The mask in the bokeh
        object is the size of the input data, but the mask returned by the
        fit is only the size of the non-user/band/other-masked pixels."""
        goodpix = np.array(
            [m not in [self.UserMasked.name] + self.extra_mask_names for m in self.mask]
        )

        mask = self.mask.copy()
        fit_mask = np.zeros_like(self.x, dtype=bool)
        if self.sigma_clip:
            # Now pull in the sigma mask
            fit_mask[goodpix] = self.fit.mask

        for i in range(fit_mask.size):
            if fit_mask[i] and mask[i] == self.Good.name:
                mask[i] = self.SigmaClipped.name

            elif not fit_mask[i] and mask[i] == self.SigmaClipped.name:
                mask[i] = self.Good.name

        self.data.data["mask"] = mask

    def evaluate(self, x):
        if self.fit is None:
            # fit_1D.evaluate() always returns an array so we need to also
            retval = self.default_model(x)
            if isinstance(retval, float):
                return np.array([retval])
            return retval
        return self.fit.evaluate(x)


class FittingParametersUI:
    """Manager for the UI controls and their interactions with the fitting
    model.
    """

    def __init__(self, vis, fit, fitting_parameters):
        """Class to manage the set of UI controls for the inputs to the fitting
        model.

        This sets up the controls for sigma rejection, iterations, etc., and
        manages them.  For instance, unchecking sigma causes the sigma related
        parameters to be disabled.

        Parameters
        ----------
        vis : :class:`~geminidr.interactive.fit.fit1d.Fit1DVisualizer`
            The visualizer related to these inputs

        fit : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model information for doing the 1-D fit

        fitting_parameters : dict
            The parameters for performing the fit using fit_1D.  These can be
            generated with :meth:`fit_1D.translate_params(params)` where params
            are the parameters for the primitive.  This will be passed down
            from the top level
            :class:`~geminidr.interactive.fit.fit1d.Fit1DVisualizer`
        """
        self.vis = vis
        self.fit = fit
        self.saved_sigma_clip = self.fit.sigma_clip
        self.fitting_parameters = fitting_parameters
        self.fitting_parameters_for_reset = {
            x: y for x, y in self.fitting_parameters.items()
        }

        if "function" in vis.ui_params.fields:
            function = fitting_parameters["function"]
            fn_allowed = [
                k for k in vis.ui_params.fields["function"].allowed.keys()
            ]

            # Dropdown for selecting fit_1D function
            self.function = Select(
                title="Fitting Function:",
                value=function,
                options=fn_allowed,
                width=200,
                stylesheets=dragons_styles(),
            )

            # pylint: disable=unused-argument
            def fn_select_change(attr, old, new):
                self.fit.set_function(new)
                self.fit.perform_fit()

            self.function.on_change("value", fn_select_change)

        else:
            # If the function is fixed
            function_text = (
                f"Fit Function: <b>"
                f"{fitting_parameters['function'].capitalize()}</b>"
            )

            self.function = bm.Div(
                text=function_text,
                min_width=100,
                max_width=202,
                sizing_mode="stretch_width",
                styles={
                    "color": "black",
                    "font-size": "115%",
                    "margin-top": "5px",
                },
                stylesheets=dragons_styles(),
            )

        self.description = self.build_description()

        def builder(ui_params, key, title):
            alt_keys = {"sigma_upper": "hsigma", "sigma_lower": "lsigma"}
            pkey = key
            if pkey not in ui_params.fields and key in alt_keys:
                pkey = alt_keys[key]
            field = ui_params.fields[pkey]

            step = 1 if field.dtype == int else 0.1

            field_min = field.min if hasattr(field, "min") else None

            if field_min and step > field_min > 0:
                step = field_min

            field_max = field.max if hasattr(field, "max") else None

            if key == "niter":
                # override this, min should be 1 as it is only used when sigma
                # is checked
                field_min = 1

            return interactive.build_text_slider(
                title=title,
                value=fitting_parameters[key],
                step=step,
                min_value=field_min,
                max_value=field_max,
                obj=fitting_parameters,
                attr=key,
                handler=fit.perform_fit,
                throttled=True,
                slider_width=128,
            )

        self.order_slider = builder(vis.ui_params, "order", "Order")

        self.sigma_upper_slider = builder(
            vis.ui_params, "sigma_upper", "Sigma (Upper)"
        )

        self.sigma_lower_slider = builder(
            vis.ui_params, "sigma_lower", "Sigma (Lower)"
        )

        self.niter_slider = builder(vis.ui_params, "niter", "Max Iterations")

        if "grow" in fitting_parameters:  # not all have them
            self.grow_slider = builder(vis.ui_params, "grow", "Grow")

        self.sigma_button = bm.CheckboxGroup(
            labels=["Sigma clip"],
            active=[0] if self.fit.sigma_clip else [],
            stylesheets=dragons_styles(),
        )

        self.sigma_button.on_change("active", self.sigma_button_handler)

        self.enable_disable_sigma_inputs()

        self.controls_column = self.build_column()

    def enable_disable_sigma_inputs(self):
        """This updates the state of the sliders depending on whether the sigma
        clipping checkbox is enabled or not.
        """
        # enable/disable sliders
        disabled = not self.fit.sigma_clip
        for child in self.niter_slider.children:
            child.disabled = disabled

        for child in self.sigma_upper_slider.children:
            child.disabled = disabled

        for child in self.sigma_lower_slider.children:
            child.disabled = disabled

        if hasattr(self, "grow_slider"):
            for child in self.grow_slider.children:
                child.disabled = disabled

    def build_column(self):
        """Builds a list with the components that belong to the Fit 1D
        Parameters column. The element's order inside the list represent the
        top-to-bottom order of the elements in the column.

        Return
        ------
        list : elements displayed in the column.
        """
        # TODO: The current setup doesn't have a clear way to add alt text to
        # various sliders and buttons in this interface. It would be good to
        # refactor this so that it accepts widgets as a list of tuples, for
        # example, so that adding in individual elements is not a fight with
        # the source.
        rejection_styles = {
            "color": "black",
            "font-size": "115%",
            "margin-top": "10px",
        }

        rejection_title = bm.Div(
            text="Rejection Parameters",
            min_width=100,
            max_width=202,
            sizing_mode="stretch_width",
            styles=rejection_styles,
            stylesheets=dragons_styles(),
        )

        if self.function:
            column_list = [
                self.function,
                self.order_slider,
                rejection_title,
                self.sigma_button,
                self.niter_slider,
                self.sigma_lower_slider,
                self.sigma_upper_slider,
            ]

        else:
            column_title_text = (
                f"Fit Function: <b>{self.vis.function_name.capitalize()}</b>"
            )

            col_div_styles = {
                "color": "black",
                "font-size": "115%",
                "margin-top": "5px",
            }

            column_title = bm.Div(
                text=column_title_text,
                min_width=100,
                max_width=202,
                sizing_mode="stretch_width",
                styles=col_div_styles,
                stylesheets=dragons_styles(),
            )

            column_list = [
                column_title,
                self.order_slider,
                rejection_title,
                self.sigma_button,
                self.niter_slider,
                self.sigma_lower_slider,
                self.sigma_upper_slider,
            ]

        if hasattr(self, "grow_slider"):
            column_list.append(self.grow_slider)

        return column_list

    def build_description(self, text=""):
        """Adds a description text to the fitting function parameters panel.

        Parameters
        ----------
        text : str
            Text that shows up in the description.

        Return
        ------
        :class:`~bokeh.models.Div`
            :class:`~bokeh.models.Div` component containing the short
            description.
        """
        description_styles = {
            "color": "black",
            "font-size": "115%",
            "margin-top": "10px",
        }

        return bm.Div(
            text=text,
            min_width=100,
            max_width=202,
            sizing_mode="stretch_width",
            styles=description_styles,
            width_policy="min",
            stylesheets=dragons_styles(),
        )

    def reset_ui(self):
        """Resets all the inputs to their original state.  This can be used by
        a reset button to undo any changes a user has made to the inputs.
        """
        self.fitting_parameters = {
            x: y for x, y in self.fitting_parameters_for_reset.items()
        }

        for key in ("order", "sigma_upper", "sigma_lower", "niter", "grow"):
            try:
                slider = getattr(self, f"{key}_slider")

            except AttributeError as err:
                logging.debug("AttributeError in reset_ui: %s", err)

            else:
                slider.children[0].value = self.fitting_parameters[key]
                slider.children[1].value = self.fitting_parameters[key]

        if self.function and hasattr(self.function, "value"):
            # if it's a selector and not a div...
            self.function.value = self.fitting_parameters["function"]

        self.sigma_button.active = [0] if self.saved_sigma_clip else []
        self.fit.perform_fit()

    def get_bokeh_components(self):
        """Return the bokeh components to be added with all the input widgets.

        Returns
        -------
        list : :class:`~bokeh.models.layout.LayoutDOM`
            List of bokeh components to add to the UI
        """
        return self.controls_column

    def sigma_button_handler(self, attr, old, new):
        """Handle the sigma clipping being turned on or off.

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
        logging.debug("sigma_button_handler: %s %s %s", attr, old, new)

        self.fit.sigma_clip = bool(new)
        self.enable_disable_sigma_inputs()
        self.fit.perform_fit()

    def sigma_slider_handler(self, val):
        """Handle the sigma clipping being adjusted.

        This will trigger a fit since the result may
        change.
        """
        logging.debug("sigma_slider_handler: %s", val)

        # If we're not sigma-clipping, we don't need to refit the model if
        # sigma changes
        if self.fit.sigma_clip:
            self.fit.perform_fit()


class InfoPanel:
    """Panel class for displaying information about the fit."""

    def __init__(self, enable_regions, enable_user_masking):
        """Build an informational panel to hold statistics about the fit.

        This shows the user the RMS of the fit to the data.  It also lists
        masking counts for the various mask types.

        Parameters
        ----------
        enable_regions : bool
            If True, will show region mask counts

        enable_user_masking : bool
            If True, will show user mask counts

        extra_masks : bool
            If True, will show the extra masks as provided to the visualizer
        """
        self.component = Div(text="", stylesheets=dragons_styles())
        self.enable_regions = enable_regions
        self.enable_user_masking = enable_user_masking

    def model_change_handler(self, model):
        """Respond to a model change by updating the displayed statistics.

        Parameters
        ----------
        model : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model that has changed.
        """
        try:
            rms_str = (
                "--" if np.isnan(model.fit.rms) else f"{model.fit.rms:.4f}"
            )
        except AttributeError:
            rms_str = "--"

        rms = (
            f'<div class="info_panel">'
            f'<div class="info_header">RMS: </div>'
            f'<div class="info_text">{rms_str}</div>'
        )

        band_count = model.mask.count(model.BandMasked.name)
        user_count = model.mask.count(model.UserMasked.name)
        fit_count = model.mask.count(model.SigmaClipped.name)

        extra_counts = dict()
        for extra_mask in model.extra_mask_names:
            extra_counts[extra_mask] = model.mask.count(extra_mask)

        total_count = model.x.size

        total = (
            f'<div class="info_header">Total Points: </div>'
            f'<div class="info_text">{total_count}</div>'
        )

        band = (
            f'<div class="info_header">Outside Region(s): </div>'
            f'<div class="info_text">{band_count}</div>'
            if self.enable_regions
            else ""
        )

        user = (
            f'<div class="info_header">User Masked: </div>'
            f'<div class="info_text">{user_count}</div>'
            if self.enable_user_masking
            else ""
        )

        fit = (
            f'<div class="info_header">Sigma Clipped:</div>'
            f'<div class="info_text">{fit_count}</div>'
        )

        extra_masks = "\n".join(
            [
                f'<div class="info_header">{em.capitalize()}:</div>'
                f'<div class="info_text">{extra_counts[em]}</div>'
                for em in model.extra_mask_names
            ]
        )

        self.component.update(
            text=rms + total + band + user + fit + extra_masks + "</div>"
        )


class Fit1DPanel:
    """Panel class for visualizing a 1-D fit. This includes the fit itself,
    and the data being fit, as a figure.

    This class is typically used in tabs within the interactive module. It
    is meant to handle one set of data being fit at a time.
    """

    def __init__(
        self,
        visualizer,
        fitting_parameters,
        domain=None,
        x=None,
        y=None,
        weights=None,
        idx=0,
        interactive_model_class=InteractiveModel1D,
        xlabel="x",
        ylabel="y",
        plot_width=600,
        plot_height=400,
        plot_residuals=True,
        plot_ratios=True,
        enable_user_masking=True,
        enable_regions=True,
        central_plot=True,
        extra_masks=None,
        default_model=None,
        initial_fit=None,
    ):
        """Panel for visualizing a 1-D fit, perhaps in a tab.

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
            True if we want the lower plot showing the differential between the
            data and the fit

        plot_ratios : bool
            True if we want the lower plot showing the ratio between the data
            and the fit

        enable_user_masking : bool
            True to enable fine-grained data masking by the user using bokeh
            selections

        enable_regions : bool
            True if we want to allow user-defind regions as a means of masking
            the data

        extra_masks : dict of boolean arrays
            points to display but not use in the fit

        default_model : callable
            function to evaluate model if self.fit is None
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
        self.xpoint = "x"
        self.ypoint = "y"
        self.p_main = None

        # prep params to clean up sigma related inputs for the interface
        # i.e. niter min of 1, etc.
        prep_fit1d_params_for_fit1d(fitting_parameters)

        # Avoids having to check whether this is None all the time
        band_model = GIRegionModel(domain=domain)

        self.model = interactive_model_class(
            fitting_parameters,
            domain,
            x,
            y,
            weights,
            band_model=band_model,
            extra_masks=extra_masks,
            default_model=default_model,
            initial_fit=initial_fit,
            visualizer=visualizer
        )
        # Hack for Telluric code to pass additional data to the GUI model
        # Should use visualizer.get_auxiliary_data(idx) once idx is properly
        # used!
        try:
            self.model.aux_data = self.aux_data
        except AttributeError:
            pass

        self.model.add_listener(self.model_change_handler)

        self.fitting_parameters_ui = FittingParametersUI(
            visualizer, self.model, fitting_parameters
        )

        controls_column = self.fitting_parameters_ui.get_bokeh_components()

        reset_button = bm.Button(
            label="Reset",
            align="center",
            button_type="warning",
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        self.reset_dialog = self.visualizer.make_ok_cancel_dialog(
            reset_button,
            "Reset will change all inputs for this tab back "
            "to their original values.  Proceed?",
            self.reset_dialog_handler,
        )

        controller_div = Div(
            margin=(20, 0, 0, 0),
            width=220,
            styles={"color": "gray", "padding": "5px"},
            stylesheets=dragons_styles(),
        )

        controls = column(
            *controls_column,
            reset_button,
            controller_div,
            width=220,
            stylesheets=dragons_styles(),
        )

        fig_column = self.build_figures(
            domain=domain,
            controller_div=controller_div,
            plot_residuals=plot_residuals,
            plot_ratios=plot_ratios,
        )

        # Initializing regions here ensures the listeners are notified of the
        # changes to regions.
        if fitting_parameters.get("regions") is not None:
            region_tuples = cartesian_regions_to_slices(
                fitting_parameters["regions"]
            )

            band_model.load_from_tuples(region_tuples)

        # TODO refactor? this is dupe from band_model_handler hacking it in
        # here so I can account for the initial state of the band model (which
        # used to be always empty)
        mask = [
            self.BandMasked.name if not band_model.contains(x) and m == self.model.Good.name else m
            for x, m in zip(self.model.x, self.model.mask)
        ]

        self.model.data.data["mask"] = mask
        self.model.perform_fit()

        if enable_regions:
            region_editor = RegionEditor(band_model)
            fig_column.append(region_editor.get_widget())

        col = column(
            *fig_column,
            stylesheets=dragons_styles(),
            sizing_mode="stretch_width",
        )

        col_order = [col, controls] if central_plot else [controls, col]
        self.component = row(
            *col_order,
            css_classes=["tab-content"],
            spacing=10,
            stylesheets=dragons_styles(),
            sizing_mode="stretch_width",
        )

    # pylint: disable=unused-argument
    def build_figures(
        self,
        domain=None,
        controller_div=None,
        plot_residuals=True,
        plot_ratios=True,
    ):
        """Construct the figures containing the various plots needed for this
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

        Returns
        -------
        fig_column : list
            list of bokeh objects with attached listeners
        """
        p_main, p_supp = fit1d_figure(
            width=self.width,
            height=self.height,
            xpoint=self.xpoint,
            ypoint=self.ypoint,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            model=self.model,
            enable_user_masking=self.enable_user_masking,
        )

        if self.enable_regions:
            self.model.band_model.add_listener(
                Fit1DRegionListener(self.update_regions)
            )

            connect_region_model(p_main, self.model.band_model)

        if self.enable_user_masking:
            mask_handlers = (
                self.mask_button_handler,
                self.unmask_button_handler,
            )

        else:
            mask_handlers = None

        Controller(
            p_main,
            None,
            self.model.band_model if self.enable_regions else None,
            controller_div,
            mask_handlers=mask_handlers,
            domain=domain,
            helpintrotext="While the mouse is over the upper plot, "
            "choose from the following commands:",
        )

        info_panel = InfoPanel(
            self.enable_regions,
            self.enable_user_masking,
        )

        self.model.add_listener(info_panel.model_change_handler)

        # self.add_custom_cursor_behavior(p_main)
        fig_column = [p_main, info_panel.component]
        if p_supp is not None:
            fig_column.append(p_supp)

        self.p_main = p_main

        # Do a custom padding for the ranges
        self.reset_view()

        return fig_column

    def reset_view(self):
        """This calculates the x and y ranges for the figure with some custom
        padding.

        This is used when we initially build the figure, but also as a listener
        for whenever the data changes.
        """
        if not hasattr(self, "p_main") or self.p_main is None:
            # This may be a subclass, p_main is not being stored so nothing to
            # reset
            return

        x_range = None
        y_range = None
        try:
            xdata = self.model.data.data[self.xpoint]
            ydata = self.model.data.data[self.ypoint]
        except (AttributeError, KeyError):
            pass

        else:

            def min_max_pad(data, default_min, default_max):
                if data is None or len(data) == 0:
                    return default_min, default_max, 0.0
                data_max = max(data)
                data_min = min(data)
                return data_min, data_max, 0.1 * (data_max - data_min)

            # if xdata or ydata are empty, we set some arbitrary values so the
            # UI is ok
            x_min, x_max, x_pad = min_max_pad(xdata, 0, 4000)
            if x_min != x_max:
                self.p_main.x_range.update(
                    start=x_min - x_pad, end=x_max + x_pad * 2
                )
            y_min, y_max, y_pad = min_max_pad(ydata, 0, 100)
            if y_min != y_max:
                self.p_main.y_range.update(
                    start=y_min - y_pad, end=y_max + y_pad
                )
        if x_range is not None:
            self.p_main.x_range = x_range
        if y_range is not None:
            self.p_main.y_range = y_range

    def reset_dialog_handler(self, result):
        """Reset fit parameter values.

        Parameters
        ----------
        result : bool
            This is the user response to an ok/cancel confirmation dialog.  If
            False, we do not reset.
        """
        if result:
            self.fitting_parameters_ui.reset_ui()

    def update_regions(self):
        """Update fitting regions"""
        self.model.regions = self.model.band_model.build_regions()

    def model_change_handler(self, model):
        """If the model changes, this gets called to evaluate the fit and save
        the results.

        Parameters
        ----------
        model : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model that changed.
        """
        model.evaluation.data["model"] = model.evaluate(
            model.evaluation.data["xlinspace"]
        )

    def mask_button_handler(self, x, y, mult):
        """Handler for the mask button.

        When the mask button is clicked, this method will find the selected
        data points and set the user mask for them.

        Parameters
        ----------
        x : float
            The pointer x coordinate

        y : float
            The pointer y coordinate

        mult : float
            The ratio for the X-axis vs Y-axis, so we can calculate "pixel
            distance"
        """
        indices = self.model.data.selected.indices
        if not indices:
            self._point_mask_handler(x, y, mult, "mask")

        else:
            # pylint: disable-next=no-member
            self.model.data.selected.update(indices=[])
            mask = self.model.mask.copy()
            for i in indices:
                mask[i] = self.model.UserMasked.name

            self.model.data.data["mask"] = mask
            self.model.perform_fit()

    def unmask_button_handler(self, x, y, mult):
        """Handler for the unmask button.

        When the unmask button is clicked, this method will find the selected
        data points and unset the user mask for them.

        Parameters
        ----------
        x : float
            The pointer x coordinate
        y : float
            The pointer y coordinate
        mult : float
            The ratio for the X-axis vs Y-axis, so we can calculate "pixel
            distance"
        """
        indices = self.model.data.selected.indices
        if not indices:
            self._point_mask_handler(x, y, mult, "unmask")

        else:
            x_data = self.model.x
            # pylint: disable-next=no-member
            self.model.data.selected.update(indices=[])
            mask = self.model.mask.copy()

            for i in indices:
                if mask[i] == self.model.UserMasked.name:
                    mask[i] = (
                        self.model.Good.name
                        if self.model.band_model.contains(x_data[i])
                        else self.model.BandMasked.name
                    )

            self.model.data.data["mask"] = mask
            self.model.perform_fit()

    def _point_mask_handler(self, x, y, mult, action):
        """Handler for the mask button.

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
            The ratio of the x axis vs the y axis, for calculating pixel
            distance

        action : str
            The type of masking action being done
        """
        dist = None
        sel = None
        xarr = self.model.data.data[self.xpoint]
        yarr = self.model.data.data[self.ypoint]
        mask = self.model.mask

        if action not in ("mask", "unmask"):
            action = None

        for i, (xdata, ydata) in enumerate(zip(xarr, yarr)):
            is_masked = (action == "mask") ^ (mask[i] == self.model.UserMasked.name)

            if action is None or is_masked:
                if xdata is not None and ydata is not None:
                    ddist = (x - xdata) ** 2 + ((y - ydata) * mult) ** 2
                    if dist is None or ddist < dist:
                        dist = ddist
                        sel = i

        if sel is not None:
            # we have a closest point, toggle the user mask
            if mask[sel] == self.model.UserMasked.name:
                if self.model.band_model.contains(xarr[sel]):
                    mask[sel] = self.model.Good.name

                else:
                    mask[sel] = self.model.BandMasked.name

            else:
                mask[sel] = self.model.UserMasked.name

        self.model.perform_fit()

    # TODO refactored this down from tracing, but it breaks
    # x/y tracking when the mouse moves in the figure for calculateSensitivity
    @staticmethod
    def add_custom_cursor_behavior(pointer):
        """Customize cursor behavior depending on which tool is active."""
        pan_start = """
            var mainPlot = document.getElementsByClassName('plot-main')[0];
            var active = [...mainPlot.getElementsByClassName('bk-active')];

            console.log(active);

            if ( active.some(e => e.title == "Pan") ) {
                Bokeh.cursor = 'move'; }
        """

        pan_end = """
            var mainPlot = document.getElementsByClassName('plot-main')[0];
            var elm = mainPlot.getElementsByClassName('bk-canvas-events')[0];

            Bokeh.cursor = 'default';
            elm.style.cursor = Bokeh.cursor;
        """

        mouse_move = """
            var mainPlot = document.getElementsByClassName('plot-main')[0];
            var elm = mainPlot.getElementsByClassName('bk-canvas-events')[0];
            elm.style.cursor = Bokeh.cursor;
        """

        pointer.js_on_event(events.MouseMove, bm.CustomJS(code=mouse_move))
        pointer.js_on_event(events.PanStart, bm.CustomJS(code=pan_start))
        pointer.js_on_event(events.PanEnd, bm.CustomJS(code=pan_end))


# pylint: disable=invalid-name
class Fit1DRegionListener(GIRegionListener):
    """Wrapper class so we can just detect when a band is finished. We don't
    want to do an expensive recalc as a user is dragging a band around. It
    creates a band listener that just updates on `finished`.

    Attributes
    ----------
    fn : function
        function to call when band is finished.
    """

    def __init__(self, function):
        """Initializes the Fit1DRegionListener.

        Parameters
        ----------
        fn : function
            function to call when band is finished.
        """
        self.fn = function

    def adjust_region(self, region_id, start, stop):
        pass

    def delete_region(self, region_id):
        self.fn()

    def finish_regions(self):
        self.fn()


class Fit1DVisualizer(interactive.PrimitiveVisualizer):
    """The generic class for interactive fitting of one or more 1D functions

    Attributes
    ----------
        reinit_panel:
            layout containing widgets that control the parameters affecting the
            initialization of the (x,y) array(s)

        reinit_button:
            the button to reconstruct the (x,y) array(s)

        tabs:
            layout containing all the stuff required for an interactive 1D fit

        submit_button:
            the button signifying a successful end to the interactive session

        config:
            the Config object describing the parameters are their constraints

        widgets:
            a dict of (param_name, widget) elements that allow the properties
            of the widgets to be set/accessed by the calling primitive.  So far
            I'm only including the widgets in the reinit_panel

        fits:
            list of InteractiveModel instances, one per (x,y) array
    """

    def __init__(
        self,
        data_source,
        fitting_parameters,
        modal_message=None,
        modal_button_label=None,
        tab_name_fmt=None,
        xlabel="x",
        ylabel="y",
        domains=None,
        title=None,
        primitive_name=None,
        filename_info=None,
        template="fit1d.html",
        help_text=None,
        recalc_inputs_above=False,
        ui_params=None,
        turbo_tabs=False,
        panel_class=Fit1DPanel,
        reinit_live=False,
        mask_glyphs=None,
        allow_skip=False,
        **kwargs,
    ):
        """Initializes the Fit1DVisualizer and its parent class.

        Parameters
        ----------
        data_source : dict or dict-returning function
            input data or the function to calculate the input data. The dict
            must have keys of "x" and "y" indicating the input data values,
            with each being an array or a list of arrays. There are also
            optional "weights" (array of weights), "*_mask" (additional input
            masks), and "meta" (any additional data)

        fitting_parameters :
                list of
                :class:`~geminidr.interactive.fit.fit1d.FittingParameters`
                or :class:`~geminidr.interactive.fit.fit1d.FittingParameters`
            Description of parameters to use for `fit_1d`.  These can be easily
            generated from the primitive's parameters with
            `fit_1D.translate_params(params)`

        modal_message : str
            If set, datapoint calculation is expected to be expensive and a
            'recalculate' button will be shown below the reinit inputs rather
            than doing it live.

        modal_button_label : str
            If set and if modal_message was set, this will be used for the
            label on the recalculate button.  It is not required.

        tab_name_fmt : callable
            Turns ext.id into a title for the tab name

        xlabel : str
            String label for X axis

        ylabel : str
            String label for Y axis

        domains : list
            List of domains for the inputs

        title : str
            Title for UI (Interactive <Title>)

        primitive_name : str
            Name of the primitive this tool is a visualizer for, for displaying
            in the UI

        template : str
            Name of the HTML template to use for the UI

        help_text : str
            HTML help text for popup help, or None to use the default

        recalc_inputs_above : bool
            If True, put the inputs for recalculating the data points above the
            plot horizontally, instead of left of the plot vertically

        ui_params : :class:`~geminidr.interactive.interactive.UIParams`
            Parameter set for user input

        turbo_tabs : bool
            If set, only the data for the tab being displayed will be sent to
            the browser. This can speed up the general responsiveness when
            there are many tabs, but introduces a small delay when the user
            switches between tabs.

        panel_class : :class:`TabPanel`
            The class of Panel to use in each tab. This allows specific
            operability for each primitive since most of the functions that do
            the work are methods of this class.

        reinit_live : bool
            If True, some buttons and parameters will recalculate the data
            points immediately.  If False, the reinit button will be disabled
            until the user clicks the "Recalculate" button. Default is False.

        mask_glyphs : dict/None
            glyphs for rendering additional masks

        allow_skip : bool
            add a button to allow exit and return unmodified data?
        """
        super().__init__(
            title=title,
            primitive_name=primitive_name,
            filename_info=filename_info,
            template=template,
            help_text=help_text,
            ui_params=ui_params,
            reinit_live=reinit_live,
            allow_skip=allow_skip,
        )
        self.layout = None
        self.recalc_inputs_above = recalc_inputs_above
        self.mask_glyphs = mask_glyphs

        if "pad_buttons" in kwargs:
            # Deprecation warning
            warnings.warn(
                "pad_buttons is no longer supported",
                DeprecationWarning,
            )

        # Keep a list of panels for access later
        self.panels = []

        # If we have a widget driving the modal dialog via it's enable/disable
        # state, store it in this so the recalc knows to re-enable the widget
        self.modal_widget = None

        # Make the panel with widgets to control the creation of (x, y) arrays

        # Create left panel
        reinit_widgets = self.make_widgets_from_parameters(ui_params)

        if reinit_widgets:
            # This should really go in the parent class, like submit_button
            if modal_message:
                if len(reinit_widgets) > 1:
                    self.reinit_button = bm.Button(
                        button_type="primary",
                        height=35,
                        width=202,
                        label=modal_button_label
                        if modal_button_label
                        else "Reconstruct points",
                        stylesheets=dragons_styles(),
                    )

                    self.reinit_button.on_click(self.reconstruct_points)
                    self.make_modal(self.reinit_button, modal_message)
                    reinit_widgets.append(self.reinit_button)
                    self.modal_widget = self.reinit_button

                    # If the "Reset" button we create here is clicked, we want
                    # the UI to reconstruct the input data.  To do this, we
                    # pass the reconstruct_points function as the optional
                    # extra handler
                    reset_reinit_button = self.build_reset_button(
                        extra_handler_fn=self.reconstruct_points
                    )
                    reinit_widgets.append(reset_reinit_button)

                elif len(reinit_widgets) == 1:
                    # pylint: disable=unused-argument
                    def kickoff_modal(attr, old, new):
                        self.reconstruct_points()

                    reinit_widgets[0].children[1].on_change(
                        "value", kickoff_modal
                    )

                    self.make_modal(reinit_widgets[0], modal_message)
                    self.modal_widget = reinit_widgets[0]
            else:
                reset_reinit_button = self.build_reset_button()
                reinit_widgets.append(reset_reinit_button)

            if recalc_inputs_above:
                self.reinit_panel = row(
                    *reinit_widgets, stylesheets=dragons_styles()
                )

            else:
                self.reinit_panel = column(
                    *reinit_widgets, stylesheets=dragons_styles()
                )

        else:
            # left panel with just the function selector (Chebyshev, etc.)
            self.reinit_panel = None  # column()

        if callable(data_source):
            self.reconstruct_points_fn = data_source
            data = data_source(ui_params=ui_params)

        else:
            data = data_source
            self.reconstruct_points_fn = None

        self.returns_list = isinstance(data["x"], list)

        # Some sanity checks now
        if self.returns_list:
            if not isinstance(fitting_parameters, list):
                raise ValueError(
                    "'data_source' is a list but "
                    "'fitting_parameters' is not"
                )

            if not (
                len(fitting_parameters) == len(data["x"]) == len(data["y"])
            ):
                raise ValueError("Different numbers of models and coordinates")

            if not all(
                [x.size == y.size for x, y in zip(data["x"], data["y"])]
            ):
                raise ValueError("Different (x, y) array sizes")

            self.nfits = len(fitting_parameters)

        else:
            if data["x"].size != data["y"].size:
                raise ValueError("Different (x, y) array sizes")

            self.nfits = 1

        kwargs.update({"xlabel": xlabel, "ylabel": ylabel})

        self.tabs = bm.Tabs(
            css_classes=["tabs"],
            sizing_mode="stretch_width",
            tabs=[],
            name="tabs",
            stylesheets=dragons_styles(),
        )

        if self.nfits == 1:
            turbo_tabs = False

        elif turbo_tabs:
            self.turbo = TabsTurboInjector(self.tabs)

        if tab_name_fmt is None:
            tab_name_fmt = lambda i: f"Extension {i+1}"

        for idx in range(self.nfits):
            if self.returns_list:
                this_dict = {k: v[idx] for k, v in data.items()}
                domain = domains[idx] if domains else None
                fitting_params = fitting_parameters[idx]

            else:
                this_dict = data
                domain = domains
                fitting_params = fitting_parameters

            # Glyphs need to be provided for extra masks, so we
            # know what they're called
            extra_masks = ({} if mask_glyphs is None else
                           {k: this_dict.pop(f"{k}_mask") for k in mask_glyphs})
            #for k in list(this_dict.keys()):
            #    if k.endswith("_mask"):
            #        extra_masks[k.replace("_mask", "")] = this_dict.pop(k)

            # making the fit panel
            tui = panel_class(
                self,
                fitting_params,
                domain=domain,
                idx=idx,
                **this_dict,
                **kwargs,
                extra_masks=extra_masks
            )

            # tab_name_fmt is expected to be a callable, which traditionally is
            # just an anonymous function passed to the relevant visualizer as
            # an argument.
            if not callable(tab_name_fmt):
                msg = (
                    f"tab_name_fmt must be callable, but got "
                    f"{type(tab_name_fmt)} ({tab_name_fmt})."
                )

                if isinstance(tab_name_fmt, str):
                    msg += (
                        " If you want to use a string, you should use "
                        "a lambda function like:\n"
                        "  tab_name_fmt = lambda i: f'Extension {i}'\n"
                        "or, if you want to ignore the argument, use:\n"
                        "  tab_name_fmt = lambda _: 'Extension'"
                    )

                raise ValueError(msg)

            if turbo_tabs:
                self.turbo.add_tab(tui.component, title=str(tab_name_fmt(idx)))

            else:
                tab = bm.TabPanel(
                    child=tui.component, title=str(tab_name_fmt(idx))
                )

                self.tabs.tabs.append(tab)

            self.fits.append(tui.model)
            self.panels.append(tui)

    def visualize(self, doc):
        """Start the bokeh document using this visualizer.

        This call is responsible for filling in the bokeh document with the
        user interface.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            bokeh document to draw the UI in
        """
        super().visualize(doc)
        col = column(
            self.tabs,
            stylesheets=dragons_styles(),
            sizing_mode="stretch_width",
        )

        buttons = (self.abort_button,)
        if self.skip_button is not None:
            buttons += (self.skip_button,)
        buttons += (self.submit_button,)

        for btn in buttons:
            btn.align = "end"
            btn.margin = (0, 5, 0, 0)
            btn.width = 212
            btn.sizing_mode = "scale_both"

        layout_ls = list()
        if self.filename_info:
            self.submit_button.align = "end"

            abort_submit_buttons = row(
                *buttons,
                align="end",
                stylesheets=dragons_styles(),
            )

            top_row = column(
                self.get_filename_div(),
                abort_submit_buttons,
                align="end",
                stylesheets=dragons_styles(),
                margin=(0, 0, 0, 10),
                css_classes=["top-row"],
            )

        else:
            top_row = column(
                *buttons,
                align="end",
                stylesheets=dragons_styles(),
                css_classes=["top-row"],
            )

        layout_ls.append(top_row)

        if self.reinit_panel is None:
            layout_ls.append(col)

        elif len(self.reinit_panel.children) <= 1 or self.recalc_inputs_above:
            panel_col = column(
                Spacer(sizing_mode="stretch_height", min_height=0),
                self.reinit_panel,
                sizing_mode="stretch_both",
            )

            # This overrides the "top_row" above. Using same variable since
            # it's decribing the same area, even with new functionality.
            top_row = row(
                panel_col,
                layout_ls.pop(0),
                sizing_mode="stretch_width",
                stylesheets=dragons_styles(),
                min_width=0,
            )

            layout_ls.append(top_row)
            layout_ls.append(col)

        else:
            layout_ls.append(
                row(
                    self.reinit_panel,
                    col,
                    sizing_mode="stretch_width",
                    stylesheets=dragons_styles(),
                )
            )

        self.layout = column(
            *layout_ls,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        doc.add_root(self.layout)

    def reconstruct_points_additional_work(self, data):
        """This method may contain any additional work the UI wants to do when
        reconstructing points.

        This is a no-op by default, but if we have subclasses, they are able to
        override this to add additional functionality with full access to the
        reconstructed dictionary of input data.  For example, wavecal can
        update the spectra being displayed based on the selected row.
        """

    def reconstruct_points(self):
        """Reconstruct the initial points to work with.

        This is expected to be expensive.  The core inputs are separated out in
        the UI as they are too slow to be interactive.  When a user is ready
        and submits updated core config parameters, this is what gets executed.
        The configuration is updated with the new values form the user.  The UI
        is disabled and the expensive function is wrapped in the bokeh Tornado
        event look so the modal dialog can display.
        """
        if self.modal_widget:
            self.modal_widget.disabled = True

        rollback_config = self.ui_params.values.copy()

        def function():
            """Top-level code to update the Config with the values from the
            widgets.
            """
            config_update = {
                k: (v.value if getattr(v, "show_value", True) else None)
                if hasattr(v, "value")
                else bool(v.active)
                for k, v in self.widgets.items()
            }
            self.ui_params.update_values(**config_update)

        self.do_later(function)

        if self.reconstruct_points_fn is not None:

            def rfn():
                data = None
                try:
                    data = self.reconstruct_points_fn(ui_params=self.ui_params)

                # pylint: disable-next=broad-except
                except Exception as err:
                    # something went wrong, let's revert the inputs handling
                    # immediately to specifically trap the
                    # reconstruct_points_fn call
                    self.ui_params.update_values(**rollback_config)
                    self.show_user_message(
                        f"Unable to build data from inputs due to exception "
                        f"({err.__class__.__name__})\nreverting"
                    )

                if data is not None:
                    for i, fit in enumerate(self.fits):
                        extra_masks = {}

                        if self.returns_list:
                            this_dict = {k: v[i] for k, v in data.items()}

                        else:
                            this_dict = data

                        for k in list(this_dict.keys()):
                            if k.endswith("_mask"):
                                extra_masks[
                                    k.replace("_mask", "")
                                ] = this_dict.pop(k)

                        fit.populate_bokeh_objects(
                            this_dict["x"],
                            this_dict["y"],
                            this_dict.get("weights"),
                            extra_masks=extra_masks,
                        )

                        fit.perform_fit()

                    self.reconstruct_points_additional_work(data)

                if self.modal_widget:
                    self.modal_widget.disabled = False

                for pnl in self.panels:
                    pnl.reset_view()

            self.do_later(rfn)

    def results(self):
        """Get the results of the interactive fit.

        This gets the list of `~gempy.library.fitting.fit_1D` fits of the data
        to be used by the caller.

        Returns
        -------
        list of `~gempy.library.fitting.fit_1D`
        """
        if self.return_fit:
            return [fit.fit for fit in self.fits]


def prep_fit1d_params_for_fit1d(fit1d_params):
    """In the UI, which relies on `fit1d_params`, we constrain `niter` to 1 at
    the low end and separately have a `sigma` boolean checkbox.

    To support the `sigma` checkbox, here we remap the inputs based on the
    value of `niter`.  If `niter` is 0, this tells us no `sigma` rejection is
    desired.  So, in that case, we set `sigma` to False.  We then set `niter`
    to 1 for the UI to work as desired.  Were `niter` passed in as `1`
    originally, it would remain `1` but `sigma` would be set to `True`.

    The UI will disable all the sigma related inputs when `sigma` is set to
    False.  It will also exclude them from the parameters sent to `fit_1d`.

    Parameters
    ----------
    fit1d_params : dict
        Dictionary of parameters for the UI and the `fit_1d` fitter, modified
        in place
    """
    # If niter is 0, set sigma to None and niter to 1
    if "niter" in fit1d_params and fit1d_params["niter"] == 0:
        # we use a min of 1 for niter, then remove the sigmas to clue the UI in
        # that we are not sigma clipping.  If we disable sigma clipping, niter
        # will be disabled in the UI.  Allowing a niter selection of 0 with
        # sigma clipping turned on is counterintuitive for the user.
        fit1d_params["niter"] = 1
        fit1d_params["sigma"] = False

    else:
        fit1d_params["sigma"] = True


# pylint: disable=unused-argument
def fit1d_figure(
    width=None,
    height=None,
    xpoint="x",
    ypoint="y",
    xline="xlinspace",
    yline="model",
    xlabel=None,
    ylabel=None,
    model=None,
    fit_line_legend=None,
    plot_ratios=True,
    plot_residuals=True,
    enable_user_masking=True,
):
    """Generic function to produce bokeh objects for the main scatter/fit
    plot and the residuals and/or ratios plot. Listeners are not added here.

    Parameters
    ----------
    width : int
        width of the plots

    height : int
        height of the main plot (ratios/residuals are half-height)

    xpoint, ypoint : str
        column names in model.data containing x and y data for points

    xline, yline : str
        column names in model.evaluation containing x and y data for line
        representing the fit

    xlabel, ylabel : str
        label for axes of main plot

    model : InteractiveModel1D
        object containing the fit information

    fit_line_legend : str/None
        string to include in legend for fit line

    plot_ratios : bool
        make a ratios plot?

    plot_residuals : bool
        make a residuals plot?

    enable_user_masking : bool
        is user masking enabled? If so, additional tools are required

    Returns
    -------
    tuple : (Figure, Tabs/Figure/None)
        the main plotting figure and the ratios/residuals plot
    """

    tools = "pan,wheel_zoom,box_zoom,reset"

    if enable_user_masking:
        tools += ",lasso_select,box_select,tap"

    p_main = figure(
        height=height,
        min_width=200,
        title="Fit",
        x_axis_label=xlabel,
        y_axis_label=ylabel,
        tools=tools,
        output_backend="webgl",
        min_border_left=80,
        stylesheets=dragons_styles(),
        sizing_mode="stretch_width",
    )

    p_main.scatter(
        x=xpoint,
        y=ypoint,
        source=model.data,
        size=5,
        legend_field="mask",
        **model.mask_rendering_kwargs(),
    )

    kwargs = {"legend_label": fit_line_legend} if fit_line_legend else {}
    p_main.line(
        x=xline,
        y=yline,
        source=model.evaluation,
        line_width=3,
        color="crimson",
        **kwargs
    )

    if plot_residuals:
        # x_range is linked to the main plot so that zooming tracks between
        # them
        p_resid = figure(
            height=height // 2,
            min_width=400,
            title="Fit Residuals",
            x_axis_label=xlabel,
            y_axis_label="Delta",
            tools="pan,box_zoom,reset",
            output_backend="webgl",
            x_range=p_main.x_range,  # y_range=None,
            min_border_left=80,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        connect_region_model(p_resid, model.band_model)

        # Initializing this will trigger the residuals to be calculated.
        model.data.data["residuals"] = np.zeros_like(model.x)

        p_resid.scatter(
            x=xpoint,
            y="residuals",
            source=model.data,
            size=5,
            legend_field="mask",
            **model.mask_rendering_kwargs(),
        )

    if plot_ratios:
        p_ratios = figure(
            height=height // 2,
            min_width=400,
            title="Fit Ratios",
            x_axis_label=xlabel,
            y_axis_label="Ratio",
            tools="pan,box_zoom,reset",
            output_backend="webgl",
            x_range=p_main.x_range,
            min_border_left=80,
            sizing_mode="stretch_width",
            stylesheets=dragons_styles(),
        )

        connect_region_model(p_ratios, model.band_model)

        # Initializing this will trigger the ratios to be calculated
        model.data.data["ratio"] = np.ones_like(model.x)

        p_ratios.scatter(
            x=xpoint,
            y="ratio",
            source=model.data,
            size=5,
            legend_field="mask",
            **model.mask_rendering_kwargs(),
        )

    if plot_residuals and plot_ratios:
        tabs = bm.Tabs(
            tabs=[], sizing_mode="stretch_width", stylesheets=dragons_styles()
        )

        tabs.tabs.append(bm.TabPanel(child=p_resid, title="Residuals"))
        tabs.tabs.append(bm.TabPanel(child=p_ratios, title="Ratios"))
        return p_main, tabs

    elif plot_residuals:
        return p_main, p_resid

    elif plot_ratios:
        return p_main, p_ratios

    return p_main, None
