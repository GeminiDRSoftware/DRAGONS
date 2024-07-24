import numpy as np

from bokeh import models as bm
from bokeh.models import Div
from bokeh.layouts import row, column
from bokeh.plotting import figure

from gempy.library import astrotools as at

from geminidr.interactive.interactive import (
    GIRegionModel, RegionEditor, connect_region_model, FitQuality, do_later)
from ..controls import Controller
from .fit1d import (
    Fit1DPanel, Fit1DVisualizer, InfoPanel, fit1d_figure, USER_MASK_NAME,
    BAND_MASK_NAME, SIGMA_MASK_NAME, Fit1DRegionListener,
    prep_fit1d_params_for_fit1d, FittingParametersUI, InteractiveModel, InteractiveModel1D)
from ..styles import dragons_styles

from gempy.library.telluric_models import Planck
from gempy.library.telluric import parse_magnitude, VEGA_INFO

from .help import TELLURIC_CORRECT_HELP_TEXT


class TelluricInteractiveModel1D(InteractiveModel1D):
    #MASK_TYPE = [BAND_MASK_NAME, USER_MASK_NAME, 'good', SIGMA_MASK_NAME] + INPUT_MASK_NAMES
    #MARKERS = ['triangle', 'inverted_triangle', 'circle', 'square', 'inverted_triangle']
    #PALETTE = ['lightsteelblue', 'lightskyblue', 'black', 'darksalmon', 'red']  # Category10[4]

    # Mapping widget titles to the UI params names
    # TODO: should be in parent class in case it's needed
    WIDGET_MAPPING = {'Max Iterations': 'niter',
                      'Sigma (Lower)': 'sigma_lower',
                      'Sigma (Upper)': 'sigma_upper',
                      'Grow': 'grow'}

    extra_masks = ['stellar']

    def __init__(self, fitting_parameters, domain, x=None, y=None, weights=None, mask=None,
                 section=None, listeners=None, band_model=None, extra_masks=None,
                 visualizer=None):

        # I think these class attributes of InteractiveModel1D should be
        # defined with INPUT_MASK_NAMES = [] (so the "[:-1]" isn't needed)
        # and things like SIGMA_MASK_NAME should also be attributes
        self.MASK_TYPE = self.MASK_TYPE[:-1] + self.extra_masks
        self.MARKERS = self.MARKERS[:-1] + ['inverted_triangle']
        self.PALETTE = self.PALETTE[:-1] + ['red']

        InteractiveModel.__init__(self)

        # We keep 'weights' in the call signature for consistency
        if weights is not None:
            raise RuntimeWarning("weights has been set but this parameter is not used")

        # Make the TelluricSpectrumCollection accessible to all models
        # and when a model in one panel is changed, it has the ability
        # to affect the other panels. Also we want to know which panel
        # this model is part of.
        self.visualizer = visualizer

        # TODO: This looks like it should be handled by the "idx" argument
        # of Fit1DPanel, which is unused
        # This code is run *before* the object gets attached to the Visualizer
        self.my_fit_index = len(visualizer.fits)

        if not listeners:
            listeners = []
        self.band_model = band_model
        if band_model:
            band_model.add_listener(Fit1DRegionListener(self.band_model_handler))

        self.fitting_parameters = fitting_parameters
        self.domain = domain
        self.fit = None  # a SingleTelluricModel() instance, eventually
        self.listeners = listeners

        self.section = section
        self.data = bm.ColumnDataSource({'x': [], 'y': [], 'mask': []})

        self.populate_bokeh_objects()

        self.sigma_clip = "sigma" in fitting_parameters and fitting_parameters["sigma"]

        # Can't use default 'xlinspace', since we need to evaluate each pixel
        # We have to modify the 'data' attribute directly since the figure
        # already has a reference to self.model.evaluation, which we can't reset
        # We only do the first fit after all panels have been created so let's
        # create a dummy 'model' dataset here.
        #
        # We need this, despite the existence of aux_data, because the fit1d_figure
        # expects it!
        self.evaluation = bm.ColumnDataSource({'xlinspace': self.x,
                                               'model': np.zeros_like(self.x),
                                               'continuum': np.zeros_like(self.x)})

    def perform_fit(self, *args):
        """Needs its own fitting method due to extra complexity"""
        # The code here needs to do several things:
        #     1. Update the Calibrator with the fitting parameters
        #     2. Update the Calibrator's user_mask with the UI mask
        #     3. Perform the fit by calling Calibrator.perform_fit()
        #     4. Update the UI mask with the sigma-clip mask from the fit
        #     5. Update "residuals" (and/or "ratio")
        #     6. Compute an rms for the InfoPanel (since this isn't done
        #        automatically as it's not a fit_1D object)
        #     7. Assess whether the fit is fully constrained ("good")
        #     8. Update the widgets in the UI's other panels to match those
        #        in this panel [edit: update the widgets in all UI panels
        #        with the values in the Calibrator]
        #
        # This is *only* called if widgets in the right-hand panel change
        # We don't want to recursively fit as we update the other panels
        # due to changes in this panel
        #
        # The reconstruct_points() method will build a new intrinsic_spectrum
        # (which isn't instantaneous) do we don't want to call that
        #
        # Because I'm not touching the core DRAGONS code, I have to do some
        # updates here (e.g., of the mask) that could normally be done by
        # the listener
        vis = self.visualizer

        # This prevents the code trying to re-fit when we update the fitting
        # widgets in the other panels to match this one
        if vis.actively_fitting:
            return

        vis.actively_fitting = True
        vis.modal_widget.disabled = True

        # This code is probably standard for all InteractiveModel1Ds
        # Update the parameters of the Calibrator
        for k, v in self.fitting_parameters.items():
            if k in vis.calibrator.fit_params:
                vis.calibrator.fit_params[k][self.my_fit_index] = v
                print(f"Updating {k} to value {v}")

        # Update the user mask. Since this method is called as soon as
        # a mask is updated in one panel, we know that changes could only
        # have happened to this panel
        mask = [m not in ('good', SIGMA_MASK_NAME) for m in self.mask]
        vis.calibrator.user_mask[self.my_fit_index][~vis.calibrator.mask[self.my_fit_index]] = np.asarray(mask)

        # This is where we diverge because of the complexity of Telluric
        def fn():
            # Perform the fit
            m_final, new_mask = vis.calibrator.perform_fit(
                self.my_fit_index, sigma_clipping=self.sigma_clip)
            #for k, v in zip(m_final.param_names, m_final.parameters):
            #    print(k, v)
            m_final.update_individual_models()

            # Ensure points are correctly marked as "sigma" or "good"
            # based on sigma-clipping
            start_pix = 0
            for i, (fit, nparams) in enumerate(zip(vis.fits, m_final.nparams)):
                ngoodpix = (~vis.calibrator.mask[i]).sum()
                # Obviously this naming is ridiculous!
                fit.fit = m_final.models[i]
                fit.fit.mask = new_mask[start_pix:start_pix+ngoodpix]
                #print(i, start_pix, start_pix+ngoodpix, new_mask.size)
                fit.update_mask()
                start_pix += ngoodpix

                # Assume that the PCA parameters are being constrained by other tabs
                if nparams <= fit.mask.count('good'):
                    fit.quality = FitQuality.GOOD
                else:
                    fit.quality = FitQuality.BAD

            # Some of this code is very similar to what's in the standard
            # perform_fit() post-fit, but we have to unpack the mask.
            # vis.fits is a list of TelluricModel1D instances
            for fit, tspek in zip(vis.fits, vis.calibrator.spectra):
                if 'residuals' in fit.data.data:
                    fit.data.data['residuals'] = fit.y - fit.evaluate(fit.x)
                if 'ratio' in fit.data.data:
                    raise NotImplementedError("We don't plot the ratio")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        fit.data.data['ratio'] = fit.y / fit.evaluate(fit.x)

                # Since the fit a TelluricWithContinuum instance, not a fit_1D,
                # it doesn't have an rms attribute. We can't ask the TWC to
                # calculate it because the instance doesn't know about the data
                # so instead we calculate it here and poke it into the attribute
                rms_pixels = np.array([m == 'good' for m in fit.mask])
                fit.fit.rms = (fit.y - fit.evaluate(fit.x))[rms_pixels].std()

                fit.notify_listeners()

            # Set *all* fits to BAD if we don't have enough points to constrain
            total_good = np.sum(fit.data.data['mask'].count('good')
                                for fit in vis.fits)
            if total_good < len(m_final.parameters):
                for fit in self.fits:
                    fit.quality = FitQuality.BAD

            # Store this since the individual tabs don't have fits
            vis.fitted_model = m_final

            # Update the LSF parameters. We update the slider but, since
            # the callback in "value_throttled" and not "value", it's not
            # called when we set the value here. We have to call it manually.
            for k, v in vis.widgets.items():
                if k in vis.calibrator.lsf_parameter_bounds:
                    old = v.value
                    v.value = getattr(m_final, k).value
                    for callback in v._callbacks.get('value_throttled', []):
                        callback("value", old, v.value)

            vis.modal_widget.disabled = False

        vis.do_later(fn)

        # We need to copy the fitting parameters from this panel to the other
        # panels. They've already been updated in the Calibrator. We can do
        # this outside the do_later(); in fact, we probably need to since it
        # has to be done before vis.actively_fitting is unset. Here we update
        # the NumericInput for slider/textbox combos, which automatically
        # updates the Slider.
        #
        # This code actually modifies all the fits, using the values stored in
        # the Calibrator. This allows the Calibrator to modify the fitting
        # parameters if they appear to be invalid for some reason.
        for i, panel in enumerate(vis.panels):
            for widget in vis.panels[i].fitting_parameters_ui.controls_column[-5:]:
                try:
                    title = widget.children[0].title
                except AttributeError:
                    widget.active = [0] if self.sigma_clip else []
                else:
                    try:
                        key = self.WIDGET_MAPPING[title]
                    except KeyError:
                        print(f"Cannot find key {title}")
                    else:
                        try:
                            value = vis.calibrator.fit_params[key][i]
                        except KeyError:
                            print(f"Cannot find value for key {key}")
                        else:
                            widget.children[-1].value = value

        vis.actively_fitting = False

    def populate_bokeh_objects(self):
        """
        THIS IS HERE ONLY BECAUSE THE PARENT METHOD HAS ITS OWN
        "INPUT_MASK_NAMES" which is not an attribute;
        IT CAN GO ONCE ADDITIONAL MASKS ARE PROPERLY ABSTRACTED!
        Actually, INPUT_MASK_NAMES should be self.extra_masks

        Work needs to be done on the core code which requires extra mask names
        to be defined there in INPUT_MASK_NAMES
        """
        data = self.visualizer.get_data(self.my_fit_index)
        x = data['x']
        y = data['y']
        init_mask = data['mask']
        extra_masks = {k: data[f"{k}_mask"] for k in self.extra_masks}
        super().populate_bokeh_objects(x, y, None, mask=init_mask,
                                       extra_masks=extra_masks)

        mask = self.mask
        if extra_masks is not None:
            for k, v in extra_masks.items():
                mask = [k if vv else m for m, vv in zip(mask, v[~init_mask])]
        for i in np.arange(len(mask)):
            if self.band_model.contains(x[i]):
                # User mask takes preference
                if mask[i] not in [USER_MASK_NAME] + self.extra_masks:
                    mask[i] = 'good'
            elif mask[i] not in [USER_MASK_NAME] + self.extra_masks:
                mask[i] = BAND_MASK_NAME
        self.data.data['mask'] = mask

    def band_model_handler(self):
        """
        THIS IS HERE ONLY BECAUSE THE PARENT METHOD HAS ITS OWN
        "INPUT_MASK_NAMES"; IT CAN GO ONCE ADDITIONAL MASKS ARE
        PROPERLY ABSTRACTED!

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
                if mask[i] not in [USER_MASK_NAME] + self.extra_masks:
                    mask[i] = 'good'
            elif mask[i] not in [USER_MASK_NAME] + self.extra_masks:
                mask[i] = BAND_MASK_NAME
        self.data.data['mask'] = mask
        # Band operations can come in through the keypress URL
        # so we defer the fit back onto the Bokeh IO event loop

        do_later(self.perform_fit)

    def evaluate(self, x):
        return self.fit(x)


class TelluricPanel(Fit1DPanel):
    """
    Copying Fit1DPanel.__init__() since we need to abstract InteractiveModel1D

    While that method calls perform_fit() when each panel is created, we can
    stop this by setting visualizer.actively_fitting=True

    So this should basically just be something like:
        visualizer.actively_fitting = True
        self.aux_data = bm.ColumnDataSource(visualizer.get_auxiliary_data(idx))
        super().__init__(model_class=TelluricInteractiveModel1D,
                         *args, **kwargs)
        visualizer.actively_fitting = False
    """
    def __init__(self, visualizer, fitting_parameters, domain=None,
                 x=None, y=None, weights=None, idx=0, xlabel='x', ylabel='y',
                 plot_width=600, plot_height=400, plot_residuals=True, plot_ratios=True,
                 enable_user_masking=True, enable_regions=True, central_plot=True,
                 extra_masks=None):
        self.visualizer = visualizer
        self.index = idx

        self.width = plot_width
        self.height = plot_height
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.enable_regions = enable_regions
        self.enable_user_masking = enable_user_masking
        self.xpoint = 'x'
        self.ypoint = 'y'
        self.p_main = None

        prep_fit1d_params_for_fit1d(fitting_parameters)

        # Avoids having to check whether this is None all the time
        band_model = GIRegionModel(domain=domain)
        self.model = TelluricInteractiveModel1D(
            fitting_parameters, domain, x, y, weights,
            band_model=band_model, visualizer=visualizer
        )
        self.model.add_listener(self.model_change_handler)

        # Needs to be a new ColumnDataSource because it includes
        # init-masked points, which the regular bokeh CDSs don't.
        # Again, this should be using "idx"
        self.model.aux_data = bm.ColumnDataSource(
            self.visualizer.get_auxiliary_data(self.model.my_fit_index)
        )

        self.fitting_parameters_ui = FittingParametersUI(visualizer, self.model,
                                                         fitting_parameters)
        controls_column = self.fitting_parameters_ui.get_bokeh_components()

        reset_button = bm.Button(label="Reset", align='center',
                                 button_type='warning', width_policy='min')
        self.reset_dialog = self.visualizer.make_ok_cancel_dialog(
            reset_button, 'Reset will change all inputs for this tab back '
            'to their original values.  Proceed?', self.reset_dialog_handler)

        controller_div = Div(margin=(20, 0, 0, 0), width=220,
                             styles={"color": "gray", "padding": "5px"})
        controls = column(*controls_column, reset_button, controller_div,
                          width=220)

        fig_column = self.build_figures(domain=domain, controller_div=controller_div,
                                        plot_residuals=plot_residuals,
                                        plot_ratios=plot_ratios)

        # Initializing regions here ensures the listeners are notified of the region(s)
        if fitting_parameters.get("regions") is not None:
            region_tuples = at.cartesian_regions_to_slices(fitting_parameters["regions"])
            band_model.load_from_tuples(region_tuples)

        # TODO refactor? this is dupe from band_model_handler
        # hacking it in here so I can account for the initial
        # state of the band model (which used to be always empty)
        mask = [BAND_MASK_NAME if not band_model.contains(x) and m == 'good' else m
                for x, m in zip(self.model.x, self.model.mask)]
        self.model.data.data['mask'] = mask
        #self.model.perform_fit()

        if enable_regions:
            region_editor = RegionEditor(band_model)
            fig_column.append(region_editor.get_widget())
        col = column(*fig_column)
        col.sizing_mode = 'scale_width'

        col_order = [col, controls] if central_plot else [controls, col]
        self.component = row(*col_order, css_classes=["tab-content"],
                             spacing=10)

    def build_figures(self, domain=None, controller_div=None,
                      plot_residuals=True, plot_ratios=True):
        p_main, p_supp = fit1d_figure(width=self.width, height=self.height,
                                      xpoint=self.xpoint, ypoint=self.ypoint,
                                      xlabel=self.xlabel, ylabel=self.ylabel,
                                      model=self.model, plot_ratios=False,
                                      enable_user_masking=True)
        p_main.line(x='waves', y='continuum', source=self.model.aux_data,
                    line_width=2, color='blue')

        if self.enable_regions:
            self.model.band_model.add_listener(Fit1DRegionListener(self.update_regions))
            connect_region_model(p_main, self.model.band_model)

        if self.enable_user_masking:
            mask_handlers = (self.mask_button_handler,
                             self.unmask_button_handler)
        else:
            mask_handlers = None

        Controller(p_main, None, self.model.band_model if self.enable_regions else None, controller_div,
                   mask_handlers=mask_handlers, domain=domain, helpintrotext=
                   "While the mouse is over the upper plot, "
                   "choose from the following commands:")

        info_panel = InfoPanel(self.enable_regions, self.enable_user_masking,
                               extra_masks=self.model.extra_masks)
        self.model.add_listener(info_panel.model_change_handler)

        # Plot showing the intrinsic spectrum
        try:
            intrinsic_units = self.model.aux_data.data['intrinsic_spectrum'].unit
        except AttributeError:
            intrinsic_units = 'arbitrary units'
        else:
            self.model.aux_data.data['intrinsic_spectrum'] = self.model.aux_data.data['intrinsic_spectrum'].value
        p_intrinsic = figure(width=self.width, height=int(0.8 * self.height),
                            min_width=400, title='Model Spectrum',
                            x_axis_label=self.xlabel,
                            y_axis_label=f"Flux density ({intrinsic_units})",
                            tools = "pan,wheel_zoom,box_zoom,reset",
                            output_backend="webgl",
                            x_range=p_main.x_range,
                            min_border_left=80, stylesheets=dragons_styles())
        p_intrinsic.height_policy = 'fixed'
        p_intrinsic.width_policy = 'fit'
        p_intrinsic.sizing_mode = 'stretch_width'
        p_intrinsic.step(x='waves', y='corrected', source=self.model.aux_data,
                         line_width=2, color="crimson")
        intrinsic_line = p_intrinsic.step(x='waves', y='intrinsic_spectrum', source=self.model.aux_data,
                         line_width=2, color="blue", mode="center")
        # We only want to scale to the intrinsic spectrum
        p_intrinsic.y_range.renderers = [intrinsic_line]

        self.p_main = p_main

        # Do a custom padding for the ranges
        self.reset_view()

        return [p_main, info_panel.component, p_supp, p_intrinsic]

    def model_change_handler(self, model):
        """
        If the model changes, this gets called to evaluate the fit and save the results.

        Parameters
        ----------
        model : :class:`~geminidr.interactive.fit.fit1d.InteractiveModel1D`
            The model that changed.
        """
        model.evaluation.data['model'] = model.evaluate(model.evaluation.data['xlinspace'])
        model.aux_data.data['continuum'] = model.fit.continuum(model.aux_data.data['waves'])
        model.aux_data.data['corrected'] = model.aux_data.data['spectrum'] * model.fit.self_correction()


class TelluricVisualizer(Fit1DVisualizer):
    """
    A Visualizer specific to telluricCorrect
    """
    def __init__(self, tcal, **kwargs):
        self.calibrator = tcal
        # Reformat from dict of lists to list of dicts
        all_fp_init = [{k: v[i] for k, v in tcal.fit_params.items()}
                       for i in range(len(tcal))]
        kwargs['domains'] = [(min(tspek.waves), max(tspek.waves))
                             for tspek in tcal.spectra]

        # This is a bit of a hack. The parent __init__() method calls
        # the reconstruct_points() function but we don't want that at init
        # time because the Calibrator will already have done it. As currently
        # coded, the main reason for this is so the Visualizer knows how many
        # panels to create, but it can now get that as len(Calibrator) and
        # the actual data are sourced by the Visualizer.get_data() method,
        # which knows about the internal storage of the Calibrator (so, as in
        # this case, the data are stored as a single-element list if there's
        # only one spectrum).
        #
        # So to avoid calling reconstruct_points() again at init, we pass a
        # static list of the data and then update the relevant attribute of
        # the Visualizer to have the function called whenever we need to
        # *actually* reconstruct the points later.
        # We also have to cope with a bug in Fit1DVisualizer.__init__()
        # that it doesn't recognize "mask" as a parameter, i.e., the
        # mask must be passed by virtue of making y a masked_array.

        init_data = self.get_data(None)
        if isinstance(init_data['x'], list):
            init_data['y'] = [np.ma.masked_array(data, mask=mask)
                              for data, mask in zip(init_data['y'], init_data['mask'])]
        else:
            init_data['y'] = np.ma.masked_array(init_data['y'], mask=init_data['mask'])
        del init_data['mask']

        # The band_model listeners will try to perform a fit but we've
        # already done it
        self.actively_fitting = True
        super().__init__(init_data, all_fp_init,
                         **kwargs, panel_class=TelluricPanel,
                         help_text=TELLURIC_CORRECT_HELP_TEXT,
                         turbo_tabs=True)
        self.reconstruct_points_fn = tcal.reconstruct_points
        self.actively_fitting = False

        # Some stuff so we can get the desired behaviour in fit.perform_fit()
        self.actively_fitting = False
        for i, fit in enumerate(self.fits):
            fit.my_fit_index = i

        # Hacky fix to sort out the widgets in the left panel;
        # but at least it's agnostic to the order of the widgets
        for widget in self.reinit_panel.children:
            if isinstance(widget, bm.layouts.Row):
                widget_title = getattr(widget.children[0], 'title', None)
                widget = widget.children[-1]
            else:
                widget_title = getattr(widget, 'title', None)
            if isinstance(widget, bm.TextInput):
                # This seems to be a bug in make_widgets_from_parameters() that
                # doesn't set the value of the TextInput (magnitude)
                widget.value = self.ui_params.magnitude
                # Since the TextInput and CheckboxGroup only rescale the
                # intrinsic spectrum, it
                # doesn't affect any of the fits really so we could just update
                # the "scaling" attribute of each TelluricWithContinuum model
                # and then ensure that the sensitivity curve is recalculated
                def _scaling_handler(attr, old, new):
                    self.rescale_intrinsic_spectrum(old, new)
                widget.on_change('value', _scaling_handler)
            elif isinstance(widget, bm.CheckboxGroup):
                # There's no way to remove the existing callback (that calls
                # reconstruct_points() in an unwanted way) unless we can
                # provide it with the existing function object, which is in
                # an inaccessible namespace. So hack it!
                widget._callbacks['active'] = []
                widget.on_change('active', _scaling_handler)
            elif isinstance(widget, bm.NumericInput):
                # We make the BBtemp slider/textbox "modal", i.e., it gets
                # disabled when reconstruct_points() is called and that
                # shuts down the GUI until the fitting is complete
                # LSF_param sliders will also be selected here
                def _textslider_handler(attr, old, new):
                    if not self.actively_fitting:
                        self.reconstruct_points()
                widget.on_change('value', _textslider_handler)
                if "temp" in widget_title:
                    self.modal_widget = widget
                    self.make_modal(widget, "Refitting all data")

        # We need to do this here (it performs *all* the fits) because we
        # suppressed fitting when creating the Panels (special for Telluric)
        self.fits[0].perform_fit()
        for lsf_param in self.calibrator.lsf_parameter_bounds:
            print("LSF_PARAM", lsf_param, getattr(self.fitted_model, lsf_param))
            self.ui_params.values[lsf_param] = getattr(self.fitted_model, lsf_param).value

    def rescale_intrinsic_spectrum(self, old, new):
        """
        Perform a computationally-unintensive rescaling of the intrinsic
        spectrum and modify the fit coefficients.
        """
        if isinstance(new, str):
            w0old, f0old = parse_magnitude(old, self.widgets['abmag'])
            w0new, f0new = parse_magnitude(new, self.widgets['abmag'])
            if w0old is None:
                # means we've just reset the "magstr" TextInput after an
                # invalid value was entered
                return
            if w0new is None:
                self.show_user_message("Invalid magnitude; reverting")
                self.widgets['magnitude'].value = old
                return
            planck = Planck(self.widgets['bbtemp'].value)
            scaling = (f0new / f0old * planck(w0old.value) /
                       planck(w0new.value) * (w0old / w0new) ** 2).value
        else:
            # "ABmag" checkbox has changed, so we know the magstr is OK
            filt = self.widgets['magnitude'].value.split("=")[0]
            dmag = VEGA_INFO[filt][1]
            if not new:  # unchecked, so AB->Vega
                dmag *= -1  # which means it gets fainterf
            scaling = 10 ** (0.4 * dmag)

        # Update the plotted data
        for panel, fit, m in zip(self.panels, self.fits, self.fitted_model.models):
            fit.aux_data.data['intrinsic_spectrum'] *= scaling
            fit.aux_data.data['corrected'] *= scaling
            panel.reset_view()

        # Update the scaling values (the fit coefficients don't change!)
        for m in self.fitted_model.models:
            m.intrinsic_spectrum *= scaling

    def reconstruct_points(self):
        """
        We need to override the parent method since it performs the fit
        on each tab.
        """
        self.modal_widget.disabled = True

        def fn():
            # I don't think we need to check for invalid inputs since they're
            # checked when changed by the handlers
            self.calibrator.set_fitting_params(ui_params=self.ui_params)
            self.calibrator.reconstruct_points()

            for i, (fit, tspek) in enumerate(zip(self.fits, self.calibrator.spectra)):
                fit.populate_bokeh_objects()
                fit.aux_data.data["intrinsic_spectrum"] = \
                    tspek.intrinsic_spectrum.value

            # This actually performs all the fits
            self.fits[0].perform_fit()

            # We don't need (or want) to do this since perform_fit() does it
            # and will unblock the GUI
            #self.modal_widget.disabled = False
            for pnl in self.panels:
                pnl.reset_view()

        self.do_later(fn)

    def results(self):
        return self.fitted_model

    def get_data(self, index=None):
        """
        Get the fundamental data for a Panel object to plot. This must
        return a dict with keys 'x', 'y', and 'mask', plus any other masks
        relevant for the main figure.

        Our Calibrator always returns a list, even if there's only 1 panel
        """
        if index is None:  # allow this to return everything in one call
            index = slice(None)
        data = {'x': self.calibrator.x[index],
                'y': self.calibrator.y[index],
                'mask': self.calibrator.mask[index],
                'stellar_mask': self.calibrator.stellar_mask[index]
                }
        return data

    def get_auxiliary_data(self, index):
        """
        New method that should exist (no-op) in the parent to provide data
        for an additional CDS for other plots. This should return a dict so
        the container CDS can be updated without figures losing the reference
        """
        this_tspek = self.calibrator.spectra[index]
        # Note that the last two entries are set to zero. They are only
        # used in the UI, so the UI calculates them.
        data = {"waves": this_tspek.waves,
                "spectrum": this_tspek.data,
                "intrinsic_spectrum": this_tspek.intrinsic_spectrum,
                "continuum": np.zeros_like(this_tspek.data),
                "corrected": np.zeros_like(this_tspek.data)
                }
        return data
