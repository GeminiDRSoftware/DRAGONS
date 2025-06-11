import os
from copy import copy, deepcopy

from abc import ABC, abstractmethod
from functools import partial

from importlib import import_module

import numpy as np

from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from scipy.stats import gmean

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.modeling import models
from astropy.table import Table
from astropy import units as u

from . import Spect
from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance
from geminidr.interactive.interactive import UIParameters
import geminidr.interactive.server
from ..gemini.lookups import qa_constraints

from gempy.library import astromodels as am, astrotools as at
from gempy.library import convolution, peak_finding
from gempy.library.config import RangeField
from gempy.library.calibrator import TelluricCalibrator, TelluricCorrector
from gempy.library.telluric import TelluricModels, TelluricSpectrum
from gempy.library.telluric_models import PCA
from gempy.library.wavecal import LineList
from geminidr.interactive.fit.telluric import TelluricCorrectVisualizer, TelluricVisualizer

from . import parameters_telluric
from geminidr import CalibrationNotFoundError

from datetime import datetime


# geminidr.gemini.lookups
LOOKUPS_PATH = os.path.split(qa_constraints.__file__)[0]


# Create a class for testing, primitives will move into an existing class
@parameter_override
@capture_provenance
class Telluric(Spect):
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_telluric)

        try:
            lsf_module = import_module('.lsf', self.inst_lookups)
        except ModuleNotFoundError:
            pass
        else:
            self._line_spread_function = lsf_module.lsf_factory(self.__class__.__name__)

    def fitTelluric(self, adinputs=None, **params):
        """
        Simultaneously fit telluric absorption features and instrument
        sensitivity to a 1D spectrum.

        The output has a SENSFUNC table attached to each extension, with the
        same format as the one produced by Spect.calculateSensitivity(), and
        also a TELLURIC table attached to the PHU with the coefficients for
        each of the telluric PCA components and the name of the PCA file.

        Parameters
        ----------
        suffix: str/None
            Suffix to be added to output files
        bbtemp: float
            Blackbody temperature of telluric standard
        magnitude: str (format like "K=15")
            Telluric magnitude for absolute flux normalization
        abmag: bool
            is the magnitude on the AB system? (if not, Vega)
        regions: str
            regions (in wavelength space) to sample for fitting
        weighting: str ["variance" | "none"]
            how to weight points
        shift_tolerance: float/None
            maximum pixel shift to ignore (None means ignore)
        apply_shift: bool
            permanently apply the pixel shift?
        debug_lsf_sampling: int
            number of sampling points to cover each LSF parameter
        other parameters define the Line Spread Function scaling
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        weighting = params["weighting"]
        interactive = params["interactive"]
        shift_tolerance = params["shift_tolerance"]
        apply_shift = params["apply_shift"]
        lsf_sampling = params["debug_lsf_sampling"]
        iter_list = [(False, interactive)]
        if shift_tolerance is not None:
            iter_list = [(True, False)] + iter_list
        sampling = 10

        for ad in adinputs:
            log.stdinfo(f"Processing {ad.filename} (this can be slow)")

            # We do this "interactive" stuff here because we want to
            # call create_interactive_inputs() to set the intrinsic_spectrum
            data_units = "adu" if ad.is_in_adu() else "electron"
            config = copy(self.params[self.myself()])
            config.update(**params)
            title_overrides = {"abmag": "Magnitude is on the AB scale?"}
            # 'placeholders' needs to be set due to a bug in interactive
            uiparams = UIParameters(
                config, reinit_params=["bbtemp", "magnitude", "abmag"],
                title_overrides=title_overrides,
                placeholders={"magnitude": ""})

            apertures = set(ad.hdr.get('APERTURE')) - {None}
            aperture_to_use = min(apertures) if apertures else None
            if len(apertures) > 1:
                log.warning(f"Multiple apertures found: using {aperture_to_use}")

            pixel_shift = None
            for (modify, inter) in iter_list:
                # Prepare for fitting for each 1D extension. We need to do
                # this a second time if we modify the WCS because the PCA
                # models will be resampled to a different wavelength solution
                # -- unless we determined a shift and it was zero!
                if pixel_shift != 0:
                    tspek_list = []
                    spectral_indices = []
                    spectral_order_names = []
                    start = datetime.now()
                    for i, ext in enumerate(ad):
                        if (len(ext.shape) == 1 and
                                ext.hdr.get('APERTURE') == aperture_to_use):
                            lsf = self._line_spread_function(ext)
                            axis_name = ext.wcs.output_frame.axes_names[0]
                            in_vacuo = axis_name == "WAVE"
                            tspek = TelluricSpectrum(
                                ext.nddata, line_spread_function=lsf,
                                name=f"Extension {ext.id}", in_vacuo=in_vacuo)

                            if len(ad) > 1:
                                log.stdinfo(f"    Convolving models for extension {ext.id}")
                            lsf_param_names = getattr(lsf, 'parameters', None)
                            if lsf_param_names:
                                uiparams.reinit_params.extend(
                                    [pname for pname in lsf_param_names
                                     if pname not in uiparams.reinit_params]
                                )
                                lsf_params = {k: np.logspace(np.log10(v.min), np.log10(v.max), lsf_sampling)
                                              for k, v in zip(config.keys(), config.iterfields())
                                              if k in lsf_param_names}
                                tspek.set_pca(lsf_params=lsf_params)
                            else:
                                tspek.set_pca()

                            # The NDData object is deepcopied, so we can
                            # modify it without any unwanted side effects
                            if weighting == "uniform":
                                tspek.nddata.variance = None

                            tspek_list.append(tspek)
                            spectral_indices.append(i)
                            try:
                                spectral_order_names.append(f"Order {ext.hdr['SPECORDR']}")
                            except KeyError:
                                spectral_order_names.append(f"Extension {i}")
                    # print(datetime.now() - start, "Making Calibrator() object")
                    if not tspek_list:
                        raise ValueError(f"No 1D spectra found in {ad.filename}")
                    tcal = TelluricCalibrator(tspek_list, ui_params=uiparams)
                    # print(datetime.now() - start, "Made Calibrator() object")

                if inter:
                    visualizer = TelluricVisualizer(
                        tcal, tab_name_fmt=lambda i: spectral_order_names[i],
                        xlabel="Wavelength (nm)", ylabel=f"Signal ({data_units})",
                        title="Fit Telluric",
                        primitive_name=self.myself(),
                        filename_info=ad.filename,
                        ui_params=uiparams
                    )
                    geminidr.interactive.server.interactive_fitter(visualizer)
                    m_final = visualizer.results()
                else:
                    # Before measuring the pixel shift, we perform a fit
                    # so that we have the best model for cross-correlation
                    if modify:
                        log.stdinfo("Calculating pixels shift(s)")
                    m_final, mask = tcal.perform_all_fits()
                pca_coeffs, fit_models = m_final.fit_results()

                # Cross-correlate model with data and shift wavelengths
                if modify:
                    pixel_shifts = []
                    for ext, tspek in zip(ad, tcal.spectra):
                        if len(ext.shape) > 1:
                            continue

                        # We need to define "typical" lsf_param values in
                        # order to evaluate the absorption for xcorr
                        if lsf_param_names:
                            pca_parameters = np.r_[pca_coeffs,
                                                   [gmean(v) for v in lsf_params.values()]]
                        else:
                            pca_parameters = pca_coeffs

                        pixel_shift = peak_finding.cross_correlate_subpixels(
                            tspek.nddata, tspek.pca.evaluate(None, pca_parameters),
                            sampling)
                        if pixel_shift is None:
                            log.warning("Cannot determine cross-correlation"
                                        f"peak for {ext.id}")
                            pixel_shift = None  # needs to exist for later
                        else:
                            log.stdinfo(f"Shift for extension {ext.id} is "
                                        f"{pixel_shift:.2f} pixels")
                            pixel_shifts.append(pixel_shift)

                    if len(ad) > 1:
                        pixel_shift = self._calaculate_mean_pixel_shift(pixel_shifts)

                    if pixel_shift:
                        # shift_tolerance must have a value since modify=True
                        if abs(pixel_shift) > shift_tolerance:
                            # Slightly lazily copy them all, even for non-spectra
                            if not apply_shift:
                                orig_wcs_list = [deepcopy(ext.wcs) for ext in ad]
                            for i in spectral_indices:
                                ad[i].wcs.insert_transform(
                                    ad[i].wcs.input_frame, models.Shift(pixel_shift),
                                    after=True)
                        else:
                            pixel_shift = 0
                            log.stdinfo(f"Shift is within {shift_tolerance}"
                                        " tolerance so will not be applied")

            # Stuff the results into the ad object
            tellfit_table = Table([pca_coeffs], names=("PCA coefficients",))
            # Versioning and other useful stuff
            tellfit_table.meta = {'airmass': ad.airmass(),
                                  'pca_name': tcal.spectra[0].pca.name}
            lsf_param_dict = {k: getattr(m_final, k).value for k in lsf_param_names}
            tellfit_table.meta.update(lsf_param_dict)
            ad.TELLFIT = tellfit_table

            # Attach results to correct extensions in telluric AD object
            result_index = 0
            for ext in ad:
                if (len(ext.shape) == 1 and
                        ext.hdr.get('APERTURE') == aperture_to_use):
                    tspek = tspek_list[result_index]
                    absorption = tspek.data / m_final.models[result_index].continuum(tspek.waves)
                    goodpix = ~(tspek.mask | tcal.stellar_mask[result_index]).astype(bool)
                    spline = make_interp_spline(tcal.spectra[result_index].waves[goodpix],
                                                absorption[goodpix], k=3)
                    spline.extrapolate = False  # will return np.nan outside range
                    ext.TELLABS = spline(tspek.waves).astype(ext.data.dtype)
                    result_index += 1

            # We have to correct for exposure time and add the SENSFUNC units
            # The models are hardcoded to return the correct units
            sensfunc_unit = u.Magnitude(1 * u.Unit('erg cm-2') / u.Unit(data_units)).unit
            offset = -2.5 * np.log10(ad.exposure_time())
            for ext, m in zip(ad, fit_models):
                try:
                    m.c0 += offset
                except AttributeError:  # it's a BSpline
                    m.c += offset
                ext.SENSFUNC = am.model_to_table(m, xunit=u.nm, yunit=sensfunc_unit)

            if not apply_shift:
                log.debug(f"Resetting all gWCS objects in {ad.filename}")
                for ext, orig_wcs in zip(ad, orig_wcs_list):
                    ext.wcs = orig_wcs

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def telluricCorrect(self, adinputs=None, **params):
        """
        Apply a previous telluric fit to a "science" spectrum. This is done
        by reading the TELLFIT table from an object that has gone through the
        fitTelluric() primitive and applying a PCA model with those
        coefficients to the input AD objects. An appropriate line spread
        function is derived for each extension of each input AD, and an
        airmass correction is also applied.

        Alternatively, the data-derived telluric absorption can be applied,
        which is stored as a 1D TELLABS array in the telluric, and is simply
        the ratio of the telluric spectrum to the zero-airmass model spectrum
        computed by fitTelluric().

        A pixel shift between the telluric spectrum and the science spectrum
        can be calculated before the correct, and this shift may be permanently
        applied to the output science spectrum's wavelength solution. The
        shift is determined by looking at the highest-frequency Fourier
        component of the corrected spectrum, to identify the "ringing" that
        arises from a wavelength offset. Note that if apply_model=False and the
        "data"-derived telluric absorption spectrum is used, then the output
        science spectrum will be modified to match that of the telluric. If the
        model absorption spectrum is used, its wavelength scale is obviously
        correct.

        There is no need for the TELLFIT table to be on a spectrum taken
        with the same set-up or even the same instrument as the input ADs.

        Parameters
        ----------
        suffix: str/None
            Suffix to be added to output files
        telluric: str/AstroData/None
            File/AD object with TELLFIT table listing PCA coefficients
            of telluric absorption fit
        apply_model: bool
            apply a correction from the PCA model rather than the data?
        interactive: bool
            run primitive interactively using GUI?
        shift_tolerance: float/None
            minimum allowed tolerance when calculating pixel shift between
            telluric and science spectra. If None, then no shift is
            calculated. If a shift is less than this value, then it is not
            applied.
        apply_shift: bool
            apply the pixel shift permanently to the wavelength solution of
            the corrected science spectrum?
        pixel_shift: float/None
            apply a shift of this value, or (if None), calculate the pixel
            shift between the telluric and science spectra
        delta_airmass: float/None
            if not None, override the header airmass of the telluric by adding
            this value to the airmass of the science spectrum
        do_cal: str ["procmode" | "force" | "skip"]
            Perform this calibration? ("skip" skips, the others will attempt
            but be OK if no suitable calibration is found)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = "TELLCORR"  # self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        telluric = params["telluric"]
        apply_model = params["apply_model"]
        interactive = params["interactive"]
        shift_tolerance = params["shift_tolerance"]
        apply_shift = params["apply_shift"]
        manual_shift = params["pixel_shift"]
        user_airmass = params["delta_airmass"]
        do_cal = params["do_cal"]

        if do_cal == 'skip':
            log.warning("Telluric correction has been turned off.")
            return adinputs

        # Get a suitable standard (should have a TELLFIT table)
        if telluric is None:
            telluric_list = self.caldb.get_processed_telluric(adinputs)
        else:
            telluric_list = (telluric, None)

        # Provide a standard AD object for every science frame, and an origin
        for ad, telluric, origin in zip(*gt.make_lists(adinputs, *telluric_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            f"{self.myself()}. Continuing.")
                continue

            if telluric is None:
                if 'sq' in self.mode or do_cal == 'force':
                    raise CalibrationNotFoundError("No processed telluric "
                                                   f"listed for {ad.filename}")
                else:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no processed telluric was specified")
                    continue

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the processed telluric {telluric.filename}"
                        f"{origin_str}")

            # In case there are multiple apertures in the telluric
            apertures = set(ad.hdr.get('APERTURE')) - {None}
            aperture_to_use = min(apertures) if apertures else None

            # Check that all the necessary extensions exist, or bail
            try:
                tellfit = telluric.TELLFIT
            except AttributeError:
                has_model = False
                # if apply_model:
                #     log.warning(f"{ad.filename}: no TELLFIT table on processed telluric"
                #                 f" {telluric.filename} so cannot apply correction.")
                #     continue
                header = {}  # can still progress
            else:
                has_model = True
                header = tellfit.meta['header']
                pca_name = header['pca_name']
                pca_coeffs = tellfit['PCA coefficients'].data

            has_data = True
            tellabs_dict = {}
            for ext_telluric in telluric:
                if (ext_telluric.hdr.get('APERTURE') == aperture_to_use and
                        len(ext_telluric.shape) == 1):
                    try:
                        tellabs = ext_telluric.TELLABS
                    except AttributeError:
                        tellabs_dict = {}  # makes it easier to find errors
                        has_data = False
                        break
                    else:
                        # Save the telluric spectrum as a spline so that it
                        # can be interpolated onto any wavelength solution
                        # order=3 seems to cause some ringing, and order=1 is
                        # what the TelluricCorrector uses anyway
                        tell_waves = ext_telluric.wcs(np.arange(
                            ext_telluric.data.size))
                        tellabs_dict[ext_telluric.hdr.get('SPECORDR')] = \
                            make_interp_spline(tell_waves[~np.isnan(tellabs)],
                                               tellabs[~np.isnan(tellabs)], k=1)

            # TODO: If we're interactive we can override apply_model if necessary
            if not interactive:
                if apply_model and not has_model:
                    log.warning(f"{ad.filename} has no TELLFIT model but "
                                "apply_model=True. Continuing.")
                    continue
                elif not apply_model and not has_data:
                    log.warning(f"{ad.filename}: one or more 1D extensions on"
                                f" standard {telluric.filename} are missing a"
                                " TELLABS spectrum. Continuing.")
                    continue
            elif not (has_data and has_model):
                log.warning(f"{ad.filename} does not have the necessary "
                            "telluric information to be used. Continuing.")
                continue

            # We're hot to go! (as Chappell Roan would say)
            # First is to get a cross-correlation estimate, unless one has
            # been provided or the user has said not to (we don't apply an
            # airmass correction here). We loop through the AD even if we're
            # not cross-correlating, so we can construct the TelluricSpectrum
            # objects
            if manual_shift is not None:
                pixel_shift = manual_shift
            elif shift_tolerance is None:
                pixel_shift = 0
            else:
                pixel_shift = None
            tspek_list = []
            pixel_shifts = []
            tell_int_splines = []
            tellabs_data = []

            for ext in ad:
                if len(ext.shape) > 1:
                    continue

                axis_name = ext.wcs.output_frame.axes_names[0]
                in_vacuo = axis_name != "WAVE"
                if axis_name not in ("AWAV", "WAVE"):
                    log.warning("Cannot determine whether wavelength "
                                "scale is calibrated to vacuum or air "
                                "from name '{axis_name}'on extension "
                                f"{ext.id}. Assuming vacuum.")

                lsf = self._line_spread_function(ext)
                tspek = TelluricSpectrum(
                    ext.nddata, line_spread_function=lsf,
                    name=f"Extension {ext.id}", in_vacuo=in_vacuo)
                tspek_list.append(tspek)

                # Has to be here because of in_vacuo
                # TODO? Create multiple int_splines for different lsf_params?
                if apply_model or interactive:
                    w_pca, t_pca = TelluricModels.data(pca_file=pca_name,
                                                       in_vacuo=True)
                    tellabs_model = PCA(t_pca).evaluate(tspek.waves, pca_coeffs)
                    lsf_param_names = getattr(lsf, 'parameters', [])
                    lsf_params = {k: header[k] for k in lsf_param_names}
                    wconv, tconv = lsf.convolve(tspek.waves, w_pca,
                                                tellabs_model, **lsf_params)
                    spline = make_interp_spline(wconv, tconv, axis=-1, k=3)
                    spline.extrapolate = False
                    tell_int_splines.append(spline.antiderivative())

                if pixel_shift is None:
                    if apply_model:
                        trans = convolution.resample(tspek.waves, wconv, tconv)
                    else:
                        trans = tellabs_dict[ext.hdr.get('SPECORDR')](tspek.waves)

                    # Old cross-correlation method
                    #pixel_shift = peak_finding.cross_correlate_subpixels(
                    #    ext.data, trans, sampling)
                    #pixel_shift = 0.01 * np.round(pixel_shift * 100)

                    pixels = np.arange(ext.shape[0])
                    # We want to avoid integer shifts since they produce
                    # artifacts because there's no interpolation and smoothing
                    dx_all = np.arange(-5.1667, 5.1667, 0.33333)
                    fft_all = np.empty_like(dx_all)
                    for i, dx in enumerate(dx_all):
                        shift_trans = np.interp(pixels + dx, pixels, trans,
                                                left=np.nan, right=np.nan)
                        # Avoid edge effects by clipping pixels
                        fft_all[i] = abs(np.fft.fft(at.divide0(ext.data[10:-10],
                                                               shift_trans[10:-10]))[0])

                    try:
                        best = peak_finding.pinpoint_peaks(
                            -fft_all, [fft_all.argmin()], halfwidth=3)[0]
                    except IndexError:
                        log.warning("Cannot determine cross-correlation"
                                    f"peak for {ext.id}")
                        pixel_shift = None  # needs to exist for later
                    else:
                        pixel_shift = np.round(np.interp(
                            best[0], np.arange(fft_all.size), dx_all), decimals=2)
                        log.stdinfo(f"Shift for extension {ext.id} is "
                                    f"{pixel_shift:.2f} pixels")
                        pixel_shifts.append(pixel_shift)

                if has_data:
                    # Interpolate telluric spectrum onto science wavelengths
                    tellabs_data.append(tellabs_dict[ext.hdr.get('SPECORDR')](tspek.waves))

            if len(pixel_shifts) > 1:
                pixel_shift = self._calculate_mean_pixel_shift(pixel_shifts)
            # i.e., pixel shift has been calculated
            if (pixel_shift and manual_shift is None and
                    abs(pixel_shift) < shift_tolerance):
                pixel_shift = 0
                log.stdinfo(f"Shift is within {shift_tolerance}"
                            " tolerance so will not be applied")

                # Get LSF parameters from first 1D spectrum extension
                #lsf = self._line_spread_function(
                #    [ext for ext in ad if len(ext.shape) == 1][0])
                #lsf_param_names = getattr(lsf, 'parameters', None)

            telluric_airmass = header.get('AIRMASS', telluric.airmass())
            sci_airmass = ad.airmass()
            if user_airmass is None:
                try:
                    sci_airmass - telluric_airmass
                except TypeError:  # if either airmass() returns None
                    log.warning("Cannot determine airmass of target "
                                f"{ad.filename} and/or telluric {telluric.filename}." +
                                ("" if interactive else " Not performing airmass correction."))
                    # Set values for UI, or ones that will allow a calculation
                    if sci_airmass is None:
                        sci_airmass = telluric_airmass or 1.2
                    if telluric_airmass is None:
                        telluric_airmass = sci_airmass
            else:
                if telluric_airmass is None:
                    if sci_airmass:
                        telluric_airmass = sci_airmass - user_airmass
                    else:
                        telluric_airmass = 1.2
                        sci_airmass = telluric_airmass + user_airmass
                        log.warning("Cannot determine airmass of standard "
                                    f"{telluric.filename} so assuming {telluric_airmass}.")
                else:
                    sci_airmass = telluric_airmass + user_airmass

            config = copy(self.params[self.myself()])
            config.update(**params)
            config.update(pixel_shift=pixel_shift)
            reinit_extras = {
                "sci_airmass": RangeField(
                    doc="Airmass of target", dtype=float, default=sci_airmass,
                    min=1, max=2.5, inclusiveMax=True),
                "telluric_airmass": RangeField(
                    doc=f"Airmass of telluric", dtype=float, default=telluric_airmass,
                    min=1, max=2.5, inclusiveMax=True)
            }
            uiparams = UIParameters(
                config, reinit_params=["pixel_shift", "sci_airmass",
                                       "telluric_airmass", "apply_model"],
                extras=reinit_extras)

            tcal = TelluricCorrector(tspek_list, ui_params=uiparams,
                                     tellabs_data=tellabs_data,
                                     tell_int_splines=tell_int_splines)

            if interactive:
                tab_labels = self._make_tab_labels(ad)
                data_units = "adu" if ad.is_in_adu() else "electron"
                visualizer = TelluricCorrectVisualizer(
                    tcal, tab_name_fmt=lambda i: tab_labels[i],
                    xlabel="Wavelength (nm)", ylabel=f"Signal ({data_units})",
                    title="Telluric Correct",
                    primitive_name=self.myself(),
                    filename_info=ad.filename,
                    ui_params=uiparams
                )
                geminidr.interactive.server.interactive_fitter(visualizer)
                # We don't need to grab the results because they're in "tcal"
            else:
                tcal.perform_all_fits()

            if apply_shift:
                log.stdinfo(f"Applying shift of {pixel_shift} pixels "
                            f"to {ad.filename}")
            abs_spectra = tcal.absorption_spectra()
            for ext in ad:
                if len(ext.shape) > 1:
                    continue

                ext.divide(next(abs_spectra))
                if apply_shift:
                    ext.wcs.insert_transform(
                        ext.wcs.input_frame, models.Shift(pixel_shift),
                        after=True)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

    def _calculate_mean_pixel_shift(self, pixel_shifts):
        """Helper method to calculate the mean shift and perform logging"""
        if pixel_shifts:
            shift_mask = find_outliers(pixel_shifts)
            if np.any(shift_mask):
                self.log.warning("Ignoring shift(s) of " +
                                 ", ".join(np.array(shift_mask)[shift_mask]))
            pixel_shift = np.mean(np.asarray(pixel_shifts)[~shift_mask])
            self.log.stdinfo(f"Average shift is {pixel_shift:.2f} pixels")
            return pixel_shift
        self.log.warning("Could not determine *any* pixel "
                         "shifts. Not shifting data.")
        return None

    def _get_atran_linelist(self, wave_model=None, ext=None, config=None):
        """
        Return a list of spectral lines to be matched in the wavelength
        calibration, and a reference plot of a convolved synthetic spectrum,
        to aid the user in making the correct identifications (which is
        an attribute of the LineList object).

        The linelist can be generated on-the-fly by finding peaks in the
        convolved spectrum, or read from disk if there exists a suitable
        list for this instrumental setup.

        Parameters
        ----------
        wave_model: ``astropy.modeling.models.Chebyshev1D``
            the current wavelength model (pixel -> wavelength), with an
            appropriate domain describing the illuminated region
        ext: single-slice ``AstroData``
            the extension for which a sky spectrum is being constructed
        config: ``config.Config`` object
            containing various parameters

        Returns
        -------
        ``wavecal.Linelist``
            list of lines to match, including data for a reference plot
        """
        log = self.log
        observatory = ext.telescope()
        site = {'Gemini-North': 'mk', 'Gemini-South': 'cp'}[observatory]
        altitude = {'Gemini-North': 13825, 'Gemini-South': 8980}[observatory]
        wv_band = config.get("wv_band", "header")
        if wv_band == "header":
            wv_band = ext.raw_wv()
            if wv_band is None:
                wv_band = ext.requested_wv()
                log.stdinfo("Unknown RAWWV for this observation: "
                            f"using requested band WV{wv_band}")
        wv_band = int(wv_band)
        if wv_band == 100:
            # WV value to use for the case of RAWWV='Any'
            wv_content = {'Gemini-North': 5, 'Gemini-South': 10}[observatory]
        else:
            wv_content = qa_constraints.wvBands[observatory].get(str(wv_band))

        resolution = config.get("resolution") or self._get_resolution(ext)
        # The wave_model's domain describes the illuminated region
        wave_model_bounds = self._wavelength_model_bounds(wave_model, ext)
        try:
            domain = wave_model.domain
        except AttributeError:
            for m in wave_model:
                if hasattr(m, 'domain'):
                    domain = m.domain
                    break
            else:
                raise ValueError("No domain in wavelength model")
        start_wvl, end_wvl = (np.sort(wave_model(domain)) +
                              np.asarray(wave_model_bounds['c0']) -
                              np.mean(wave_model_bounds['c0']))

        # A linelist may be in the Gemini lookup directory, or one may
        # have been created in the cwd
        atran_linelist = (f'atran_linelist_{site}_{start_wvl:.0f}-{end_wvl:.0f}'
                          f'_wv{wv_content:.0f}_r{resolution:.0f}.dat')
        try:
            linelist = LineList(os.path.join(LOOKUPS_PATH, atran_linelist))
            log.stdinfo(f"Using generic linelist {atran_linelist}")
        except FileNotFoundError:
            try:  # prevent using previously-created linelist (for now)
                linelist = LineList(atran_linelist)
                log.stdinfo("Using previously-created linelist in current "
                            f"directory {atran_linelist}")
            except FileNotFoundError:
                # We will need to create one on the fly
                linelist = None

        absorption = config.get("absorption", False)

        # Get the high-resolution spectrum for the site/conditions
        atran_file = os.path.join(LOOKUPS_PATH, "atran_spectra.fits")
        atran_models = Table.read(atran_file)
        waves = atran_models['wavelength']
        data = atran_models[f"{site}_wv{wv_content * 1000:.0f}_za48"]

        # Convolve the appropriate wavelength region with a Gaussian of
        # constant FWHM (only works if wavelength scale is linear)
        wave_range = np.logical_and(waves >= start_wvl, waves <= end_wvl)
        sampling = abs(np.diff(waves).mean())
        sigma_pix = 0.42 * 0.5 * (start_wvl + end_wvl) / resolution / sampling
        atran_spec = convolve(data[wave_range], Gaussian1DKernel(sigma_pix),
                              boundary='extend')
        refplot_spec = np.asarray([waves[wave_range], atran_spec],
                                  dtype=np.float32)

        # Resample the reference spectrum so it has about twice as many pixels
        # as the data, to avoid too much plotting overhead
        resampling = max(int(0.5 * atran_spec.size / np.diff(domain)[0]), 1)
        refplot_spec = refplot_spec[:, ::resampling]

        refplot_spec[1] = 1 - refplot_spec[1]
        if linelist is None:
            # Invert spectrum because we want the wavelengths of troughs
            linelist_data = make_linelist(refplot_spec,
                                          resolution=resolution,
                                          num_lines=config.get('num_atran_lines', 50))
            header = (f"Sky emission line list: {start_wvl:.0f}-{end_wvl:.0f}nm\n"
                      f"Generated at R={int(resolution)} from ATRAN synthetic spectrum "
                      "(Lord, S. D., 1992, NASA Technical Memorandum 103957)\n"
                      "Model parameters:\n"
                      f"Obs altitude: {altitude}ft, Obs latitude: 39 degrees,\n"
                      f"Water vapor overburden: {wv_content * 1000:.0f} microns,"
                      "Number of atm. layers: 2,\n"
                      "Zenith angle: 48 deg, Wavelength range: 1-6 microns, Smoothing R:0\n"
                      "units nanometer\n"
                      "wavelengths IN VACUUM")
            #np.savetxt(atran_linelist, linelist_data, fmt=['%.3f', '%.3f'], header=header)
            np.savetxt(atran_linelist, linelist_data[:, 0], fmt=['%.3f'], header=header)
            linelist = LineList(atran_linelist)

        # In L and M bands, the sky spectrum has emission where the ATRAN
        # spectrum has absorption, so keep the inverted version for display.
        # But if we're actually matching absorption features, then we want
        # to display the original version, so revert it.
        if absorption:
            refplot_spec[1] = 1 - refplot_spec[1]

        refplot_name = (f'ATRAN spectrum (Alt={altitude}ft, WV={wv_content}mm,'
                        f'AM=1.5, R={resolution:.0f})')
        refplot_y_axis_label = ("Atmospheric transmission" if absorption else
                                "Inverse atm. transmission")

        refplot_data = {"refplot_spec": refplot_spec.T,
                "refplot_name": refplot_name,
                "refplot_y_axis_label": refplot_y_axis_label}

        linelist.reference_spectrum = refplot_data
        return linelist


def make_linelist(spectrum, resolution=1000, num_bins=10, num_lines=50):
    """
    Create a linelist (with weights) from a spectrum by selecting the most
    prominent peaks. The spectrum is split into several bins and the
    strongest peaks found in each of those bins, to ensure that there are
    suitable features to match across the entire spectral range.

    Feature strengths are determined by the prominence divided by the width,
    so that broad features must be more pronounced than narrower ones.

    Parameters
    ----------
    data: array (2, N)
        spectrum (wavelengths, flux densities) within which to find peaks
    resolution: float
        approximate spectral resolution of the spectrum (because we only
        want to find unresolved or barely-resolved features)
    num_bins: int
        number of bins (equally spaced in wavelength) to divide the spectrum
        into
    num_lines: int
        total number of lines to keep (these will be spread equally among
        the `nbins`)

    Returns
    -------
    array of shape (M, 2)
        line wavelengths and weights
    """
    wavelength, flux = spectrum
    wavelength_sampling = np.diff(wavelength).mean()
    fwhm = wavelength.mean() / resolution / wavelength_sampling
    pixel_peaks, properties = find_peaks(
        flux, prominence=0.005, width=(None, 5 * fwhm))
    weights = properties["prominences"] / properties["widths"]

    def trim_peaks(peaks, weights, bin_edges, nlargest=10, sort=True):
        """
        Filters the peaks list, binning it over the range of the whole
        signal, preserving only the N-largest ones on each bin

        peaks: array
            pixel locations of peaks
        weights: array
            strengths of peaks
        bin_edges: array of shape (N+1,)
            edges of the N desired bins
        nlargest: int
            number of largest peaks to extract from each bin

        Returns: array of shape (M, 2)
            the M (M <= N * nlargest) line wavelengths and weights
        """
        result = []
        for i1, i2 in zip(bin_edges[:-1], bin_edges[1:]):
            indices = np.logical_and(peaks >= i1, peaks < i2)
            indices_to_keep = weights[indices].argsort()[-nlargest:]
            result.extend(list(zip(peaks[indices][indices_to_keep],
                                   weights[indices][indices_to_keep])))
        return np.array(sorted(result) if sort else result,
                        dtype=peaks.dtype)

    # For the final line list select n // 10 peaks with largest weights
    # within each of 10 wavelength bins.
    bin_edges = np.linspace(0, flux.size + 1, num_bins + 1)
    best_pixel_peaks = trim_peaks(pixel_peaks, weights, bin_edges,
                                  nlargest=(num_lines + num_bins - 1) // num_bins,
                                  sort=True)

    # Pinpoint peak positions, and cull any peaks that couldn't be fit
    # (keep_bad will return location=NaN)
    atran_linelist = np.vstack(peak_finding.pinpoint_peaks(
        flux, peaks=best_pixel_peaks[:, 0], halfwidth=2, keep_bad=True)).T
    atran_linelist = atran_linelist[~np.isnan(atran_linelist).any(axis=1)]

    # Convert back to wavelengths
    atran_linelist[:, 0] = np.interp(atran_linelist[:, 0],
                                  np.arange(wavelength.size),
                                  wavelength)
    return atran_linelist


def find_outliers(data, sigma=3, cenfunc=np.median):
    """
    Examine a list for individual outlying points and flag them.
    This operates better than sigma-clip for small lists as it sequentially
    removes a single point from the list and determined whether it is an
    outlier based on the statistics of the remaining points.

    Parameters
    ----------
    data: array-like
        array to be searched for outliers
    sigma: float
        number of standard deviations for rejection
    cenfunc: callable (np.median/np.mean usually)
        function for deriving average

    Returns
    -------
    mask: bool array
        outlying points are flagged
    """
    mask = np.zeros_like(data, dtype=bool)
    if mask.size < 3:  # doesn't work with 1 or 2 elements
        return mask

    for i in range(mask.size):
        omitted_data = list(set(data) - {data[i]})
        average = cenfunc(omitted_data)
        stddev = np.std(omitted_data)
        if abs(data[i] - average) > sigma * stddev:
            mask[i] = True
    return mask


# Generic LSF classes that will be inherited by the instrument-specific ones
class LineSpreadFunction(ABC):
    def __init__(self, ext):
        assert len(ext.shape) == 1, "Input is not 1-dimensional"
        npix = ext.shape[0]
        self.all_waves = ext.wcs(np.arange(npix))
        self.dispersion = abs(np.median(np.diff(self.all_waves)))
        # For lack of any better estimate, but these should get overridden
        # in the relevant subclass
        self.resolution = self.all_waves.mean() / (2 * self.dispersion)
        self.mean_resolution = self.resolution

    @abstractmethod
    def convolutions(self):
        """Returns list of (convolution function, kernel size in nm)"""

    def convolve(self, waves, w, data, **kwargs):
        """
        Convolve a spectrum/spectra with a kernel

        Parameters
        ----------
        waves: sequence
            provides min and max wavelengths for output convolved spectra
        w: array
            wavelengths of thing to be convolved
        data: array
            data to be convolved; last dimension must match w
        kwargs: dict
            list of parameters to be passed to the convolutions() method

        Returns
        -------
        w: (N,) array of wavelengths of convolved spectra (a subset of w)
        spectra: (..., N) array of convolved spectra
        """
        convolution_list = self.convolutions(**kwargs)
        dw = sum(x[1] for x in convolution_list)
        w1, w2 = np.min(waves) - 1.05 * dw, np.max(waves) + 1.05 * dw
        windices = np.logical_and(w > w1, w < w2)
        spectra = data[..., windices]
        for conv_func, dw in convolution_list:
            spectra = convolution.convolve(w[windices], spectra,
                                           conv_func, dw=dw)
        return w[windices], spectra

    def convolve_and_resample(self, waves, w, data, **kwargs):
        """
        Convolve and resample a spectrum/spectra to the same resolution and
        wavelength array as the main model

        Parameters
        ----------
        waves: array (1D)
            output wavelength scale
        w: array (can be 2D)
            wavelengths of thing to be convolved
        data: array
            data to be convolved; last dimension must match w
        kwargs: dict
            list of parameters to be passed to the convolutions() method

        Returns
        -------
        array of shape of (data.shape[0], waves.size) of resampled spectra
        """
        w, spectra = self.convolve(waves, w, data, **kwargs)
        results = convolution.resample(waves, w, spectra)
        return results


class GaussianLineSpreadFunction(LineSpreadFunction):
    """A generic Gaussian LSF with a constant resolution"""
    parameters = ["resolution"]

    def __init__(self, ext, resolution=None):
        super().__init__(ext)
        self.resolution = resolution
        self.mean_resolution = resolution

    def convolutions(self, resolution=None):
        if resolution is None:
            resolution = self.resolution
        gaussian_func = partial(convolution.gaussian_constant_r, r=resolution)
        gaussian_dw = 3 * self.all_waves.max() / resolution
        convolutions = [(gaussian_func, gaussian_dw)]
        return convolutions
