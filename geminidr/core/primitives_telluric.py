import os
from copy import copy, deepcopy

from abc import ABC, abstractmethod
from functools import partial

from importlib import import_module

import numpy as np

from scipy.interpolate import interp1d

from astropy.modeling import models
from astropy.table import Table
from astropy import units as u

from . import Spect
from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance
from geminidr.interactive.interactive import UIParameters
import geminidr.interactive.server
from gempy.library import astromodels as am, convolution, peak_finding

from gempy.library.calibrator import TelluricCalibrator
from gempy.library.telluric import TelluricSpectrum
from geminidr.interactive.fit.telluric import TelluricVisualizer

from . import parameters_telluric

from datetime import datetime
from matplotlib import pyplot as plt

PATH = os.path.split(__file__)[0]


# Create a class for testing, primitives will move into an existing class
@parameter_override
@capture_provenance
class Telluric(Spect):
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_telluric)

        lsf_module = import_module('.lsf', self.inst_lookups)
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
            log.stdinfo(f"Processing {ad.filename}")

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

            pixel_shift = None
            for (modify, inter) in iter_list:
                # Prepare for fitting for each 1D extension. We need to do
                # this a second time if we modify the WCS because the PCA
                # models will be resampled to a different wavelength solution
                # -- unless we determined a shift and it was zero!
                if pixel_shift != 0:
                    tspek_list = []
                    spectral_indices = []
                    start = datetime.now()
                    for i, ext in enumerate(ad):
                        if len(ext.shape) == 1:
                            lsf = self._line_spread_function(ext)

                            tspek = TelluricSpectrum(
                                ext.nddata, line_spread_function=lsf,
                                name=f"Extension {ext.id}")

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
                            # modify it without any side effects
                            if weighting == "uniform":
                                tspek.nddata.variance = None

                            tspek_list.append(tspek)
                            spectral_indices.append(i)
                    print(datetime.now() - start, "Making Calibrator() object")
                    tcal = TelluricCalibrator(tspek_list, ui_params=uiparams)
                    print(datetime.now() - start, "Made Calibrator() object")

                if inter:
                    visualizer = TelluricVisualizer(
                        tcal, tab_name_fmt=lambda i: f"Order {i+1}",
                        xlabel="Wavelength (nm)", ylabel=f"Signal ({data_units})",
                        title="TITLE",
                        primitive_name=self.myself(),
                        filename_info=ad.filename,
                        ui_params=uiparams
                    )
                    geminidr.interactive.server.interactive_fitter(visualizer)
                    m_final = visualizer.results()
                else:
                    if modify:
                        log.stdinfo("Calculating wavelength shift(s)")
                    m_final, mask = tcal.perform_all_fits()
                pca_coeffs, fit_models = m_final.fit_results()

                # Cross-correlate model with data and shift wavelengths
                if modify:
                    pixel_shifts = []
                    for ext, tspek in zip(ad, tcal.spectra):
                        if len(ext.shape) > 1:
                            continue

                        pixel_shift = peak_finding.cross_correlate_subpixels(
                            tspek.nddata, tspek.pca.evaluate(None, pca_coeffs),
                            sampling)
                        if pixel_shift is None:
                            log.warning("Cannot determine cross-correlation"
                                        f"peak for {ext.id}")
                            pixel_shift = None  # needs to exist for later
                        else:
                            log.stdinfo(f"Shift for extension {ext.id} is "
                                        f"{pixel_shift:.2f} pixels")
                            pixel_shifts.append(pixel_shift)

                        #fig, ax = plt.subplots()
                        #ax.plot(np.arange(xcorr.size) - xcorr.size // 2, xcorr)
                        #plt.show()

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
            ad.TELLFIT = tellfit_table

            # Also attach data divided by continuum?
            # Problem with intrinsic absorption not being fitted perfectly
            for ext, tspek, stellar_mask, tmodel in zip(ad, tcal.spectra, tcal.stellar_mask, m_final.models):
                absorption = tspek.data / tmodel.continuum(tspek.waves)
                goodpix = ~(tspek.mask | stellar_mask).astype(bool)
                spline = interp1d(tspek.waves[goodpix],
                                  absorption[goodpix], kind='cubic',
                                  copy=False, bounds_error=False,
                                  fill_value=np.nan)
                ext.TELLABS = spline(tspek.waves)
                #ext.TELLABS2 = absorption

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
        do_cal: str ["procmode" | "force" | "skip"]
            Perform this calibration? ("skip" skips, the others will attempt
            but be OK if no suitable calibration is found)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = "TELLCORR"  # self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        std = params["telluric"]
        apply_model = params["apply_model"]
        shift_tolerance = params["shift_tolerance"]
        apply_shift = params["apply_shift"]
        manual_shift = params["manual_shift"]
        user_airmass = params["delta_airmass"]
        sampling = 10

        if params["do_cal"] == 'skip':
            log.warning("Flux calibration has been turned off.")
            return adinputs

        # Get a suitable standard (should have a TELLFIT table)
        if std is None:
            std_list = self.caldb.get_processed_standard(adinputs)
        else:
            std_list = (std, None)

        # Provide a standard AD object for every science frame, and an origin
        for ad, std, origin in zip(*gt.make_lists(adinputs, *std_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            f"{self.myself()}. Continuing.")
                continue

            if std is None:
                log.warning(f"No changes will be made to {ad.filename}, "
                            "since no standard was specified")
                continue

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the standard {std.filename}"
                        f"{origin_str}")

            # Check that all the necessary extensions exist, or bail
            try:
                tellfit = std.TELLFIT
            except AttributeError:
                if apply_model:
                    log.warning(f"{ad.filename}: no TELLFIT table on standard"
                                f" {std.filename}. Continuing.")
                    continue
                header = {}  # can still progress
            else:
                header = tellfit.meta['header']
                pca_name = header['PCA_NAME']
                pca_coeffs = tellfit['PCA coefficients'].data
            if not apply_model:
                if not all(hasattr(ext_std, "TELLABS") for ext_std in std
                           if len(ext_std.shape) == 1):
                    log.warning(f"{ad.filename}: one or more 1D extensions on"
                                f"standard {std.filename} are missing a "
                                "TELLABS spectrum. Continuing.")
                    continue

            std_airmass = header.get('AIRMASS', std.airmass())
            if std_airmass is None:
                std_airmass = 1.58
            if user_airmass is None:
                sci_airmass = ad.airmass()
                try:
                    delta_airmass = sci_airmass - std_airmass
                except TypeError:  # if either airmass() returns None
                    log.warning("Cannot determine airmass of target "
                                f"{ad.filename} and/or standard {std.filename}. "
                                "Not performing airmass correction.")
                    delta_airmass = None
                else:
                    log.stdinfo(f"{ad.filename}: Correcting for difference of "
                                f"{delta_airmass:5.3f} airmasses")
            else:
                delta_airmass = user_airmass
                if std_airmass is None:
                    log.warning("Cannot determine airmass of standard "
                                f"{std.filename} so cannot correct {ad.filename}")
                else:
                    log.stdinfo(f"{ad.filename}: Applying user-supplied "
                                f"difference of {delta_airmass:5.3f} airmasses")

            pixel_shifts = []
            pixel_shift = manual_shift
            if manual_shift:
                log.stdinfo(f"Applying user-provided shift of {manual_shift} pixels")

            # Loop twice if we're cross-correlating
            for xcorr in range(shift_tolerance is not None and manual_shift is None, -1, -1):
                for ext, ext_std in zip(ad, std):
                    if len(ext.data.shape) != 1:
                        continue

                    axis_name = ext.wcs.output_frame.axes_names[0]
                    if apply_model:
                        if axis_name == "AWAV":
                            in_vacuo = False
                        else:
                            in_vacuo = True
                            # Only report on the first pass
                            if axis_name != "WAVE" and pixel_shift is None:
                                # Assume vacuum because we're probably in the IR
                                log.warning("Cannot determine whether wavelength scale is "
                                            "calibrated to vacuum or air from name "
                                            f"'{axis_name}'. Assuming vaccum.")

                        if pixel_shift:
                            ext.wcs.insert_transform(
                                ext.wcs.input_frame, models.Shift(pixel_shift),
                                after=True)
                        tspek = TelluricSpectrum(
                            ext.nddata, line_spread_function=self._line_spread_function(ext),
                            name=f"Extension {ext.id}", pca_file=pca_name,
                            in_vacuo=in_vacuo)
                        trans = tspek.pca.evaluate(tspek.waves, pca_coeffs)
                        #fig, ax = plt.subplots()
                        #ax.plot(tspek.waves, trans, 'k-')
                        #pixels = np.arange(trans.size)
                        #trans2 = np.interp(pixels + pixel_shift,
                        #                  pixels, ext_std.TELLABS, left=np.nan,
                        #                  right=np.nan)
                        #ax.plot(tspek.waves, trans2, 'b-')
                        #plt.show()
                    else:
                        trans = ext_std.TELLABS
                        if pixel_shift:
                            pixels = np.arange(trans.size)
                            trans = np.interp(pixels + pixel_shift,
                                              pixels, trans, left=np.nan,
                                              right=np.nan)

                    if delta_airmass:
                        pix_to_correct = np.logical_and(trans > 0, trans < 1)
                        tau = -np.log(trans[pix_to_correct]) / std_airmass
                        trans[pix_to_correct] *= np.exp(-tau * delta_airmass)

                    # Now that we have the best model, let's cross-correlate!
                    if xcorr:
                        pixel_shift = peak_finding.cross_correlate_subpixels(
                            ext.data, trans, sampling)
                        if pixel_shift is None:
                            log.warning("Cannot determine cross-correlation"
                                        f"peak for {ext.id}")
                            pixel_shift = None  # needs to exist for later
                        else:
                            log.stdinfo(f"Shift for extension {ext.id} is "
                                        f"{pixel_shift:.2f} pixels")
                            pixel_shifts.append(pixel_shift)

                    else:  # means we're actually correcting!
                        fig, ax = plt.subplots()
                        w = ext.wcs(np.arange(ext.data.size))
                        ax.plot(w, ext.data, 'b-')
                        ax.plot(w, trans*1e6, 'r-')
                        ax.plot(w, ext.data / trans, 'k-')
                        plt.show()

                        ext.divide(trans)

                        # We've only modified the WCS if using the model
                        if pixel_shift and apply_model and not apply_shift:
                            from_frame = ext.wcs.input_frame
                            to_frame = ext.wcs.available_frames[1]
                            ext.wcs.set_transform(from_frame, to_frame,
                                                  ext.wcs.get_transform(from_frame, to_frame)[1:])

                if xcorr and len(ad) > 1:
                    pixel_shift = self._calaculate_mean_pixel_shift(pixel_shifts)
                    if (pixel_shift is not None and
                            abs(pixel_shift) < shift_tolerance):
                        pixel_shift = 0
                        log.stdinfo(f"Shift is within {shift_tolerance}"
                                    " tolerance so will not be applied")

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

    def _line_spread_function(self, ext):
        cls_name = f"{ext.instrument().upper()}LineSpreadFunction"
        instance = getattr(lsf_classes, cls_name)(ext)
        return instance


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

    @abstractmethod
    def convolutions(self):
        """Returns list of (convolution function, kernel size in nm)"""

    def convolve_and_resample(self, waves, w, data, **kwargs):
        """
        Convolve and resample a spectrum/spectra to the same resolution and
        wavelength array as the main model

        Parameters
        ----------
        waves: array
            output wavelength scale
        w: array
            wavelengths of thing to be convolved
        data: array
            data to be convolved; last dimension must match w
        kwargs: dict
            list of parameters to be passed to the convolutions() method

        Returns
        -------
        array of shape of "m", but last dimension of shape "self.nwaves"
            the convolved and resampled models
        """
        convolution_list = self.convolutions(**kwargs)
        dw = sum(x[1] for x in convolution_list)
        w1, w2 = waves.min() - 1.05 * dw, waves.max() + 1.05 * dw
        windices = np.logical_and(w > w1, w < w2)
        spectra = data[..., windices]
        for conv_func, dw in convolution_list:
            spectra = convolution.convolve(w[windices], spectra,
                                           conv_func, dw=dw)
        return convolution.resample(waves, w[windices], spectra)


class GaussianLineSpreadFunction(LineSpreadFunction):
    """A generic Gaussian LSF with a constant resolution"""
    parameters = ["resolution"]

    def __init__(self, ext, resolution=None):
        super().__init__(ext)
        self.resolution = resolution

    def convolutions(self, resolution=None):
        if resolution is None:
            resolution = self.resolution
        gaussian_func = partial(convolution.gaussian_constant_r, r=resolution)
        gaussian_dw = 3 * self.all_waves.max() / resolution
        convolutions = [(gaussian_func, gaussian_dw)]
        return convolutions
