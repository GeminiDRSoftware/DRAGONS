# Copyright(c) 2016-2020 Association of Universities for Research in Astronomy, Inc.
#
#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
import os
import re
import warnings
from copy import copy
from functools import partial, reduce
from importlib import import_module

import matplotlib
import numpy as np
from astropy import units as u
from astropy.io.ascii.core import InconsistentTableError
from astropy.io.registry import IORegistryError
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import Model, fitting, models
from astropy.stats import sigma_clip
from astropy.table import Table, vstack, MaskedColumn
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from matplotlib import pyplot as plt
from numpy.ma.extras import _ezclump
from scipy import optimize
from specutils import SpectralRegion

import astrodata
import geminidr.interactive.server
from astrodata import AstroData
from geminidr import PrimitivesBASE
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gemini.lookups import extinction_data as extinct
from geminidr.interactive.fit import fit1d
from geminidr.interactive.fit.aperture import interactive_find_source_apertures
from geminidr.interactive.fit.tracing import interactive_trace_apertures
from geminidr.interactive.fit.wavecal import WavelengthSolutionVisualizer
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from gempy.library import astrotools as at
from gempy.library import tracing, transform, wavecal
from gempy.library.astrotools import array_from_list
from gempy.library.config import RangeField, Config
from gempy.library.fitting import fit_1D
from gempy.library.spectral import Spek1D
from recipe_system.utils.decorators import parameter_override

from . import parameters_spect
from ..interactive.fit.help import CALCULATE_SENSITIVITY_HELP_TEXT, SKY_CORRECT_FROM_SLIT_HELP_TEXT

matplotlib.rcParams.update({'figure.max_open_warning': 0})

# ------------------------------------------------------------------------------


# noinspection SpellCheckingInspection
@parameter_override
class Spect(PrimitivesBASE):
    """
    This is the class containing all of the pre-processing primitives
    for the `Spect` level of the type hierarchy tree.
    """
    tagset = {"GEMINI", "SPECT"}

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_spect)

    def adjustWCSToReference(self, adinputs=None, **params):
        """
        Compute offsets along the slit by cross-correlation, or use offset
        from the headers (QOFFSET). The computed offset is stored in the
        SLITOFF keyword.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 1D or 2D spectra.
        suffix : str
            Suffix to be added to output files
        method : str ['correlation' | 'offsets']
            Method to use to compute offsets. 'correlation' uses a
            correlation of the slit profiles (the 2d images stacked
            on the dispersion axis), 'offsets' uses the QOFFSET keyword.
        tolerance : float
            Maximum distance from the header offset, for the correlation
            method (arcsec). If the correlation computed offset is too
            different from the header offset, then the latter is used.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        methods = (params["method"], params["fallback"])
        tolerance = params["tolerance"]

        if len(adinputs) <= 1:
            log.warning("No correction will be performed, since at least two "
                        "input images are required")
            return adinputs

        if not all(len(ad) == 1 for ad in adinputs):
            raise OSError("All input images must have only one extension")

        if {len(ad[0].shape) for ad in adinputs} != {2}:
            raise OSError("All inputs must be two dimensional")

        # Use first image in list as reference
        refad = adinputs[0]
        ref_sky_model = am.get_named_submodel(refad[0].wcs.forward_transform, 'SKY').copy()
        ref_sky_model.name = None
        log.stdinfo(f"Reference image: {refad.filename}")
        refad.phu['SLITOFF'] = 0
        if any('sources' in m for m in methods):
            ref_profile = tracing.stack_slit(refad[0])
        if 'sources_wcs' in methods:
            world_coords = (refad[0].central_wavelength(asNanometers=True),
                            refad.target_ra(), refad.target_dec())
            ref_coords = refad[0].wcs.backward_transform(*world_coords)

        # The reference doesn't go through the loop so update it now
        gt.mark_history(adinputs[0], primname=self.myself(), keyword=timestamp_key)
        adinputs[0].update_filename(suffix=params["suffix"], strip=True)

        for ad in adinputs[1:]:
            for method in methods:
                if method is None:
                    break

                adjust = False
                dispaxis = 2 - ad[0].dispersion_axis()  # python sense

                # Calculate offset determined by header (WCS or offsets)
                if method == 'sources_wcs':
                    coords = ad[0].wcs.backward_transform(*world_coords)
                    hdr_offset = ref_coords[dispaxis] - coords[dispaxis]
                elif dispaxis == 1:
                    hdr_offset = refad.detector_y_offset() - ad.detector_y_offset()
                else:
                    hdr_offset = refad.detector_x_offset() - ad.detector_x_offset()

                # Cross-correlate to find real offset and compare. Only look
                # for a peak in the range defined by "tolerance".
                if 'sources' in method:
                    profile = tracing.stack_slit(ad[0])
                    corr = np.correlate(ref_profile, profile, mode='full')
                    expected_peak = corr.size // 2 + hdr_offset
                    peaks, snrs = tracing.find_peaks(corr, np.arange(3,20),
                                                     reject_bad=False, pinpoint_index=0)
                    if peaks.size:
                        if tolerance is None:
                            found_peak = peaks[snrs.argmax()]
                        else:
                            # Go down the peaks in order of decreasing SNR
                            # until we find one within "tolerance"
                            found_peak = None
                            for peak, snr in sorted(zip(peaks, snrs),
                                                    key=lambda pair: pair[1],
                                                    reverse=True):
                                if (abs(peak - expected_peak) <=
                                        tolerance / ad.pixel_scale()):
                                    found_peak = peak
                                    break
                        if found_peak:
                            # found_peak = tracing.pinpoint_peaks(corr, None, found_peak)[0]
                            offset = found_peak - ref_profile.shape[0] + 1
                            adjust = True
                        else:
                            log.warning("No cross-correlation peak found for "
                                        f"{ad.filename} within tolerance")
                    else:
                        log.warning(f"{ad.filename}: Cross-correlation failed")

                elif method == 'offsets':
                    offset = hdr_offset
                    adjust = True

                if adjust:
                    wcs = ad[0].wcs
                    frames = wcs.available_frames
                    for input_frame, output_frame in zip(frames[:-1], frames[1:]):
                        t = wcs.get_transform(input_frame, output_frame)
                        try:
                            sky_model = am.get_named_submodel(t, 'SKY')
                        except IndexError:
                            pass
                        else:
                            new_sky_model = models.Shift(offset) | ref_sky_model
                            new_sky_model.name = 'SKY'
                            ad[0].wcs.set_transform(input_frame, output_frame,
                                                    t.replace_submodel('SKY', new_sky_model))
                            break
                    else:
                        raise OSError("Cannot find 'SKY' model in WCS for "
                                      f"{ad.filename}")

                    log.stdinfo("Offset for image {} : {:.2f} pixels"
                                .format(ad.filename, offset))
                    ad.phu['SLITOFF'] = offset
                    break

            if not adjust:
                no_offset_msg = f"Cannot determine offset for {ad.filename}"
                if 'sq' in self.mode:
                    raise OSError(no_offset_msg)
                else:
                    log.warning(no_offset_msg)
            else:
                # Timestamp and update filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def calculateSensitivity(self, adinputs=None, **params):
        """
        Calculates the overall sensitivity of the observation system
        (instrument, telescope, detector, etc) for each wavelength using
        spectrophotometric data. It is obtained using the ratio
        between the observed data and the reference look-up data.

        For that, it looks for reference data using the stripped and lower
        case name of the observed object inside :mod:`geminidr.gemini.lookups`,
        :mod:`geminidr.core.lookups` and inside the instrument lookup module.

        The reference data is fit using a Spline in order to match the input
        data sampling.

        See Also
        --------
        - :class:`~gempy.library.astromodels.UnivariateSplineWithOutlierRemoval`

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            1D spectra of spectrophotometric standard stars

        suffix :  str, optional
            Suffix to be added to output files (default: _sensitivityCalculated).

        filename: str or None, optional
            Location of spectrophotometric data file. If it is None, uses
            look up data based on the object name stored in OBJECT header key
            (default).

        function : str
            type of function to fit (splineN or polynomial types)

        order : int
            Order of the spline fit to be performed

        lsigma, hsigma : float/None
            lower and upper rejection limit in standard deviations

        niter : int
            maximum number of rejection iterations

        bandpass : float, optional
            default bandpass width (in nm) to use if not present in the
            spectrophotometric data table (default: 5.)

        interactive: bool, optional
            Run the interactive UI for selecting the fit parameters

        individual : bool - TODO - Not in calculateSensitivityConfig
            Calculate sensitivity for each AD spectrum individually?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has a
            `.SENSFUNC` table appended to each of its extensions. This table
            provides details of the fit which describes the sensitivity as
            a function of wavelength.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        datafile = params["filename"]
        bandpass = params["bandpass"]
        airmass0 = params["debug_airmass0"]
        debug_plot = params["debug_plot"]
        interactive = params["interactive"]
        fit1d_params = fit_1D.translate_params(params)

        # We're going to look in the generic (gemini) module as well as the
        # instrument module, so define that
        module = self.inst_lookups.split('.')
        module[-2] = 'gemini'
        gemini_lookups = '.'.join(module)

        for ad in adinputs:
            if datafile is None:
                filename = '{}.dat'.format(ad.object().lower().replace(' ', ''))
                for module in (self.inst_lookups, gemini_lookups, 'geminidr.core.lookups'):
                    try:
                        path = import_module('.', module).__path__[0]
                    except (ImportError, ModuleNotFoundError):
                        continue
                    full_path = os.path.join(path, 'spectrophotometric_standards', filename)
                    try:
                        spec_table = self._get_spectrophotometry(full_path)
                    except (FileNotFoundError, InconsistentTableError):
                        pass
                    else:
                        break
                else:
                    log.warning("Cannot read spectrophotometric data table. "
                                "Unable to determine sensitivity for {}".
                                format(ad.filename))
                    continue
            else:
                try:
                    spec_table = self._get_spectrophotometry(datafile)
                except FileNotFoundError:
                    log.warning(f"Cannot find spectrophotometric data table {datafile}."
                                f"Unable to determine sensitivity for {ad.filename}")
                    continue
                except InconsistentTableError:
                    log.warning(f"Cannot read spectrophotometric data table {datafile}."
                                f"Unable to determine sensitivity for {ad.filename}")
                    continue

            exptime = ad.exposure_time()
            if 'WIDTH' not in spec_table.colnames:
                log.warning(f"Using default bandpass of {bandpass} nm")
                spec_table['WIDTH'] = bandpass * u.nm

            # Do some checks now to avoid failing on every extension
            if airmass0:
                telescope = ad.telescope()
                site = extinct.telescope_sites.get(telescope)
                if site is None:
                    log.warning(f"{ad.filename}: Cannot determine site of "
                                f"telescope '{telescope}' so cannot apply "
                                "correction to zero airmass.")
                else:
                    airmass = ad.airmass()
                    if airmass is None:
                        log.warning(f"{ad.filename}: Cannot determine airmass"
                                    " of observation so cannot apply "
                                    "correction to zero airmass.")
                    else:
                        log.stdinfo(f"{ad.filename}: Correcting from airmass "
                                    f"{airmass} to zero airmass using curve "
                                    f"for site '{site}'.")
                        ad.phu["EXTCURVE"] = (
                            site, self.keyword_comments["EXTCURVE"])

            def _get_fit1d_input_data(ext, exptime, spec_table):
                spectrum = Spek1D(ext) / (exptime * u.s)
                wave, zpt, zpt_err = [], [], []

                # Compute values that are counts / (exptime * flux_density * bandpass)
                for w0, dw, fluxdens in zip(spec_table['WAVELENGTH'].quantity,
                                            spec_table['WIDTH'].quantity, spec_table['FLUX'].quantity):
                    region = SpectralRegion(w0 - 0.5 * dw, w0 + 0.5 * dw)
                    data, mask, variance = spectrum.signal(
                        region, interpolate=DQ.bad_pixel)
                    if mask == 0 and fluxdens > 0:
                        # Regardless of whether FLUX column is f_nu or f_lambda
                        flux = fluxdens.to(u.Unit('erg cm-2 s-1 nm-1'),
                                           equivalencies=u.spectral_density(w0)) * dw.to(u.nm)
                        if data > 0:
                            wave.append(w0)
                            # This is (counts/s) / (erg/cm^2/s), in magnitudes (like IRAF)
                            zpt.append(u.Magnitude(flux / data))
                            if variance is not None:
                                zpt_err.append(np.log(1 + np.sqrt(variance) / data))
                wave = at.array_from_list(wave, unit=u.nm)
                zpt = at.array_from_list(zpt)
                weights = 1. / array_from_list(zpt_err) if zpt_err else None

                # Correct for atmospheric extinction. This correction makes
                # the real data brighter, so makes the zpt magnitude more +ve
                # (since "data" is in the denominator)
                if airmass0 and site is not None and airmass is not None:
                    zpt += u.Magnitude(airmass *
                                       extinct.extinction(wave, site=site))

                return wave, zpt, weights

            # We can only calculate the sensitivity for one extension in
            # non-XD data, so keep track of this in case it's not the first one
            calculated = False
            xunits = yunits = None
            all_exts = list()
            for ext in ad:
                if len(ext.shape) != 1:
                    log.warning(f"{ad.filename} extension {ext.id} is not a "
                                "1D spectrum")
                    continue

                if calculated and 'XD' not in ad.tags:
                    log.warning("Found additional 1D extensions in non-XD data."
                                " Ignoring.")
                    break

                waves, zpt, weights = _get_fit1d_input_data(ext, exptime, spec_table)
                if xunits is None:
                    xunits = waves.unit
                elif xunits != waves.unit:
                    log.warning(f"Unit mismatch between {xunits} and {waves.unit}")
                    continue
                if yunits is None:
                    yunits = zpt.unit
                elif xunits != zpt.unit:
                    log.warning(f"Unit mismatch between {yunits} and {zpt.unit}")
                    continue
                all_exts.append((ext, waves, zpt, weights))
                calculated = True

            if interactive:
                all_domains = [(0, x[0].shape[0]) for x in all_exts]
                all_waves = np.array([x[1] for x in all_exts])
                all_zpt = np.array([x[2] for x in all_exts])
                all_weights = np.array([x[3] for x in all_exts])
                all_fp_init = []
                for i in range(len(all_exts)):
                    all_fp_init.append(fit_1D.translate_params(params))

                # build config for interactive
                config = self.params[self.myself()]
                config.update(**params)

                # Get filename to display in visualizer
                filename_info = getattr(ad, 'filename', '')

                visualizer = fit1d.Fit1DVisualizer((all_waves, all_zpt, all_weights),
                                                   fitting_parameters=all_fp_init,
                                                   config=config,
                                                   tab_name_fmt="CCD {}",
                                                   xlabel=f'Wavelength ({xunits})',
                                                   ylabel=f'Sensitivity ({yunits})',
                                                   domains=all_domains,
                                                   title="Calculate Sensitivity",
                                                   primitive_name="calculateSensitivity",
                                                   filename_info=filename_info,
                                                   help_text=CALCULATE_SENSITIVITY_HELP_TEXT)
                geminidr.interactive.server.interactive_fitter(visualizer)

                all_m_final = visualizer.results()
                for (ext, _, _, _), fit in zip(all_exts, all_m_final):
                    ext.SENSFUNC = am.model_to_table(fit.model, xunit=xunits,
                                                     yunit=yunits)
            else:
                for (ext, waves, zpt, weights) in all_exts:
                    fit_1d = fit_1D(zpt.value, points=waves.value,
                                    weights=weights, **fit1d_params,
                                    plot=debug_plot)
                    ext.SENSFUNC = am.model_to_table(fit_1d.model, xunit=waves.unit,
                                                     yunit=zpt.unit)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def determineDistortion(self, adinputs=None, **params):
        """
        Maps the distortion on a detector by tracing lines perpendicular to the
        dispersion direction. Then it fits a 2D Chebyshev polynomial to the
        fitted coordinates in the dispersion direction. The distortion map does
        not change the coordinates in the spatial direction.

        The Chebyshev2D model is stored as part of a gWCS object in each
        `nddata.wcs` attribute, which gets mapped to a FITS table extension
        named `WCS` on disk.


        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Arc data as 2D spectral images with the distortion and wavelength
            solutions encoded in the WCS.

        suffix :  str
            Suffix to be added to output files.

        spatial_order : int
            Order of fit in spatial direction.

        spectral_order : int
            Order of fit in spectral direction.

        id_only : bool
            Trace using only those lines identified for wavelength calibration?

        min_snr : float
            Minimum signal-to-noise ratio for identifying lines (if
            id_only=False).

        nsum : int
            Number of rows/columns to sum at each step.

        step : int
            Size of step in pixels when tracing.

        max_shift : float
            Maximum orthogonal shift (per pixel) for line-tracing (unbinned).

        max_missed : int
            Maximum number of steps to miss before a line is lost.

        debug: bool
            plot arc line traces on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has the
            appropriate `nddata.wcs` defined for each of its extensions. This
            provides details of the 2D Chebyshev fit which maps the distortion.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        spatial_order = params["spatial_order"]
        spectral_order = params["spectral_order"]
        id_only = params["id_only"]
        fwidth = params["fwidth"]
        min_snr = params["min_snr"]
        nsum = params["nsum"]
        step = params["step"]
        max_shift = params["max_shift"]
        max_missed = params["max_missed"]
        debug = params["debug"]

        orders = (spectral_order, spatial_order)

        for ad in adinputs:
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            for ext in ad:
                if debug:
                    self.viewer.display_image(ext, wcs=False)
                    self.viewer.width = 2

                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"

                # Here's a lot of input-checking
                extname = f'{ad.filename} extension {ext.id}'
                start = ext.shape[1 - dispaxis] // 2
                initial_peaks = None
                try:
                    wavecal = ext.WAVECAL
                except AttributeError:
                    log.warning("Cannot find a WAVECAL table on {} - "
                                "identifying lines in middle {}".
                                format(extname, direction))
                else:
                    try:
                        index = list(wavecal['name']).index(direction)
                    except ValueError:
                        log.warning("Cannot find starting {} in WAVECAL "
                                    "table on {} - identifying lines in "
                                    "middle {}. Wavelength calibration may "
                                    "not be correct.".format(direction, extname,
                                                             direction))
                    else:
                        start = int(wavecal['coefficients'][index])
                    if id_only:
                        try:
                            # Peak locations in pixels are 1-indexed
                            initial_peaks = (ext.WAVECAL['peaks'] - 1)
                        except KeyError:
                            log.warning("Cannot find peak locations in {} "
                                        "- identifying lines in middle {}".
                                        format(extname, direction))
                    if fwidth is None:
                        try:
                            index = list(wavecal['name']).index('fwidth')
                        except ValueError:
                            pass
                        else:
                            fwidth = float(wavecal['coefficients'][index])

                # This is identical to the code in determineWavelengthSolution()
                if fwidth is None:
                    data, _, _, _ = tracing.average_along_slit(ext, center=None, nsum=nsum)
                    fwidth = tracing.estimate_peak_width(data)
                    log.stdinfo(f"Estimated feature width: {fwidth:.2f} pixels")

                if initial_peaks is None:
                    data, mask, variance, extract_slice = tracing.average_along_slit(ext, center=None, nsum=nsum)
                    log.stdinfo("Finding peaks by extracting {}s {} to {}".
                                format(direction, extract_slice.start + 1, extract_slice.stop))

                    # Find peaks; convert width FWHM to sigma
                    widths = 0.42466 * fwidth * np.arange(0.75, 1.26, 0.05)  # TODO!
                    initial_peaks, _ = tracing.find_peaks(data, widths, mask=mask & DQ.not_signal,
                                                          variance=variance, min_snr=min_snr)
                    log.stdinfo(f"Found {len(initial_peaks)} peaks")

                # The coordinates are always returned as (x-coords, y-coords)
                rwidth = 0.42466 * fwidth
                ref_coords, in_coords = tracing.trace_lines(ext, axis=1 - dispaxis,
                                                            start=start, initial=initial_peaks,
                                                            rwidth=rwidth, cwidth=max(int(fwidth), 5), step=step,
                                                            nsum=nsum, max_missed=max_missed,
                                                            max_shift=max_shift * ybin / xbin,
                                                            viewer=self.viewer if debug else None)

                ## These coordinates need to be in the reference frame of a
                ## full-frame unbinned image, so modify the coordinates by
                ## the detector section
                # x1, x2, y1, y2 = ext.detector_section()
                # ref_coords = np.array([ref_coords[0] * xbin + x1,
                #                       ref_coords[1] * ybin + y1])
                # in_coords = np.array([in_coords[0] * xbin + x1,
                #                      in_coords[1] * ybin + y1])

                # The model is computed entirely in the pixel coordinate frame
                # of the data, so it could be used as a gWCS object
                m_init = models.Chebyshev2D(x_degree=orders[1 - dispaxis],
                                            y_degree=orders[dispaxis],
                                            x_domain=[0, ext.shape[1]],
                                            y_domain=[0, ext.shape[0]])
                # x_domain = [x1, x1 + ext.shape[1] * xbin - 1],
                # y_domain = [y1, y1 + ext.shape[0] * ybin - 1])
                # Find model to transform actual (x,y) locations to the
                # value of the reference pixel along the dispersion axis
                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           sigma_clip, sigma=3)
                m_final, _ = fit_it(m_init, *in_coords, ref_coords[1 - dispaxis])
                m_inverse, masked = fit_it(m_init, *ref_coords, in_coords[1 - dispaxis])

                # TODO: Some logging about quality of fit
                # print(np.min(diff), np.max(diff), np.std(diff))

                if dispaxis == 1:
                    model = models.Mapping((0, 1, 1)) | (m_final & models.Identity(1))
                    model.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
                else:
                    model = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_final)
                    model.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)

                self.viewer.color = "blue"
                spatial_coords = np.linspace(ref_coords[dispaxis].min(), ref_coords[dispaxis].max(),
                                             ext.shape[1 - dispaxis] // (step * 10))
                spectral_coords = np.unique(ref_coords[1 - dispaxis])
                for coord in spectral_coords:
                    if dispaxis == 1:
                        xref = [coord] * len(spatial_coords)
                        yref = spatial_coords
                    else:
                        xref = spatial_coords
                        yref = [coord] * len(spatial_coords)
                    mapped_coords = np.array(model.inverse(xref, yref)).T
                    if debug:
                        self.viewer.polygon(mapped_coords, closed=False, xfirst=True, origin=0)

                # This is all we need for the new FITCOORD table
                ext.FITCOORD = vstack([am.model_to_table(m_final),
                                       am.model_to_table(m_inverse)],
                                      metadata_conflicts="silent")

                # Put this model before the first step if there's an existing WCS
                if ext.wcs is None:
                    ext.wcs = gWCS([(cf.Frame2D(name="pixels"), model),
                                    (cf.Frame2D(name="world"), None)])
                else:
                    # TODO: use insert_frame here
                    ext.wcs = gWCS([(ext.wcs.input_frame, model),
                                    (cf.Frame2D(name="distortion_corrected"), ext.wcs.pipeline[0][1])]
                                   + ext.wcs.pipeline[1:])

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects optical distortion in science frames using a `processed_arc`
        with attached distortion map (a Chebyshev2D model).

        If the input image requires mosaicking, then this is done as part of
        the resampling, to ensure one, rather than two, interpolations.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images.
        suffix : str
            Suffix to be added to output files.
        arc : :class:`~astrodata.AstroData` or str or None
            Arc(s) containing distortion map.
        order : int (0 - 5)
            Order of interpolation when resampling.
        subsample : int
            Pixel subsampling factor.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Modified input objects with distortion correct applied.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        arc = params["arc"]
        order = params["order"]
        subsample = params["subsample"]
        do_cal = params["do_cal"]

        if do_cal == 'skip':
            log.warning('Distortion correction has been turned off.')
            return adinputs

        # Get a suitable arc frame (with distortion map) for every science AD
        if arc is None:
            arc_list = self.caldb.get_processed_arc(adinputs)
        else:
            arc_list = (arc, None)

        adoutputs = []
        # Provide an arc AD object for every science frame, and an origin
        for ad, arc, origin in zip(*gt.make_lists(adinputs, *arc_list,
                                                  force_ad=(1,))):
            # We don't check for a timestamp since it's not unreasonable
            # to do multiple distortion corrections on a single AD object

            len_ad = len(ad)
            if arc is None:
                if 'sq' not in self.mode and do_cal != 'force':
                    # TODO: Think about this when we have MOS/XD/IFU
                    if len(ad) == 1:
                        log.warning(f"{ad.filename}: no arc was specified. "
                                    "Continuing.")
                        adoutputs.append(ad)
                    else:
                        log.warning(f"{ad.filename}: no arc was specified. "
                                    "Image will be mosaicked.")
                        adoutputs.extend(self.mosaicDetectors([ad]))
                    continue

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the arc {arc.filename}"
                        f"{origin_str}")
            len_arc = len(arc)
            if len_arc not in (1, len_ad):
                log.warning(f"{ad.filename} has {len_ad} extensions but "
                            f"{arc.filename} had {len_arc} extensions so "
                            "cannot correct the distortion.")
                adoutputs.append(ad)
                continue

            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            if arc.detector_x_bin() != xbin or arc.detector_y_bin() != ybin:
                log.warning("Science frame and arc have different binnings.")
                adoutputs.append(ad)
                continue

            # Read all the arc's distortion maps. Do this now so we only have
            # one block of reading and verifying them
            distortion_models, wave_models, wave_frames = [], [], []
            for ext in arc:
                wcs = ext.nddata.wcs

                # Any failures must be handled in the outer loop processing
                # ADs, so just set the found transforms to empty and present
                # the warning at the end
                try:
                    if 'distortion_corrected' in wcs.available_frames:
                        pipeline = wcs.pipeline
                    else:
                        distortion_models = []
                        break
                except AttributeError:
                    distortion_models = []
                    break

                # We could just pass the forward transform of the WCS, which
                # already has its inverse attached, but the code currently
                # relies on a no-op forward model to size the output correctly:
                m_distcorr = models.Identity(2)
                input_frame = wcs.input_frame
                m_distcorr.inverse = wcs.get_transform(
                    input_frame, 'distortion_corrected').inverse
                distortion_models.append(m_distcorr)

                try:
                    wave_model = am.get_named_submodel(wcs.forward_transform,
                                                       'WAVE')
                except IndexError:
                    wave_models.append(None)
                    wave_frames.append(None)
                else:
                    wave_models.append(wave_model)
                    wave_frames.extend([frame for frame in wcs.output_frame.frames
                                        if isinstance(frame, cf.SpectralFrame)])

            if not distortion_models:
                log.warning("Could not find a 'distortion_corrected' frame "
                            f"in arc {arc.filename} extension {ext.id} - "
                            "continuing")
                continue

            # Determine whether we're producing a single-extension AD
            # or keeping the number of extensions as-is
            if len_arc == 1:
                arc_detsec = arc.detector_section()[0]
                ad_detsec = ad.detector_section()
                if len_ad > 1:
                    # We need to apply the mosaicking geometry, and add the
                    # same distortion correction to each input extension.
                    geotable = import_module('.geometry_conf', self.inst_lookups)
                    transform.add_mosaic_wcs(ad, geotable)
                    for ext in ad:
                        # TODO: use insert_frame() method
                        new_pipeline = []
                        for item in ext.wcs.pipeline:
                            if item[0].name == 'mosaic':
                                new_pipeline.extend([(item[0], m_distcorr),
                                                     (cf.Frame2D(name='distortion_corrected'), item[1])])
                            else:
                                new_pipeline.append(item)
                        ext.wcs = gWCS(new_pipeline)

                    # We need to consider the different pixel frames of the
                    # science and arc. The input->mosaic transform of the
                    # science maps to the default pixel space, but the arc
                    # will have had an origin shift before the distortion
                    # correction was calculated.
                    shifts = [c2 - c1 for c1, c2 in zip(np.array(ad_detsec).min(axis=0),
                                                        arc_detsec)]
                    xoff1, yoff1 = shifts[0] / xbin, shifts[2] / ybin  # x1, y1
                    if xoff1 or yoff1:
                        log.debug(f"Found a shift of ({xoff1},{yoff1}) "
                                  f"pixels between {ad.filename} and the "
                                  f"calibration {arc.filename}")
                    shifts = [c2 - c1 for c1, c2 in zip(np.array(ad_detsec).max(axis=0),
                                                        arc_detsec)]
                    xoff2, yoff2 = shifts[1] / xbin, shifts[3] / ybin  # x2, y2
                    nzeros = [xoff1, xoff2, yoff1, yoff2].count(0)
                    if nzeros < 2:
                        raise ValueError("I don't know how to process the "
                                         f"offsets between {ad.filename} "
                                         f"and {arc.filename}")

                    arc_ext_shapes = [(ext.shape[0] - yoff1 + yoff2,
                                       ext.shape[1] - xoff1 + xoff2) for ext in ad]
                    arc_corners = np.concatenate([transform.get_output_corners(
                        ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic'),
                        input_shape=arc_shape, origin=(yoff1, xoff1))
                        for ext, arc_shape in zip(ad, arc_ext_shapes)], axis=1)
                    arc_origin = tuple(np.ceil(min(corners)) for corners in arc_corners)

                    # So this is what was applied to the ARC to get the
                    # mosaic frame to its pixel frame, in which the distortion
                    # correction model was calculated. Convert coordinates
                    # from python order to Model order.
                    origin_shift = reduce(Model.__and__, [models.Shift(-origin)
                                          for origin in arc_origin[::-1]])

                    for ext in ad:
                        ext.wcs.insert_transform('mosaic', origin_shift, after=True)
                        #ext.wcs.insert_transform('distortion_corrected',
                        #                         origin_shift.inverse, after=False)

                    # ARC and AD aren't the same size
                    if nzeros < 4:
                        ad_corners = np.concatenate([transform.get_output_corners(
                            ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic'),
                            input_shape=ext.shape) for ext in ad], axis=1)
                        ad_origin = tuple(np.ceil(min(corners)) for corners in ad_corners)

                        # But a full-frame ARC and subregion AD may have different
                        # origin shifts. We only care about the one in the
                        # wavelength direction, since we need the AD to be on the
                        # same pixel basis before applying the new wave_model
                        offsets = tuple(o_ad - o_arc
                                        for o_ad, o_arc in zip(ad_origin, arc_origin))[::-1]
                        # len(arc)=1 so we only have one wave_model, but need to
                        # update the entry in the list, which gets used later
                        if wave_model is not None:
                            offset = offsets[ext.dispersion_axis()-1]
                            if offset != 0:
                                wave_model.name = None
                                wave_models[0] = models.Shift(offset) | wave_model
                                wave_models[0].name = 'WAVE'

                else:
                    # Single-extension AD, with single Transform
                    ad_detsec = ad.detector_section()[0]
                    if ad_detsec != arc_detsec:
                        if self.timestamp_keys['mosaicDetectors'] in ad.phu:
                            log.warning("Cannot distortionCorrect mosaicked "
                                        "data unless calibration has the "
                                        "same ROI. Continuing.")
                            adoutputs.append(ad)
                            continue
                        # No mosaicking, so we can just do a shift
                        m_shift = (models.Shift((ad_detsec.x1 - arc_detsec.x1) / xbin) &
                                   models.Shift((ad_detsec.y1 - arc_detsec.y1) / ybin))
                        m_distcorr = m_shift | m_distcorr
                    # TODO: use insert_frame method
                    new_pipeline = [(ad[0].wcs.input_frame, m_distcorr),
                                    (cf.Frame2D(name='distortion_corrected'), ad[0].wcs.pipeline[0][1])]
                    new_pipeline.extend(ad[0].wcs.pipeline[1:])
                    ad[0].wcs = gWCS(new_pipeline)

                ad_out = transform.resample_from_wcs(ad, 'distortion_corrected',
                                                     order=order, subsample=subsample,
                                                     parallel=False)

                if wave_model is None:
                    log.warning(f"{arc.filename} has no wavelength solution")

            else:
                log.warning("Distortion correction with multiple-extension "
                            "arcs has not been tested.")
                for i, (ext, ext_arc, dist_model) in enumerate(zip(ad, arc, distortion_models)):
                    # Shift science so its pixel coords match the arc's before
                    # applying the distortion correction
                    shifts = [c1 - c2 for c1, c2 in zip(ext.detector_section(),
                                                        ext_arc.detector_section())]
                    dist_model = (models.Shift(shifts[0] / xbin) &
                                  models.Shift(shifts[1] / ybin)) | dist_model
                    # TODO: use insert_frame method
                    new_pipeline = [(ext.wcs.input_frame, dist_model),
                                    (cf.Frame2D(name='distortion_corrected'), ext.wcs.pipeline[0][1])]
                    new_pipeline.extend(ext.wcs.pipeline[1:])
                    ext.wcs = gWCS(new_pipeline)
                    if i == 0:
                        ad_out = transform.resample_from_wcs(ext, order=order,
                                                             subsample=subsample,
                                                             parallel=False)
                    else:
                        ad_out.append(transform.resample_from_wcs(ext, order=order,
                                                                  subsample=subsample,
                                                                  parallel=False))
                    if wave_model is None:
                        log.warning(f"{arc.filename} extension {ext.id} has "
                                    "no wavelength solution")

            for i, (ext, wave_model, wave_frame) in enumerate(
                    zip(ad_out, wave_models, wave_frames)):
                # TODO: remove this; for debugging purposes only
                if arc is not None and hasattr(arc[i], "WAVECAL"):
                    ad_out[i].WAVECAL = arc[i].WAVECAL
                sky_model = am.get_named_submodel(ext.wcs.forward_transform, 'SKY')
                if ext.dispersion_axis() == 1:
                    t = wave_model & sky_model
                else:
                    t = sky_model & wave_model
                # We need to create a new output frame with a copy of the
                # ARC's SpectralFrame in case there's a change from air->vac
                # or vice versa
                new_output_frame = cf.CompositeFrame(
                    [copy(wave_frame) if isinstance(frame, cf.SpectralFrame) else
                     frame for frame in ext.wcs.output_frame.frames])
                ext.wcs = gWCS([(ext.wcs.input_frame, t),
                                (new_output_frame, None)])
            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        Determines the wavelength solution for an ARC and updates the wcs
        with this solution. In addition, the solution and pixel/wavelength
        matches are stored as an attached `WAVECAL` :class:`~astropy.table.Table`.

        2D input images are converted to 1D by collapsing a slice of the image
        along the dispersion direction, and peaks are identified. These are then
        matched to an arc line list, using piecewise-fitting of (usually)
        linear functions to match peaks to arc lines, using the
        :class:`~gempy.library.matching.KDTreeFitter`.

        The `.WAVECAL` table contains four columns:
            ["name", "coefficients", "peaks", "wavelengths"]

        The `name` and the `coefficients` columns contain information to
        re-create an Chebyshev1D object, plus additional information about
        the way the spectrum was collapsed. The `peaks` column contains the
        (1-indexed) position of the lines that were matched to the catalogue,
        and the `wavelengths` column contains the matched wavelengths.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
             Mosaicked Arc data as 2D spectral images or 1D spectra.

        suffix : str/None
            Suffix to be added to output files

        order : int
            Order of Chebyshev fitting function.

        center : None or int
            Central row/column for 1D extraction (None => use middle).

        nsum : int, optional
            Number of rows/columns to average.

        min_snr : float
            Minimum S/N ratio in line peak to be used in fitting.

        weighting : {'natural', 'relative', 'none'}
            How to weight the detected peaks.

        fwidth : float/None
            Expected width of arc lines in pixels. It tells how far the
            KDTreeFitter should look for when matching detected peaks with
            reference arcs lines. If None, `fwidth` is determined using
            `tracing.estimate_peak_width`.

        min_sep : float
            Minimum separation (in pixels) for peaks to be considered distinct

        central_wavelength : float/None
            central wavelength in nm (if None, use the WCS or descriptor)

        dispersion : float/None
            dispersion in nm/pixel (if None, use the WCS or descriptor)

        linelist : str/None
            Name of file containing arc lines. If None, then a default look-up
            table will be used.

        alternative_centers : bool
            Identify alternative central wavelengths and try to fit them?

        nbright : int (or may not exist in certain class methods)
            Number of brightest lines to cull before fitting

        interactive : bool
            Use the interactive tool?

        debug : bool
            Enable plots for debugging.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with a `.WAVECAL` attribute and improved wcs for
            each slice

        See Also
        --------
        :class:`~geminidr.core.primitives_visualize.Visualize.mosaicDetectors`,
        :class:`~gempy.library.matching.KDTreeFitter`,
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        arc_file = params["linelist"]
        interactive = params["interactive"]

        # TODO: This decision would prevent MOS data being reduced so need
        # to think a bit more about what we're going to do. Maybe make
        # central_wavelength() return a one-per-ext list? Or have the GMOS
        # determineWavelengthSolution() recipe check the input has been
        # mosaicked before calling super()?
        #
        # Top-level decision for this to only work on single-extension ADs
        # if not all(len(ad)==1 for ad in adinputs):
        #    raise ValueError("Not all inputs are single-extension AD objects")
        linelist = None
        if arc_file is not None:
            try:
                linelist = wavecal.LineList(arc_file)
            except OSError:
                log.warning(f"Cannot read file {arc_file} - "
                            "using default linelist")
            else:
                log.stdinfo(f"Read arc line list {arc_file}")

        # Pass the primitive configuration to the interactive object.
        config = copy(self.params[self.myself()])
        config.update(**params)

        for ad in adinputs:
            log.info(f"Determining wavelength solution for {ad.filename}")

            if interactive:
                all_fp_init = [fit_1D.translate_params(
                    {**params, "function": "chebyshev"})] * len(ad)
                # This feels like I shouldn't have to do it here
                domains = []
                for ext in ad:
                    axis = 0 if ext.data.ndim == 1 else 2 - ext.dispersion_axis()
                    domains.append([0, ext.shape[axis] - 1])
                reconstruct_points = partial(wavecal.create_interactive_inputs, ad, p=self,
                            linelist=linelist, bad_bits=DQ.not_signal)
                visualizer = WavelengthSolutionVisualizer(
                    reconstruct_points,
                    all_fp_init, config=config,
                    reinit_params=[#"center",
                                   "nsum", "min_snr", "min_sep",
                                   #"fwidth", "central_wavelength", "dispersion"
                                   ],
                    modal_message="Hang on, this stuff is tricky",
                    tab_name_fmt="Slit {}",
                    xlabel="Fitted wavelength (nm)", ylabel="Non-linear component (nm)",
                    domains=domains,
                    title="Wavelength Solution",
                    primitive_name=self.myself(),
                    filename_info=ad.filename,
                    enable_regions=False, plot_ratios=False, plot_height=350)
                geminidr.interactive.server.interactive_fitter(visualizer)
                for ext, fit1d, image, other in zip(ad, visualizer.results(),
                                             visualizer.image, visualizer.other_data):
                    fit1d.image = image
                    wavecal.update_wcs_with_solution(ext, fit1d, other, config)
            else:
                for ext in ad:
                    if len(ad) > 1:
                        log.info(f"Determining solution for extension {ext.id}")

                    input_data, fit1d, acceptable_fit = wavecal.get_automated_fit(
                        ext, config, p=self, linelist=linelist, bad_bits=DQ.not_signal)
                    if not acceptable_fit:
                        log.warning("No acceptable wavelength solution found "
                                    f"for {ext.id}")

                    wavecal.update_wcs_with_solution(ext, fit1d, input_data, config)
                    wavecal.save_fit_as_pdf(input_data["spectrum"], fit1d.points[~fit1d.mask],
                                            fit1d.image[~fit1d.mask], ad.filename)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def extract1DSpectra(self, adinputs=None, **params):
        """
        Extracts one or more 1D spectra from a 2D spectral image, according to
        the contents of the `.APERTURE` table.

        If the `skyCorrectFromSlit()` primitive has not been performed, then a
        1D sky spectrum is constructed from a nearby region of the image, and
        subtracted from the source spectrum.

        Each 1D spectrum is stored as a separate extension in a new AstroData
        object with the wcs copied from the parent.

        These new AD objects are placed in a separate stream from the
        parent 2D images, which are returned in the default stream.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images with a `.APERTURE` table.
        suffix : str
            Suffix to be added to output files.
        method : {'standard', 'weighted', 'optimal'}
            Extraction method.
        width : float or None
            Width of extraction aperture in pixels.
        grow : float
            Avoidance region around each source aperture if a sky aperture
            is required. Default: 10.
        subtract_sky : bool
            Extract and subtract sky spectra from object spectra if the 2D
            spectral image has not been sky subtracted?
        debug: bool
            draw apertures on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Extracted spectra as 1D data.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        method = params["method"]
        width = params["width"]
        grow = params["grow"]
        subtract_sky = params["subtract_sky"]
        debug = params["debug"]

        colors = ("green", "blue", "red", "yellow", "cyan", "magenta")
        offset_step = 2

        ad_extracted = []
        # This is just cut-and-paste code from determineWavelengthSolution()
        for ad in adinputs:
            ad_spec = astrodata.create(ad.phu)
            ad_spec.filename = ad.filename
            ad_spec.orig_filename = ad.orig_filename
            skysub_needed = (subtract_sky and
                             self.timestamp_keys['skyCorrectFromSlit'] not in ad.phu)
            if skysub_needed:
                log.stdinfo(f"Sky subtraction has not been performed on {ad.filename}"
                            " - extracting sky from separate apertures")

            kw_to_delete = [ad._keyword_for(desc)
                            for desc in ("detector_section", "array_section")]
            kw_datasec = ad._keyword_for("data_section")

            for ext in ad:
                extname = f"{ad.filename} extension {ext.id}"
                if debug:
                    self.viewer.display_image(ext, wcs=False)
                if len(ext.shape) == 1:
                    log.warning(f"{extname} is already one-dimensional")
                    continue

                try:
                    aptable = ext.APERTURE
                except AttributeError:
                    log.warning(f"{extname} has no APERTURE table. Cannot "
                                "extract spectra.")
                    continue

                num_spec = len(aptable)
                if num_spec == 0:
                    log.warning(f"{ad.filename} has an empty APERTURE table. "
                                "Cannot extract spectra.")
                    continue

                try:
                    wave_model = am.get_named_submodel(ext.wcs.forward_transform, 'WAVE')
                except (AttributeError, IndexError):
                    log.warning(f"Cannot find wavelength solution for {extname}")
                    wave_model = None
                else:
                    axes_names = tuple(frame.axes_names[0]
                                       for frame in ext.wcs.output_frame.frames
                                       if isinstance(frame, cf.SpectralFrame))
                    if len(axes_names) != 1:
                        log.warning("Problem with identifying spectral axis "
                                    f"for {extname}")

                log.stdinfo(f"Extracting {num_spec} spectra from {extname}")
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"

                # We loop twice so we can construct the aperture mask if needed
                apertures = []
                for row in aptable:
                    trace_model = am.table_to_model(row)
                    aperture = tracing.Aperture(trace_model,
                                                aper_lower=row['aper_lower'],
                                                aper_upper=row['aper_upper'])
                    if width is not None:
                        aperture.width = width
                    apertures.append(aperture)

                if skysub_needed:
                    apmask = np.logical_or.reduce([ap.aperture_mask(ext, width=width, grow=grow)
                                                   for ap in apertures])

                for apnum, aperture in enumerate(apertures, start=1):
                    log.stdinfo(f"    Extracting spectrum from aperture {apnum}")
                    self.viewer.width = 2
                    self.viewer.color = colors[(apnum-1) % len(colors)]
                    ndd_spec = aperture.extract(ext, width=width,
                                                method=method, viewer=self.viewer if debug else None)

                    # This whole (rather large) section is an attempt to ensure
                    # that sky apertures don't overlap with source apertures
                    if skysub_needed:
                        self.viewer.width = 1
                        # We're going to try to create half-size apertures
                        # equidistant from the source aperture on both sides
                        sky_width = 0.5 * aperture.width
                        sky_spectra = []

                        min_, max_ = aperture.limits()
                        for direction in (-1, 1):
                            offset = (direction * (0.5 * sky_width + grow) +
                                      (aperture.aper_upper if direction > 0 else aperture.aper_lower))
                            ok = False
                            while not ok:
                                if ((min_ + offset - 0.5 * sky_width < -0.5) or
                                        (max_ + offset + 0.5 * sky_width > ext.shape[1 - dispaxis] - 0.5)):
                                    break

                                sky_trace_model = aperture.model | models.Shift(offset)
                                sky_aperture = tracing.Aperture(sky_trace_model)
                                sky_spec = sky_aperture.extract(apmask, width=sky_width, dispaxis=dispaxis)
                                if np.sum(sky_spec.data) == 0:
                                    sky_spectra.append(sky_aperture.extract(ext, width=sky_width,
                                                                            viewer=self.viewer if debug else None))
                                    ok = True
                                offset += direction * offset_step

                        if sky_spectra:
                            # If only one, add it to itself (since it's half-width)
                            sky_spec = sky_spectra[0].add(sky_spectra[-1])
                            ad_spec.append(ndd_spec.subtract(sky_spec, handle_meta='first_found',
                                                             handle_mask=np.bitwise_or))
                        else:
                            log.warning("Difficulty finding sky aperture. No sky"
                                        f" subtraction for aperture {apnum}")
                            ad_spec.append(ndd_spec)
                    else:
                        ad_spec.append(ndd_spec)

                    # Create a new gWCS and add header keywords with the
                    # extraction location. All extracted spectra will have the
                    # same gWCS but that could change.
                    ext_spec = ad_spec[-1]
                    if wave_model is not None:
                        in_frame = cf.CoordinateFrame(naxes=1, axes_type=['SPATIAL'],
                                                      axes_order=(0,), unit=u.pix,
                                                      axes_names=('x',), name='pixels')
                        out_frame = cf.SpectralFrame(unit=u.nm, name='world',
                                                     axes_names=axes_names)
                        ext_spec.wcs = gWCS([(in_frame, wave_model),
                                             (out_frame, None)])
                    ext_spec.hdr[ad._keyword_for('aperture_number')] = apnum
                    center = aperture.model.c0.value
                    ext_spec.hdr['XTRACTED'] = (center, "Spectrum extracted "
                                                        "from {} {}".format(direction, int(center + 0.5)))
                    ext_spec.hdr['XTRACTLO'] = (aperture._last_extraction[0],
                                                'Aperture lower limit')
                    ext_spec.hdr['XTRACTHI'] = (aperture._last_extraction[1],
                                                'Aperture upper limit')

                    # Delete unnecessary keywords
                    for kw in kw_to_delete:
                        if kw in ext_spec.hdr:
                            del ext_spec.hdr[kw]

                    ext_spec.hdr[kw_datasec] = f"[1:{ext_spec.data.size}]"

            # Don't output a file with no extracted spectra
            if len(ad_spec) > 0:
                try:
                    del ad_spec.hdr['RADECSYS']
                except KeyError:
                    pass
                gt.mark_history(ad_spec, primname=self.myself(), keyword=timestamp_key)
                ad_spec.update_filename(suffix=sfx, strip=True)
                ad_extracted.append(ad_spec)

        # Only return extracted spectra
        return ad_extracted

    def findSourceApertures(self, adinputs=None, **params):
        """
        Finds sources in 2D spectral images and store them in an APERTURE table
        for each extension. Each table will, then, be used in later primitives
        to perform aperture extraction.

        The primitive operates by first collapsing the 2D spectral image in
        the spatial direction to identify sky lines as regions of high
        pixel-to-pixel variance, and the regions between the sky lines which
        consist of at least `min_sky_region` pixels are selected. These are
        then collapsed in the dispersion direction to produce a 1D spatial
        profile, from which sources are identified using a peak-finding
        algorithm.

        The widths of the apertures are determined by calculating a threshold
        level relative to the peak, or an integrated flux relative to the total
        between the minima on either side and determining where a smoothed
        version of the source profile reaches this threshold.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images.
        suffix : str
            Suffix to be added to output files.
        max_apertures : int
            Maximum number of apertures expected to be found.
        percentile : float (0 - 100) / None
            percentile to use when collapsing along the dispersion direction
            to obtain a slit profile / None => take mean
        section : str
            comma-separated list of colon-separated pixel coordinate pairs
            indicating the region(s) over which the spectral signal should be
            used. The first and last values can be blank, indicating to
            continue to the end of the data
        min_sky_region : int
            minimum number of contiguous pixels between sky lines
            for a region to be added to the spectrum before collapsing to 1D.
        use_snr : bool
            Convert data to SNR per pixel before collapsing and peak-finding?
        threshold : float (0 - 1)
            parameter describing either the height above background (relative
            to peak) or the integral under the spectrum (relative to the
            integral to the next minimum) at which to define the edges of
            the aperture.
        sizing_method : str ("peak" or "integral")
            which method to use
        interactive : bool
            Show interactive controls for fine tuning source aperture detection

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The 2D spectral images with APERTURE tables attached

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.determineDistortion`,
        :meth:`~geminidr.cofe.primitives_spect.Spect.distortionCorrect`
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        interactive = params["interactive"]

        aper_params = {key: params[key] for key in (
            'max_apertures', 'min_sky_region', 'percentile',
            'section', 'sizing_method', 'threshold', 'use_snr')}

        for ad in adinputs:
            if self.timestamp_keys['distortionCorrect'] not in ad.phu:
                log.warning(f"{ad.filename} has not been distortion corrected")

            for ext in ad:
                log.stdinfo(f"Searching for sources in {ad.filename} "
                            f"extension {ext.id}")

                dispaxis = 2 - ext.dispersion_axis()  # python sense
                npix = ext.shape[dispaxis]

                # data, mask, variance are all arrays in the GMOS orientation
                # with spectra dispersed horizontally
                if dispaxis == 0:
                    ext = ext.transpose()

                aper_params['direction'] = "column" if dispaxis == 0 else "row"

                if interactive:
                    locations, all_limits = interactive_find_source_apertures(
                        ext, **aper_params)
                else:
                    locations, all_limits, _, _ = tracing.find_apertures(
                        ext, **aper_params)

                if locations is None or len(locations) == 0:
                    # Delete existing APERTURE table
                    if 'APERTURE' in ext.tables:
                        del ext.APERTURE
                    continue

                # This is a little convoluted because of the simplicity of the
                # initial models, but we want to ensure that the APERTURE
                # table is written in an identical way to other models, and so
                # we should use the model_to_table() function
                all_tables = []
                for i, (loc, limits) in enumerate(zip(locations, all_limits),
                                                  start=1):
                    cheb = models.Chebyshev1D(degree=0, domain=[0, npix - 1],
                                              c0=loc)
                    aptable = am.model_to_table(cheb)
                    lower, upper = limits - loc
                    aptable["number"] = i
                    aptable["aper_lower"] = lower
                    aptable["aper_upper"] = upper
                    all_tables.append(aptable)
                    log.stdinfo(f"Aperture {i} found at {loc:.2f} "
                                f"({lower:.2f}, +{upper:.2f})")
                    if lower > 0 or upper < 0:
                        log.warning("Problem with automated sizing of "
                                    f"aperture {i}")

                aptable = vstack(all_tables, metadata_conflicts="silent")
                # Move "number" to be the first column
                new_order = ["number"] + [c for c in aptable.colnames if c != "number"]
                ext.APERTURE = aptable[new_order]

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def flagCosmicRays(self, adinputs=None, **params):
        """
        Detect and clean cosmic rays in a 2D wavelength-dispersed image,
        using the well-known LA Cosmic algorithm of van Dokkum (2001)*, as
        implemented in McCully's optimized version for Python, "astroscrappy"+.

        * LA Cosmic: http://www.astro.yale.edu/dokkum/lacosmic
        + astroscrappy: https://github.com/astropy/astroscrappy

        Parameters
        ----------
        suffix : str
            Suffix to be added to output files.

        x_order, y_order : int or None, optional
            Order for fitting and subtracting object continuum and sky line
            models, prior to running the main cosmic ray detection algorithm.
            When None, defaults are used, according to the image size (as in
            the IRAF task gemcrspec). When 0, no fit is done.

        bitmask : int, optional
            Bits in the input data quality `flags` that are to be used to
            exclude bad pixels from cosmic ray detection and cleaning. Default
            65535 (all non-zero bits, up to 16 planes).

        sigclip : float, optional
            Laplacian-to-noise limit for cosmic ray detection. Lower values
            will flag more pixels as cosmic rays. Default: 4.5.

        sigfrac : float, optional
            Fractional detection limit for neighboring pixels. For cosmic ray
            neighbor pixels, a lapacian-to-noise detection limit of
            sigfrac * sigclip will be used. Default: 0.3.

        objlim : float, optional
            Minimum contrast between Laplacian image and the fine structure
            image.  Increase this value if cores of bright stars are flagged
            as cosmic rays. Default: 5.0.

        pssl : float, optional
            Previously subtracted sky level in ADU. We always need to work in
            electrons for cosmic ray detection, so we need to know the sky
            level that has been subtracted so we can add it back in.
            Default: 0.0.

        niter : int, optional
            Number of iterations of the LA Cosmic algorithm to perform.
            Default: 4.

        sepmed : boolean, optional
            Use the separable median filter instead of the full median filter.
            The separable median is not identical to the full median filter,
            but they are approximately the same and the separable median filter
            is significantly faster and still detects cosmic rays well.
            Default: True

        cleantype : {'median', 'medmask', 'meanmask', 'idw'}, optional
            Set which clean algorithm is used:
            'median': An umasked 5x5 median filter
            'medmask': A masked 5x5 median filter
            'meanmask': A masked 5x5 mean filter
            'idw': A masked 5x5 inverse distance weighted interpolation
            Default: "meanmask".

        fsmode : {'median', 'convolve'}, optional
            Method to build the fine structure image:
            'median': Use the median filter in the standard LA Cosmic algorithm
            'convolve': Convolve the image with the psf kernel to calculate the
            fine structure image.
            Default: 'median'.

        psfmodel : {'gauss', 'gaussx', 'gaussy', 'moffat'}, optional
            Model to use to generate the psf kernel if fsmode == 'convolve' and
            psfk is None. The current choices are Gaussian and Moffat profiles.
            'gauss' and 'moffat' produce circular PSF kernels. The 'gaussx' and
            'gaussy' produce Gaussian kernels in the x and y directions
            respectively. Default: "gauss".

        psffwhm : float, optional
            Full Width Half Maximum of the PSF to use to generate the kernel.
            Default: 2.5.

        psfsize : int, optional
            Size of the kernel to calculate. Returned kernel will have size
            psfsize x psfsize. psfsize should be odd. Default: 7.

        psfbeta : float, optional
            Moffat beta parameter. Only used if fsmode=='convolve' and
            psfmodel=='moffat'. Default: 4.765.

        verbose : boolean, optional
            Print to the screen or not. Default: False.

        debug : bool
            Enable plots for debugging and store object and sky fits in the
            ad objects.

        """
        from astroscrappy import detect_cosmics

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        bitmask = params.pop('bitmask')
        debug = params.pop('debug')
        suffix = params.pop('suffix')
        x_order_in = params.pop('x_order')
        y_order_in = params.pop('y_order')

        fit_1D_params = dict(
            plot=debug,
            niter=params.pop('bkgfit_niter'),
            sigma_lower=params.pop('bkgfit_lsigma'),
            sigma_upper=params.pop('bkgfit_hsigma'),
        )

        for ad in adinputs:
            is_in_adu = ad[0].is_in_adu()
            if not is_in_adu:
                # astroscrappy takes data in adu
                for ext in ad:
                    ext.divide(ext.gain())

            # tile extensions by CCD to limit the number of edges
            array_info = gt.array_information(ad)
            ad_tiled = self.tileArrays([ad], tile_all=False)[0]

            for ext in ad_tiled:
                dispaxis = 2 - ext.dispersion_axis()

                # Use default orders from gemcrspec (from Bryan):
                ny, nx = ext.shape
                x_order = 9 if x_order_in is None else x_order_in
                y_order = ((2 if ny < 50 else 3 if ny < 80 else 5)
                           if y_order_in is None else y_order_in)

                if ext.mask is not None:
                    data = np.ma.array(ext.data, mask=ext.mask != 0)
                    mask = (ext.mask & bitmask) > 0
                    weights = (ext.mask == 0).astype(int)
                else:
                    data = ext.data
                    mask = None
                    weights = None

                # Fit the object spectrum:
                if x_order > 0:
                    objfit = fit_1D(data,
                                    function='legendre',
                                    axis=dispaxis,
                                    order=x_order,
                                    weights=weights,
                                    **fit_1D_params).evaluate()
                    if debug:
                        ext.OBJFIT = objfit.copy()
                else:
                    objfit = np.zeros(ext.shape)

                input_copy = data - objfit

                # Fit sky lines:
                if y_order > 0:
                    skyfit = fit_1D(input_copy, function='legendre',
                                    axis=1 - dispaxis,
                                    order=y_order,
                                    weights=weights,
                                    **fit_1D_params).evaluate()

                    # keep combined fits for later restoration
                    objfit += skyfit
                    if debug:
                        ext.SKYFIT = skyfit
                    skyfit = None

                input_copy = None

                # Run astroscrappy's detect_cosmics. We use the variance array
                # because it takes into account the different read noises if
                # the data has been tiled
                crmask, _ = detect_cosmics(ext.data,
                                           inmask=mask,
                                           inbkg=objfit,
                                           invar=ext.variance,
                                           gain=ext.gain(),
                                           satlevel=ext.saturation_level(),
                                           **params)

                # Set the cosmic_ray flags, and create the mask if needed
                if ext.mask is None:
                    ext.mask = np.where(crmask, DQ.cosmic_ray, DQ.good)
                else:
                    ext.mask[crmask] = DQ.cosmic_ray

                if debug:
                    plot_cosmics(ext, crmask)

            # Set flags in the original (un-tiled) ad
            if ad_tiled is not ad:
                xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
                for ext_tiled, indices in zip(ad_tiled, array_info.extensions):
                    tiled_arrsec = ext_tiled.array_section()
                    for i in indices:
                        ext = ad[i]
                        arrsec = ext.array_section()
                        slice_ = (slice((arrsec.y1 - tiled_arrsec.y1) // ybin,
                                        (arrsec.y2 - tiled_arrsec.y1) // ybin),
                                  slice((arrsec.x1 - tiled_arrsec.x1) // xbin,
                                        (arrsec.x2 - tiled_arrsec.x1) // xbin))

                        ext.mask = ext_tiled.mask[slice_]

                        if debug:
                            ext.OBJFIT = ext_tiled.OBJFIT[slice_]
                            ext.SKYFIT = ext_tiled.SKYFIT[slice_]

            # convert back to electron if needed
            if not is_in_adu:
                for ext in ad:
                    ext.multiply(ext.gain())

            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def fluxCalibrate(self, adinputs=None, **params):
        """
        Performs flux calibration multiplying the input signal by the
        sensitivity function obtained from
        :meth:`~geminidr.core.primitives_spect.Spec.calculateSensitivity`.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            1D or 2D Spectra of targets that need to be flux-calibrated.
            2D spectra are expected to be distortion corrected and its
            dispersion axis should be along rows.

        suffix :  str
            Suffix to be added to output files (default: _fluxCalibrated).

        standard: str or AstroData
            Standard star spectrum containing one extension or the same number
            of extensions as the input spectra. Each extension must have a
            `.SENSFUNC` table containing information about the overall
            sensitivity. Right now, if this is not provided, it will raise a
            NotImplementedError since it needs implementation.

        units : str, optional
            Units for output spectrum (default: W m-2 nm-1).

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has
            its pixel values in physical units.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        std = params["standard"]
        final_units = params["units"]
        do_cal = params["do_cal"]

        # Expectation is that the SENSFUNC table will be in units
        # like (electron/s) / (W/m^2)
        flux_units = u.Unit("W m-2")

        if do_cal == 'skip':
            log.warning("Flux calibration has been turned off.")
            return adinputs

        # Get a suitable specphot standard (with sensitivity function)
        if std is None:
            std_list = self.caldb.get_processed_standard(adinputs)
        else:
            std_list = (std, None)

        # Provide a standard AD object for every science frame, and an origin
        for ad, std, origin in zip(*gt.make_lists(adinputs, *std_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "fluxCalibrate. Continuing.")
                continue

            if std is None:
                if 'sq' in self.mode and do_cal != 'force':
                    raise OSError('No processed standard listed for {}'.
                                  format(ad.filename))
                else:
                    log.warning(f"{ad.filename}: no standard was specified. "
                                "Continuing.")
                    continue

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the standard {std.filename}"
                        f"{origin_str}")
            len_std, len_ad = len(std), len(ad)
            if len_std not in (1, len_ad):
                log.warning(f"{ad.filename} has {len_ad} extensions but "
                            f"{std.filename} has {len_std} extensions so "
                            "cannot flux calibrate.")
                continue

            if not all(hasattr(ext, "SENSFUNC") for ext in std):
                log.warning("SENSFUNC table missing from one or more extensions"
                            f" of {std.filename} so cannot flux calibrate")
                continue

            # Since 2D flux calibration just uses the wavelength info for the
            # middle row/column, non-distortion-corrected data will have the
            # wrong wavelength solution in other columns/rows
            if (any(len(ext.shape) == 2 for ext in ad) and
                    not self.timestamp_keys['distortionCorrect'] in ad.phu):
                log.warning(f"{ad.filename} has not been distortion corrected")

            telescope = ad.telescope()
            exptime = ad.exposure_time()
            try:
                std_site = std.phu["EXTCURVE"]
            except KeyError:
                try:
                    delta_airmass = ad.airmass() - std.airmass()
                except TypeError:  # if either airmass() returns None
                    log.warning("Cannot determine airmass of target "
                                f"{ad.filename} and/or standard {std.filename}"
                                ". Not performing airmass correction.")
                    delta_airmass = None
                else:
                    log.stdinfo(f"{ad.filename}: Correcting for difference of "
                                f"{delta_airmass:5.3f} airmasses")
            else:
                telescope = ad.telescope()
                sci_site = extinct.telescope_sites.get(telescope)
                if sci_site != std_site:
                    raise ValueError(f"Site of target observation {ad.filename}"
                                     f" ({sci_site}) does not match site used "
                                     f"to correct standard {std.filename} "
                                     f"({std_site}).")
                delta_airmass = ad.airmass()
                if delta_airmass is None:
                    log.warning(f"Cannot determine airmass of {ad.filename}."
                                " Not performing airmass correction.")
                else:
                    log.stdinfo(f"{ad.filename}: Correcting for airmass of "
                                f"{delta_airmass:5.3f}")


            for index, ext in enumerate(ad):
                ext_std = std[min(index, len_std-1)]
                extname = f"{ad.filename} extension {ext.id}"

                # Create the correct callable function (we may want to
                # abstract this in the future)
                sensfunc = am.table_to_model(ext_std.SENSFUNC)
                std_wave_unit = sensfunc.meta["xunit"]
                std_flux_unit = sensfunc.meta["yunit"]

                # Try to confirm the science image has the correct units
                std_physical_unit = (std_flux_unit.physical_unit if
                                     isinstance(std_flux_unit, u.LogUnit)
                                     else std_flux_unit)
                try:
                    sci_flux_unit = u.Unit(ext.hdr.get('BUNIT'))
                except:
                    sci_flux_unit = None
                if not (std_physical_unit is None or sci_flux_unit is None):
                    unit = sci_flux_unit * std_physical_unit / flux_units
                    if unit.is_equivalent(u.s):
                        log.fullinfo("Dividing {} by exposure time of {} s".
                                     format(extname, exptime))
                        ext /= exptime
                        sci_flux_unit /= u.s
                    elif not unit.is_equivalent(u.dimensionless_unscaled):
                        log.warning(f"{extname} has incompatible units ('"
                                    f"{sci_flux_unit}' and '{std_physical_unit}'"
                                    "). Cannot flux calibrate")
                        continue
                else:
                    log.warning("Cannot determine units of data and/or SENSFUNC "
                                f"table for {extname}, so cannot flux calibrate.")
                    continue

                # Get wavelengths of all pixels
                ndim = len(ext.shape)
                dispaxis = 0 if ndim == 1 else 2 - ext.dispersion_axis()

                # Get wavelengths and pixel sizes of all the pixels along the
                # dispersion axis by calculating wavelengths in the middles and
                # edges of all pixels.
                all_coords = [0.5*(length - 1) for length in ext.shape]
                all_coords[dispaxis] = np.arange(-0.5, ext.shape[dispaxis], 0.5)
                all_waves = ext.wcs(*all_coords[::-1], with_units=True)
                if ndim > 1:
                    all_waves = all_waves[0]

                waves = all_waves[1::2]
                pixel_sizes = abs(np.diff(all_waves[::2]))

                # Reconstruct the spline and evaluate it at every wavelength
                sens_factor = sensfunc(waves.to(std_wave_unit).value) * std_flux_unit
                try:  # conversion from magnitude/logarithmic units
                    sens_factor = sens_factor.physical
                except AttributeError:
                    pass

                # Apply airmass correction. If none is needed/possible, we
                # don't need to try to do this
                if delta_airmass:
                    try:
                        extinction_correction = extinct.extinction(
                            waves, telescope=telescope)
                    except KeyError:
                        log.warning(f"Telescope {telescope} not recognized. "
                                    "Not making an airmass correction.")
                    else:
                        sens_factor *= 10**(0.4 * delta_airmass * extinction_correction)

                final_sens_factor = (sci_flux_unit * sens_factor / pixel_sizes).to(
                    final_units, equivalencies=u.spectral_density(waves)).value

                if ndim == 2 and dispaxis == 0:
                    ext *= final_sens_factor[:, np.newaxis]
                else:
                    ext *= final_sens_factor
                ext.hdr['BUNIT'] = final_units

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def linearizeSpectra(self, adinputs=None, **params):
        """
        Transforms 1D spectra so that the relationship between the pixel
        location and wavelength is linear. This primitive calls
        resampleToCommonFrame to do the actual resampling.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 1D spectra.

        suffix : str
            Suffix to be added to output files.
        w1 : float
            Wavelength of first pixel (nm). See Notes below.
        w2 : float
            Wavelength of last pixel (nm). See Notes below.
        dw : float
            Dispersion (nm/pixel). See Notes below.
        npix : int
            Number of pixels in output spectrum. See Notes below.
        conserve : bool
            Conserve flux (rather than interpolate)?
        order : int
            order of interpolation during the resampling

        Notes
        -----
        Exactly 0 or 3 of (w1, w2, dw, npix) must be specified.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Linearized 1D spectra.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        w1 = params["w1"]
        w2 = params["w2"]
        dw = params["dw"]
        npix = params["npix"]
        conserve = params["conserve"]
        order = params["order"]

        # There are either 1 or 4 Nones, due to validation
        nones = [w1, w2, dw, npix].count(None)
        if nones == 1:
            # Work out the missing variable from the others
            if npix is None:
                npix = int(np.ceil((w2 - w1) / dw)) + 1
                w2 = w1 + (npix - 1) * dw
            elif w1 is None:
                w1 = w2 - (npix - 1) * dw
            elif w2 is None:
                w2 = w1 + (npix - 1) * dw
            else:
                dw = (w2 - w1) / (npix - 1)

        # We send the ADs through one-by-one so there's no attempt to
        # align them in the spatial direction
        adoutputs = []
        for ad in adinputs:
            ad_out = self.resampleToCommonFrame([ad], suffix=sfx, w1=w1, w2=w2, npix=npix,
                                                conserve=conserve, order=order,
                                                trim_data=False)[0]
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            adoutputs.append(ad_out)

        return adoutputs

    def normalizeFlat(self, adinputs=None, **params):
        """
        This primitive normalizes a spectroscopic flatfield, by fitting
        a cubic spline along the dispersion direction of an averaged
        combination of rows/columns (by default, in the center of the
        spatial direction). Each row/column is then divided by this spline.

        For multi-extension AstroData objects of MOS or XD, each extension
        is treated separately. For other multi-extension data,
        mosaicDetectors() is called to produce a single extension, and the
        spline fitting is performed with variable scaling parameters for
        each detector (identified within the mosaic from groups of DQ.no_data
        pixels). The spline fit is calculated in the mosaicked frame but it
        is evaluated for each pixel in each unmosaicked detector, so that
        the resultant flatfield always has the same format (i.e., number of
        extensions and their shape) as the input frame.

        Parameters
        ----------
        suffix : str/None
            suffix to be added to output files
        center : int/None
            central row/column for 1D extraction (None => use middle)
        nsum : int
            number of rows/columns around center to combine
        function : str
            type of function to fit (splineN or polynomial types)
        order : int
            Order of the spline fit to be performed
        lsigma : float/None
            lower rejection limit in standard deviations
        hsigma : float/None
            upper rejection limit in standard deviations
        niter : int
            maximum number of rejection iterations
        grow : float/False
            growth radius for rejected pixels
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        center = params["center"]
        nsum = params["nsum"]
        fit1d_params = fit_1D.translate_params(params)

        for ad in adinputs:
            # Don't mosaic if the multiple extensions are because the
            # data are MOS or cross-dispersed
            if len(ad) > 1 and not ({'MOS', 'XD'} & ad.tags):
                # Store original gWCS because we're modifying it
                orig_wcs = [ext.wcs for ext in ad]
                geotable = import_module('.geometry_conf', self.inst_lookups)
                transform.add_mosaic_wcs(ad, geotable)
                admos = transform.resample_from_wcs(ad, "mosaic", attributes=None,
                                                    order=3, process_objcat=False)
                mosaicked = True
            else:
                admos = ad
                mosaicked = False

            # This will loop over MOS slits or XD orders
            for ext in admos:
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"

                data, mask, variance, extract_slice = tracing.average_along_slit(ext, center=center, nsum=nsum)
                log.stdinfo("Extracting 1D spectrum from {}s {} to {}".
                            format(direction, extract_slice.start + 1, extract_slice.stop))
                mask |= (DQ.no_data * (variance == 0))  # Ignore var=0 points
                slices = _ezclump((mask & (DQ.no_data | DQ.unilluminated)) == 0)

                masked_data = np.ma.masked_array(data, mask=mask)
                weights = np.sqrt(np.where(variance > 0, 1. / variance, 0.))
                pixels = np.arange(len(masked_data))

                # We're only going to do CCD-to-CCD normalization if we've
                # done the mosaicking in this primitive; if not, we assume
                # the user has already taken care of it (if it's required).
                nslices = len(slices)
                if nslices > 1 and mosaicked:
                    coeffs = np.ones((nslices - 1,))
                    boundaries = list(slice_.stop for slice_ in slices[:-1])
                    result = optimize.minimize(QESpline, coeffs, args=(pixels, masked_data,
                                                                       weights, boundaries,
                                                                       fit1d_params["order"]),
                                               tol=1e-7, method='Nelder-Mead')
                    if not result.success:
                        log.warning(f"Problem with spline fitting: {result.message}")

                    # Rescale coefficients so centre-left CCD is unscaled
                    coeffs = np.insert(result.x, 0, [1])
                    coeffs /= coeffs[len(coeffs) // 2]
                    for coeff, slice_ in zip(coeffs, slices):
                        masked_data[slice_] *= coeff
                        weights[slice_] /= coeff
                    log.stdinfo("QE scaling factors: " +
                                " ".join("{:6.4f}".format(coeff) for coeff in coeffs))
                fit1d = fit_1D(masked_data, points=None, weights=weights,
                               **fit1d_params)

                if not mosaicked:
                    flat_data = np.tile(fit1d.evaluate(), (ext.shape[1-dispaxis], 1))
                    ext.divide(at.transpose_if_needed(flat_data, transpose=(dispaxis==0))[0])

            # If we've mosaicked, there's only one extension
            # We forward transform the input pixels, take the transformed
            # coordinate along the dispersion direction, and evaluate the
            # spline there.
            if mosaicked:
                origin = admos.nddata[0].meta.pop('transform')['origin']
                origin_shift = reduce(Model.__and__, [models.Shift(-s) for s in origin[::-1]])
                for ext, wcs in zip(ad, orig_wcs):
                    t = ext.wcs.get_transform(ext.wcs.input_frame, "mosaic") | origin_shift
                    geomap = transform.GeoMap(t, ext.shape, inverse=True)
                    flat_data = fit1d.evaluate(geomap.coords[dispaxis])
                    ext.divide(flat_data)
                    ext.wcs = wcs

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def resampleToCommonFrame(self, adinputs=None, **params):
        """
        Resample 1D or 2D spectra on a common frame, and optionally transform
        them so that the relationship between them and their respective
        wavelength calibration is linear.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 1D or 2D spectra.
        suffix : str
            Suffix to be added to output files.
        w1 : float
            Wavelength of first pixel (nm). See Notes below.
        w2 : float
            Wavelength of last pixel (nm). See Notes below.
        dw : float
            Dispersion (nm/pixel). See Notes below.
        npix : int
            Number of pixels in output spectrum. See Notes below.
        conserve : bool
            Conserve flux (rather than interpolate)?
        order : int
            order of interpolation during the resampling
        trim_data : bool
            Trim spectra to size of reference spectra?
        force_linear : bool
            Force a linear output wavelength solution?

        Notes
        -----
        If ``w1`` or ``w2`` are not specified, they are computed from the
        individual spectra: if ``trim_data`` is True, this is the intersection
        of the spectra ranges, otherwise this is the union of all ranges,

        If ``dw`` or ``npix`` are specified, the spectra are linearized.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Linearized 1D spectra.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        w1 = params["w1"]
        w2 = params["w2"]
        dw = params["dw"]
        npix = params["npix"]
        conserve = params["conserve"]
        trim_data = params["trim_data"]
        force_linear = params["force_linear"]

        # Check that all ad objects are either 1D or 2D
        ndim = {len(ext.shape) for ad in adinputs for ext in ad}
        if len(ndim) != 1:
            raise ValueError('inputs must have the same dimension')
        ndim = ndim.pop()

        # For the 2D case check that all ad objects have only 1 extension
        if ndim > 1:
            adjust_key = self.timestamp_keys['adjustWCSToReference']
            if len(adinputs) > 1 and not all(adjust_key in ad.phu
                                             for ad in adinputs):
                log.warning("2D spectral images should be processed by "
                            "adjustWCSToReference if accurate spatial "
                            "alignment is required.")
            if not all(len(ad) == 1 for ad in adinputs):
                raise ValueError('inputs must have only 1 extension')
            # Store these values for later!
            refad = adinputs[0]
            ref_coords = (refad.central_wavelength(asNanometers=True),
                          refad.target_ra(), refad.target_dec())
            ref_pixels = refad[0].wcs.backward_transform(*ref_coords)

        # If only one variable is missing we compute it from the others
        nparams = sum(x is not None for x in (w1, w2, dw, npix))
        if nparams == 3:
            if npix is None:
                npix = int(np.ceil((w2 - w1) / dw)) + 1
                w2 = w1 + (npix - 1) * dw
            elif w1 is None:
                w1 = w2 - (npix - 1) * dw
            elif w2 is None:
                w2 = w1 + (npix - 1) * dw
            else:
                dw = (w2 - w1) / (npix - 1)

        # Gather information from all the spectra (Chebyshev1D model,
        # w1, w2, dw, npix), and compute the final bounds (w1out, w2out)
        # if there are not provided
        info = []
        w1out, w2out, dwout, npixout = w1, w2, dw, npix
        for ad in adinputs:
            adinfo = []
            for ext in ad:
                try:
                    model_info = _extract_model_info(ext)
                except ValueError:
                    raise ValueError("Cannot determine wavelength solution "
                                     f"for {ad.filename} extension {ext.id}.")
                adinfo.append(model_info)

                if w1 is None:
                    if w1out is None:
                        w1out = model_info['w1']
                    elif trim_data:
                        w1out = max(w1out, model_info['w1'])
                    else:
                        w1out = min(w1out, model_info['w1'])

                if w2 is None:
                    if w2out is None:
                        w2out = model_info['w2']
                    elif trim_data:
                        w2out = min(w2out, model_info['w2'])
                    else:
                        w2out = max(w2out, model_info['w2'])
            info.append(adinfo)

        if trim_data:
            if w1 is None:
                w1out = info[0][0]['w1']
            if w2 is None:
                w2out = info[0][0]['w2']
            if w1 is None or w2 is None:
                log.fullinfo("Trimming data to size of reference spectra")

        # linearize spectra only if the grid parameters are specified
        linearize = force_linear or npix is not None or dw is not None
        if linearize:
            if npixout is None and dwout is None:
                # if both are missing, use the reference spectrum
                dwout = info[0][0]['dw']

            if npixout is None:
                npixout = int(np.ceil((w2out - w1out) / dwout)) + 1
            elif dwout is None:
                dwout = (w2out - w1out) / (npixout - 1)

        if linearize:
            new_wave_model = models.Scale(dwout) | models.Shift(w1out)
        else:
            # compute the inverse model needed to go to the reference
            # spectrum grid. Due to imperfections in the Chebyshev inverse
            # we check whether the wavelength limits are the same as the
            # reference spectrum.
            wave_model_ref = info[0][0]['wave_model'].copy()
            wave_model_ref.name = None
            limits = wave_model_ref.inverse([w1out, w2out])
            if info[0][0]['w1'] == w1out:
                limits[0] = round(limits[0])
            if info[0][0]['w2'] == w2out:
                limits[1] = round(limits[1])
            pixel_shift = int(np.ceil(limits.min()))
            new_wave_model = models.Shift(pixel_shift) | wave_model_ref
            if info[0][0]['w2'] == w2out:
                npixout = info[0][0]['npix']
            else:
               npixout = int(np.floor(new_wave_model.inverse([w1out, w2out]).max()) + 1)
            dwout = (w2out - w1out) / (npixout - 1)

        new_wave_model.name = 'WAVE'
        if ndim == 1:
            new_wcs_model = new_wave_model
        else:
            new_wcs_model = refad[0].wcs.forward_transform.replace_submodel('WAVE', new_wave_model)

        adoutputs = []
        for i, ad in enumerate(adinputs):
            flux_calibrated = self.timestamp_keys["fluxCalibrate"] in ad.phu

            for iext, ext in enumerate(ad):
                wave_model = info[i][iext]['wave_model']
                extn = f"{ad.filename} extension {ext.id}"
                wave_resample = wave_model | new_wave_model.inverse
                # TODO: This shouldn't really be needed, but it is
                wave_resample.inverse = new_wave_model | wave_model.inverse

                # Avoid performing a Cheb and its imperfect inverse
                if not linearize and new_wave_model[1:] == wave_model:
                    wave_resample = models.Shift(-pixel_shift)

                if ndim == 1:
                    dispaxis = 0
                    resampling_model = wave_resample
                else:
                    pixels = ext.wcs.backward_transform(*ref_coords)
                    dispaxis = 2 - ext.dispersion_axis()  # python sense
                    slit_offset = models.Shift(ref_pixels[dispaxis] - pixels[dispaxis])
                    if dispaxis == 0:
                        resampling_model = slit_offset & wave_resample
                    else:
                        resampling_model = wave_resample & slit_offset

                this_conserve = conserve_or_interpolate(ext, user_conserve=conserve,
                                        flux_calibrated=flux_calibrated, log=log)

                if i == 0 and not linearize:
                    log.fullinfo(f"{ad.filename}: No interpolation")
                msg = "Resampling"
                if linearize:
                    msg += " and linearizing"
                log.stdinfo("{} {}: w1={:.3f} w2={:.3f} dw={:.3f} npix={}"
                            .format(msg, extn, w1out, w2out, dwout, npixout))

                # If we resample to a coarser pixel scale, we may
                # interpolate over features. We avoid this by subsampling
                # back to the original pixel scale (approximately).
                input_dw = info[i][iext]['dw']
                subsample = int(np.ceil(abs(dwout / input_dw) - 0.1))
                attributes = [attr for attr in ('data', 'mask', 'variance')
                              if getattr(ext, attr) is not None]

                ext.wcs = gWCS([(ext.wcs.input_frame, resampling_model),
                                (cf.Frame2D(name='resampled'), new_wcs_model),
                                (ext.wcs.output_frame, None)])

                origin = (0,) * ndim
                output_shape = list(ext.shape)
                output_shape[dispaxis] = npixout
                new_ext = transform.resample_from_wcs(ext, 'resampled', subsample=subsample,
                                                      attributes=attributes, conserve=this_conserve,
                                                      origin=origin, output_shape=output_shape)
                if iext == 0:
                    ad_out = new_ext
                else:
                    ad_out.append(new_ext[0])
                #if ndim == 2:
                #    try:
                #        offset = slit_offset.offset.value
                #        ext.APERTURE['c0'] += offset
                #        log.fullinfo("Shifting aperture locations by {:.2f} "
                #                     "pixels".format(offset))
                #    except AttributeError:
                #        pass

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=suffix, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def skyCorrectFromSlit(self, adinputs=None, **params):
        """
        Performs row-by-row/column-by-column sky subtraction of 2D spectra.

        For that, it fits the sky contribution along each row/column
        perpendicular to the dispersion axis and builds a mask of rejected
        pixels during the fitting process. It also adds any apertures defined
        in the APERTURE table to this mask if it exists.

        This primitive should be called on data free of distortion.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D science spectra loaded as :class:`~astrodata.AstroData` objects.
        suffix : str or None
            Suffix to be added to output files.
        regions : str or None
            Sample region(s) to fit along rows/columns parallel to the slit,
            as a comma-separated list of pixel ranges. Any pixels outside these
            ranges (and/or included in the source aperture table) will be
            ignored when fitting each row or column.
        function : {'splineN', 'legendre', 'chebyshev', 'polynomial'}, optional
            Type of function/model to be used for fitting rows or columns
            perpendicular to the dispersion axis (default 'spline3', a cubic
            spline). For spline fits, N may be 1-5 (linear to quintic).
        order : int or None
            Order of fit to each row/column. For spline fits, this
            is the number of spline pieces; if `None`, as many pieces will be
            used as are required to get chi^2=1, otherwise the specified number
            will be reduced in proportion to the ratio of good pixels to total
            pixels in each row/column. If there are fewer than 4 good pixels in
            a given row/column, the fit will be performed using every pixel.
            For polynomial fitting functions, ``order`` is the polynomial degree
        lsigma, hsigma : float
            Lower and upper pixel rejection limits for fitting, in standard
            deviations from the fit
        niter : int
            Maximum number of fitting iterations
        grow : float or False, optional
            Masking growth radius (in pixels) for each statistically-rejected pixel
        aperture_growth : float
            Masking growth radius (in pixels) for each aperture
        debug_plot : bool
            Show diagnostic plots?
        interactive : bool
            Show interactive interface?

        Returns
        -------
        adinputs : list of :class:`~astrodata.AstroData`
            Sky subtractd 2D spectral images.

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.determineDistortion`,
        :meth:`~geminidr.core.primitives_spect.Spect.distortionCorrect`,
        :meth:`~geminidr.core.primitives_spect.Spect.findSourceApertures`,
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        apgrow = params["aperture_growth"]
        debug_plot = params["debug_plot"]
        fit1d_params = fit_1D.translate_params(params)
        interactive = params["interactive"]

        def calc_sky_coords(ad: AstroData, interactive_mode):
            """
            Calculate the sky coordinates for the extensions in the given
            AstroData object.

            This is useful for both feeding the data inputs calculation
            for the interactive interface and for the final loop over
            AstoData objects to do the fit (for both interactive and
            non-interactive).

            Parameters
            ----------
            ad : :class:`~astrodata.AstroData`
                AstroData to generate coordinates for
            interactive_mode : bool
                If True, collates aperture data mask separately to be used by UI

            Returns
            -------
            :class:`~astrodata.AstroData`, :class:`~numpy.ndarray`, :class:`~numpy.ndarray`
                extension, sky mask, sky weights yielded for each extension in the `ad`
            """
            for csc_ext in ad:
                csc_axis = csc_ext.dispersion_axis() - 1  # python sense

                # We want to mask pixels in apertures in addition to the mask.
                # Should we also leave DQ.cosmic_ray (because sky lines can get
                # flagged as CRs) and/or DQ.overlap unmasked here?
                csc_sky_mask = (np.zeros_like(csc_ext.data, dtype=DQ.datatype)
                                if csc_ext.mask is None else
                                csc_ext.mask & DQ.not_signal)

                # for interactive mode, we aggregate an aperture mask separately
                # for the UI
                csc_aperture_mask = (np.zeros_like(csc_ext.data, dtype=DQ.datatype))

                # If there's an aperture table, go through it row by row,
                # masking the pixels
                try:
                    aptable = csc_ext.APERTURE
                except AttributeError:
                    pass
                else:
                    for row in aptable:
                        trace_model = am.table_to_model(row)
                        aperture = tracing.Aperture(trace_model,
                                                    aper_lower=row['aper_lower'],
                                                    aper_upper=row['aper_upper'])
                        aperture_mask = aperture.aperture_mask(csc_ext, grow=apgrow)
                        if interactive_mode:
                            csc_aperture_mask |= aperture_mask
                        else:
                            csc_sky_mask |= aperture_mask

                if csc_ext.variance is None:
                    csc_sky_weights = None
                else:
                    csc_sky_weights = np.sqrt(at.divide0(1., csc_ext.variance))
                    # Handle columns were all the weights are zero
                    zeros = np.sum(csc_sky_weights, axis=csc_axis) == 0
                    if csc_axis == 0:
                        csc_sky_weights[:, zeros] = 1
                    else:
                        csc_sky_weights[zeros] = 1

                # Unmask rows/columns that are all DQ.no_data (e.g., GMOS
                # chip gaps) to avoid a zillion warnings about insufficient
                # unmasked points.
                no_data = np.bitwise_and.reduce(csc_sky_mask, axis=csc_axis) & DQ.no_data
                if csc_axis == 0:
                    csc_sky_mask ^= no_data
                else:
                    csc_sky_mask ^= no_data[:, None]

                if interactive_mode:
                    yield csc_ext, csc_sky_mask, csc_sky_weights, csc_aperture_mask
                else:
                    yield csc_ext, csc_sky_mask, csc_sky_weights

        def recalc_fn(ad: AstroData, conf: Config, extras: dict):
            """
            Used by the interactive code to generate all the inputs for the tabs
            per extension.

            This relies on the ``calc_sky_coords`` call to iterate on a set of
            extensions and their calculated sky_mask and sky_weights.  It then
            creates the sky masked array and pixel coordinates to return for
            the interactive code.  This function is suitable for use as the
            data source for the fit1d interactive code.

            Parameters
            ----------
            ad : :class:`~astrodata.core.AstroData`
                AstroData instance to work on
            conf : :class:`~config.Config`
                unused, but required in the function signature for interactive
            extras : dict
                Dictionary of additional values, here the ``col`` is passed as the selected column

            Returns
            -------
            :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, :class:`~numpy.ndarray`
                Yields a list of tupes with the pixel, sky, sky_weights

            See Also
            --------
            :meth:`~geminidr.core.primitives_spect.Spect.skyCorrectFromSlit.calc_sky_coords`
            """
            # pylint: disable=unused-argument
            c = max(0, extras['col'] - 1)
            # TODO alternatively, save these 3 arrays for faster recalc
            # here I am rerunning all the above calculations whenever a col select is made
            for rc_ext, rc_sky_mask, rc_sky_weights, rc_aper_mask in calc_sky_coords(ad, interactive_mode=True):
                rc_sky = np.ma.masked_array(rc_ext.data[:, c], mask=rc_sky_mask[:, c])
                yield np.arange(len(rc_sky)), rc_sky, rc_sky_weights[:, c], rc_aper_mask[:, c] \
                    if rc_sky_weights is not None else None

        final_parms = list()
        if interactive:
            # build config for interactive
            config = self.params[self.myself()]
            config.update(**params)

            # Create a 'col' parameter to add to the UI so the user can select the column they
            # want to fit.
            # We pass a default column at the 1/3 mark, since dead center is flat
            ncols = adinputs[0].shape[0][0]
            reinit_params = ["col", ]
            reinit_extras = {"col": RangeField(doc="Column of data", dtype=int, default=int(ncols / 3),
                                               min=1, max=ncols)}

            # Build the set of input shapes and count the total extensions while we are at it
            for ad in adinputs:
                all_shapes = []
                count = 0
                for ext in ad:
                    count = count+1
                    all_shapes.append((0, ext.shape[0]))  # extracting single line for interactive

                # Get filename to display in visualizer
                filename_info = getattr(ad, 'filename', '')

                # get the fit parameters
                fit1d_params = fit_1D.translate_params(params)
                visualizer = fit1d.Fit1DVisualizer(lambda conf, extras: recalc_fn(ad, conf, extras),
                                                   fitting_parameters=[fit1d_params]*count,
                                                   config=config,
                                                   reinit_params=reinit_params,
                                                   reinit_extras=reinit_extras,
                                                   tab_name_fmt="Slit {}",
                                                   xlabel='Column',
                                                   ylabel='Signal',
                                                   domains=all_shapes,
                                                   title="Sky Correct From Slit",
                                                   primitive_name="skyCorrectFromSlit",
                                                   filename_info=filename_info,
                                                   help_text=SKY_CORRECT_FROM_SLIT_HELP_TEXT,
                                                   plot_ratios=False,
                                                   enable_user_masking=False)
                geminidr.interactive.server.interactive_fitter(visualizer)

                # Pull out the final parameters to use as inputs doing the real fit
                fit_results = visualizer.results()
                final_parms_exts = list()
                for fit in fit_results:
                    final_parms_exts.append(fit.extract_params())
                final_parms.append(final_parms_exts)
        else:
            # making fit params into an array even though it all matches
            # so we can share the same final code with the interactive,
            # where a user may have tweaked per extension inputs
            for ad in adinputs:
                final_parms.append([fit1d_params] * len(ad))

        for idx, ad in enumerate(adinputs):  # idx for indexing the fit1d params per ext
            if self.timestamp_keys['distortionCorrect'] not in ad.phu:
                log.warning(f"{ad.filename} has not been distortion corrected."
                            " Sky subtraction is likely to be poor.")
            eidx = 0
            for ext, sky_mask, sky_weights in calc_sky_coords(ad):
                axis = ext.dispersion_axis() - 1  # python sense
                sky = np.ma.masked_array(ext.data, mask=sky_mask)
                sky_model = fit_1D(sky, weights=sky_weights, **final_parms[idx][eidx],
                                   axis=axis, plot=debug_plot).evaluate()
                ext.data -= sky_model
                eidx = eidx + 1

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def traceApertures(self, adinputs=None, **params):
        """
        Traces apertures listed in the `.APERTURE` table along the dispersion
        direction, and estimates the optimal extraction aperture size from the
        spatial profile of each source.

        This primitive is now designed to run on tiled and mosaicked data so
        normal long-slit spectra will be in a single extension. We keep the loop
        over extensions to allow the possibility of expanding it to cases where
        we have multiple extensions (e.g. Multi-Object Spectroscopy).

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with a `.APERTURE` table attached
            to one or more of its extensions.
        debug: bool, optional
            draw aperture traces on image display window? Default: False
        interactive: bool, optional
            Run primitive interactively? Default: False
        max_missed : int, optional
            Maximum number of interactions without finding line before line is
            considered lost forever. Default: 5
        max_shift : float, optional
            Maximum perpendicular shift (in pixels) from pixel to pixel.
            Default: 0.05
        nsum : int, optional
            Number of rows/columns to combine at each step. Default: 10
        order : int, optional
            Fitting order along spectrum. Default: 2
        step : int, optional
            Step size for sampling along dispersion direction. Default: 10
        suffix : str, optional
            Suffix to be added to output files. Default: "_aperturesTraced".

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with the `.APERTURE` the updated
            to contain its upper and lower limits.

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.findSourceApertures`

        """

        # Setup log
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Parse parameters
        debug = params["debug"]
        interactive = params["interactive"]
        max_missed = params["max_missed"]
        max_shift = params["max_shift"]
        nsum = params["nsum"]
        sfx = params["suffix"]
        step = params["step"]

        fit1d_params = fit_1D.translate_params(
            {**params, "function": "chebyshev"})

        # order is pulled out for the non-interactive version
        order = None
        if not interactive:
            # pop "order" seeing we may need to call fit_1D with a
            #  different value for the non-interactive version
            order = fit1d_params.pop("order")

        # Main Loop
        for ad in adinputs:
            for ext in ad:

                # Verify inputs
                try:
                    aptable = ext.APERTURE
                    locations = aptable['c0'].data
                except (AttributeError, KeyError):
                    log.warning("Could not find aperture locations in "
                                f"{ad.filename} extension {ext.id} - continuing")
                    continue

                if debug:
                    self.viewer.display_image(ext, wcs=False)
                    self.viewer.width = 2
                    self.viewer.color = "blue"

                if interactive:
                    # Pass the primitive configuration to the interactive object.
                    _config = self.params[self.myself()]
                    _config.update(**params)
                    aperture_models = interactive_trace_apertures(
                        ext, _config, fit1d_params)
                else:
                    dispaxis = 2 - ext.dispersion_axis()  # python sense
                    aperture_models = []

                    # For efficiency, we would like to trace all sources
                    #  simultaneously (like we do with arc lines), but we need
                    #  to start somewhere the source is bright enough, and there
                    #  may not be a single location where that is true for all
                    #  sources
                    for i, loc in enumerate(locations):
                        c0 = int(loc + 0.5)
                        spectrum = ext.data[c0] if dispaxis == 1 else ext.data[:, c0]
                        start = np.argmax(at.boxcar(spectrum, size=20))

                        # The coordinates are always returned as (x-coords, y-coords)
                        ref_coords, in_coords = tracing.trace_lines(ext, axis=dispaxis,
                                                                    start=start, initial=[loc],
                                                                    rwidth=None, cwidth=5, step=step,
                                                                    nsum=nsum, max_missed=max_missed,
                                                                    initial_tolerance=None,
                                                                    max_shift=max_shift,
                                                                    viewer=self.viewer if debug else None)
                        if i:
                            all_ref_coords = np.concatenate((all_ref_coords, ref_coords), axis=1)
                            all_in_coords = np.concatenate((all_in_coords, in_coords), axis=1)
                        else:
                            all_ref_coords = ref_coords
                            all_in_coords = in_coords

                    spectral_coords = np.arange(0, ext.shape[dispaxis], step)

                    for aperture in aptable:
                        location = aperture['c0']
                        # Funky stuff to extract the traced coords associated with
                        # each aperture (there's just a big list of all the coords
                        # from all the apertures) and sort them by coordinate
                        # along the spectrum
                        coords = np.array([list(c1) + list(c2)
                                           for c1, c2 in zip(all_ref_coords.T, all_in_coords.T)
                                           if c1[dispaxis] == location])
                        values = np.array(sorted(coords, key=lambda c: c[1 - dispaxis])).T
                        ref_coords, in_coords = values[:2], values[2:]

                        # log aperture
                        min_value = in_coords[1 - dispaxis].min()
                        max_value = in_coords[1 - dispaxis].max()
                        log.debug(f"Aperture at {c0:.1f} traced from {min_value} "
                                  f"to {max_value}")

                        # Find model to transform actual (x,y) locations to the
                        # value of the reference pixel along the dispersion axis
                        try:
                            # pylint: disable=repeated-keyword
                            _fit_1d = fit_1D(
                                in_coords[dispaxis],
                                domain=[0, ext.shape[dispaxis] - 1],
                                order=order,
                                points=in_coords[1 - dispaxis],
                                **fit1d_params)


                        # This hides a multitude of sins, including no points
                        # returned by the trace, or insufficient points to
                        # constrain fit. We call fit1d with dummy points to
                        # ensure we get the same type of result as if it had
                        # been successful.
                        except (IndexError, np.linalg.linalg.LinAlgError):
                            log.warning(
                                f"Unable to trace aperture {aperture['number']}")

                            # pylint: disable=repeated-keyword
                            _fit_1d = fit_1D(
                                np.full_like(spectral_coords, c0),
                                domain=[0, ext.shape[dispaxis] - 1],
                                order=0,
                                points=spectral_coords,
                                **fit1d_params)

                        else:
                            if debug:
                                plot_coords = np.array(
                                    [spectral_coords,
                                     _fit_1d.evaluate(spectral_coords)]).T
                                self.viewer.polygon(plot_coords, closed=False,
                                                    xfirst=(dispaxis == 1), origin=0)

                        aperture_models.append(_fit_1d.model)

                all_aperture_tables = []
                for model, aperture in zip(aperture_models, aptable):
                    this_aptable = am.model_to_table(model)

                    # Recalculate aperture limits after rectification
                    #apcoords = _fit_1d.evaluate(np.arange(ext.shape[dispaxis]))
                    this_aptable["number"] = aperture["number"]
                    this_aptable["aper_lower"] = \
                        aperture["aper_lower"] #+ (location - apcoords.min())
                    this_aptable["aper_upper"] = \
                        aperture["aper_upper"] # - (apcoords.max() - location)
                    all_aperture_tables.append(this_aptable)

                # If the traces have different orders, there will be missing
                # values that vstack will mask, so we have to set those to zero
                new_aptable = vstack(all_aperture_tables,
                                     metadata_conflicts="silent")
                colnames = new_aptable.colnames
                new_col_order = (["number"] + sorted(c for c in colnames
                                                     if c.startswith("c")) +
                                 ["aper_lower", "aper_upper"])
                for col in colnames:
                    if isinstance(new_aptable[col], MaskedColumn):
                        new_aptable[col] = new_aptable[col].filled(fill_value=0)
                ext.APERTURE = new_aptable[new_col_order]

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

    def write1DSpectra(self, adinputs=None, **params):
        """
        Write 1D spectra to files listing the wavelength and data (and
        optionally variance and mask) in one of a range of possible formats.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images.
        format : str
            format for writing output files
        header : bool
            write FITS header before data values?
        extension : str
            extension to be used in output filenames
        apertures : str
            comma-separated list of aperture numbers to write
        dq : bool
            write DQ (mask) plane?
        var : bool
            write VAR (variance) plane?
        overwrite : bool
            overwrite existing files?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The unmodified input files.
        """
        # dict of {format parameter: (Table format, file suffix)}
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        fmt = params["format"]
        header = params["header"]
        extension = params["extension"]
        apertures = params["apertures"]
        if apertures:
            these_apertures = [int(x) for x in str(apertures).split(",")]
        write_dq = params["dq"]
        write_var = params["var"]
        overwrite = params["overwrite"]

        for ad in adinputs:
            aperture_map = dict(zip(range(len(ad)), ad.hdr.get("APERTURE")))
            if apertures is None:
                these_apertures = sorted(list(aperture_map.values()))
            for aperture in these_apertures:
                indices = [k for k, v in aperture_map.items() if v == aperture]
                if len(indices) > 2:
                    log.warning(f"{ad.filename} has more than one aperture "
                                f"numbered {aperture} - continuing")
                    continue
                elif not indices:
                    log.warning(f"{ad.filename} does not have an aperture "
                                f"numbered {aperture} - continuing")
                    continue

                ext = ad[indices.pop()]
                if ext.data.ndim != 1:
                    log.warning(f"{ad.filename} aperture {aperture} is not a "
                                "1D array - continuing")
                    continue

                data_unit = u.Unit(ext.hdr.get("BUNIT"))
                t = Table((ext.wcs(range(ext.data.size)), ext.data),
                          names=("wavelength", "data"),
                          units=(ext.wcs.output_frame.unit[0], str(data_unit)))
                if write_dq:
                    t.add_column(ext.mask, name="dq")
                if write_var:
                    t.add_column(ext.variance, name="variance")
                    t["variance"].unit = str(data_unit ** 2)
                    var_col = len(t.colnames)

                filename = (os.path.splitext(ad.filename)[0] +
                            f"_{aperture:03d}.{extension}")
                log.stdinfo(f"Writing {filename}")
                try:
                    if header:
                        with open(filename, "w" if overwrite else "x") as f:
                            for line in (repr(ext.phu) + repr(ext.hdr)).split("\n"):
                                if line != " " * len(line):
                                    f.write(f"# {line.strip()}\n")
                            t.write(f, format=fmt)
                    elif fmt == "fits":
                        # Table.write isn't happy with the unit 'electron'
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=AstropyUserWarning)
                            thdu = fits.table_to_hdu(t)
                        if "TUNIT2" not in thdu.header:
                            thdu.header["TUNIT2"] = str(data_unit)
                        if write_var and f"TUNIT{var_col}" not in thdu.header:
                            thdu.header[f"TUNIT{var_col}"] = str(data_unit ** 2)
                        hlist = fits.HDUList([fits.PrimaryHDU(), thdu])
                        hlist.writeto(filename, overwrite=overwrite)
                    else:
                        t.write(filename, format=fmt, overwrite=overwrite)
                except OSError:
                    log.warning(f"{filename} already exists - cannot write")

        return adinputs

    def _get_arc_linelist(self, waves=None):
        """
        Returns a list of wavelengths of the arc reference lines used by the
        primitive `determineWavelengthSolution()`, if the user parameter
        `linelist=None` (i.e., the default list is requested).

        Parameters
        ----------
        waves : array / None
            (approx) wavelengths of each pixel in nm. This is required in case
            the linelist depends on resolution or wavelength setting.

        Returns
        -------
        array_like
            arc line wavelengths

        array_like or None
            arc line weights
        """
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        filename = os.path.join(lookup_dir, 'linelist.dat')
        return wavecal.LineList(filename)

    def _get_spectrophotometry(self, filename):
        """
        Reads a file containing spectrophotometric data for a standard star
        and returns these data as a Table(), with unit information. We
        attempt to read a range of files and interpret them, either using
        metadata or guesswork. If there's no metadata, we assume that the
        first column is the wavelength, the second is the brightness data,
        there may then be additional columns with uncertainty information,
        and the width of the bandpass is always the last column.

        We ignore any uncertainty information because, for ground-based data,
        this will be swamped by limitations of the user's data.

        Parameters
        ----------
        filename: str
            name of file containing spectrophotometric data

        Returns
        -------
        Table:
            the spectrophotometric data, with columns 'WAVELENGTH',
            'WIDTH', and 'FLUX'

        Raises
        ------
        FileNotFoundError: if file does not exist
        InconsistentTableError: if the file can't be read as ASCII
        """
        log = self.log
        try:
            tbl = Table.read(filename)
        except IORegistryError:
            # Force ASCII
            tbl = Table.read(filename, format='ascii')

        # Create table, interpreting column names (or lack thereof)
        spec_table = Table()
        colnames = ('WAVELENGTH', 'WIDTH', 'MAGNITUDE')
        aliases = (('WAVE', 'LAMBDA', 'col1'),
                   ('FWHM', 'col3'),
                   ('MAG', 'ABMAG', 'FLUX', 'FLAM', 'FNU', 'col2', 'DATA'))

        for colname, alias in zip(colnames, aliases):
            for name in (colname,) + alias:
                if name in tbl.colnames:
                    spec_table[colname] = tbl[name]
                    orig_colname = name
                    break
            else:
                log.warning("Cannot find a column to convert to '{}' in "
                            "{}".format(colname.lower(), filename))

        # Now handle units
        for col in spec_table.itercols():
            try:
                unit = col.unit
            except AttributeError:
                unit = None
            if isinstance(unit, u.UnrecognizedUnit):
                # Try chopping off the trailing 's'
                try:
                    unit = u.Unit(re.sub(r's$', '', col.unit.name.lower()))
                except:
                    unit = None
            if unit is None:
                # No unit defined, make a guess
                if col.name == 'WAVELENGTH':
                    unit = u.AA if max(col.data) > 5000 else u.nm
                elif col.name == 'WIDTH':
                    unit = spec_table['WAVELENGTH'].unit
                else:
                    if orig_colname == 'FNU':
                        unit = u.Unit("erg cm-2 s-1 Hz-1")
                    elif orig_colname in ('FLAM', 'FLUX') or np.median(col.data) < 1:
                        unit = u.Unit("erg cm-2 s-1 AA-1")
                    else:
                        unit = u.mag
            col.unit = unit

            # We've created a column called "MAGNITUDE" but it might be a flux
            if col.name == 'MAGNITUDE':
                try:
                    unit.to(u.W / u.m ** 3, equivalencies=u.spectral_density(1. * u.m))
                except:
                    pass
                else:
                    col.name = 'FLUX'

        # If we don't have a flux column, create one
        if not 'FLUX' in spec_table.colnames:
            # Use ".data" here to avoid "mag" being in the unit
            spec_table['FLUX'] = (10 ** (-0.4 * (spec_table['MAGNITUDE'].data + 48.6))
                                  * u.Unit("erg cm-2 s-1") / u.Hz)
        return spec_table


# -----------------------------------------------------------------------------
def _extract_model_info(ext):
    if len(ext.shape) == 1:
        dispaxis = 0
        wave_model = ext.wcs.forward_transform
    else:
        dispaxis = 2 - ext.dispersion_axis()
        wave_model = am.get_named_submodel(ext.wcs.forward_transform, 'WAVE')
    npix = ext.shape[dispaxis]
    limits = wave_model([0, npix])
    w1, w2 = min(limits), max(limits)
    dw = (w2 - w1) / (npix - 1)
    return {'wave_model': wave_model, 'w1': w1, 'w2': w2,
            'npix': npix, 'dw': dw}


def conserve_or_interpolate(ext, user_conserve=None, flux_calibrated=False,
                            log=None):
    """
    This helper function decides whether the data should undergo flux
    conservation (or data interpolation) based on its units and whether it
    has been flux calibrated, and compares this to what the user has asked
    for. It logs any concerns and returns what it considers to be the best
    decision.

    Parameters
    ----------
    ext : AstroData slice
        extension of interest
    user_conserve : bool/None
        user parameter for conservation of flux
    flux_calibrated : bool
        has this AstroData object gone through the fluxCalibrate primitive?
    log : logger

    Returns
    -------
    bool : whether or not to conserve the flux
    """
    ext_str = f"{ext.filename} extension {ext.id}"
    ext_unit = ext.hdr["BUNIT"]
    if ext_unit in (None, ""):
        if user_conserve is None:
            this_conserve = not flux_calibrated
            log.stdinfo(f"{ext_str} has no units but "
                        f"{'has' if flux_calibrated else 'has not'} been flux"
                        f" calibrated so setting conserve={this_conserve}")
        else:
            this_conserve = user_conserve
            if this_conserve == flux_calibrated:
                log.warning(f"{ext_str} {'has' if flux_calibrated else 'has not'}"
                            f"been flux calibrated but conserve={user_conserve}")
        return this_conserve

    ext_unit = u.Unit(ext_unit)
    # Test for units like flux density
    units_imply_conserve = True
    for unit1 in ("W", "photon", "electron", "adu"):
        for unit2 in ("m2", ""):
            try:
                ext_unit.to(u.Unit(f"{unit1} / ({unit2} nm)"),
                            equivalencies=u.spectral_density(1. * u.m))
            except u.UnitConversionError:
                pass
            else:
                units_imply_conserve = False
                break

    if flux_calibrated and units_imply_conserve:
        log.warning(f"Possible unit mismatch for {ext_str}. File has been "
                    f"flux calibrated but units are {ext_unit}")
    if user_conserve is None:
        this_conserve = units_imply_conserve
        log.stdinfo(f"Setting conserve={this_conserve} for {ext_str} since "
                    f"units are {ext_unit}")
    else:
        if user_conserve != units_imply_conserve:
            log.warning(f"conserve is set to {user_conserve} but the "
                        f"units of {ext_str} are {ext_unit}")
        this_conserve = user_conserve  # but do what we're told
    return this_conserve


def QESpline(coeffs, xpix, data, weights, boundaries, order):
    """
    Fits a cubic spline to data, allowing scaling renormalizations of
    contiguous subsets of the data.

    Parameters
    ----------
    coeffs : array_like
        Scaling factors for CCDs 2+.

    xpix : array
        Pixel numbers (in general, 0..N).

    data : masked_array
        Data to be fit.

    weights: array
        Fitting weights (inverse standard deviations).

    boundaries: tuple
        The last pixel coordinate on each CCD.

    order: int
        Order of spline to fit.

    Returns
    -------
    float
        Normalized chi^2 of the spline fit.
    """
    scaling = np.ones_like(data, dtype=np.float64)
    for coeff, boundary in zip(coeffs, boundaries):
        scaling[boundary:] = coeff
    scaled_data = scaling * data
    scaled_weights = 1. / scaling if weights is None else (weights / scaling).astype(np.float64)
    spline = am.UnivariateSplineWithOutlierRemoval(xpix, scaled_data,
                                                            order=order, w=scaled_weights, niter=1, grow=0)
    result = np.ma.masked_where(spline.mask, np.square((spline.data - scaled_data) *
                                                       scaled_weights)).sum() / (~spline.mask).sum()
    return result


def plot_arc_fit(data, peaks, arc_lines, arc_weights, model, title):
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    weights = np.full_like(arc_lines, 3) if arc_weights is None else arc_weights
    for line, wt in zip(arc_lines, weights):
        ax.plot([line, line], [0, 1], color='{}'.format(0.07 * (9 - wt)))
    for peak in model(peaks):
        ax.plot([peak, peak], [0, 1], 'r:')
    ax.plot(model(np.arange(len(data))), 0.98 * data / np.max(data), 'b-')
    limits = model([0, len(data)])
    ax.set_xlim(min(limits), max(limits))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative intensity")
    ax.set_title(title)


def plot_cosmics(ext, crmask):
    from astropy.visualization import ZScaleInterval, imshow_norm

    fig, axes = plt.subplots(5, 1, figsize=(15, 5*2), sharex=True, sharey=True)
    imgs = (ext.data, ext.OBJFIT, ext.SKYFIT,
            ext.data - (ext.OBJFIT + ext.SKYFIT), crmask)
    titles = ('data', 'object fit', 'sky fit', 'residual', 'crmask')
    mask = ext.mask & (DQ.max ^ DQ.cosmic_ray)

    for ax, data, title in zip(axes, imgs, titles):
        if title != 'crmask':
            cmap = 'Greys_r'
            interval = ZScaleInterval()
            data = np.ma.array(data, mask=mask)
        else:
            cmap = 'Greys'
            interval = None

        cmap = copy(plt.get_cmap(cmap))
        cmap.set_bad('r')
        imshow_norm(data, ax=ax, origin='lower', interval=interval, cmap=cmap)
        ax.set_title(title)

    plt.show()
