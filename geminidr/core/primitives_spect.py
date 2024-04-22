# Copyright(c) 2016-2020 Association of Universities for Research in Astronomy, Inc.
#
#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
import tempfile
from itertools import islice
import os
import re
import warnings
from copy import copy, deepcopy
from functools import partial, reduce
from importlib import import_module

import matplotlib
import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io.ascii.core import InconsistentTableError
from astropy.io.registry import IORegistryError
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import Model, fitting, models
from astropy.stats import sigma_clip
from astropy.table import Table, hstack, vstack, MaskedColumn
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from matplotlib import pyplot as plt
from numpy.ma.extras import _ezclump
from scipy import optimize
from scipy.signal import find_peaks
from specutils import SpectralRegion
from specutils.utils.wcs_utils import air_to_vac, vac_to_air

import astrodata
from gemini_instruments import gmu
from gemini_instruments.gemini import get_specphot_name
import geminidr.interactive.server
from astrodata import AstroData
from astrodata.provenance import add_provenance
from geminidr.core.primitives_resample import Resample
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gemini.lookups import extinction_data as extinct
from geminidr.gemini.lookups import qa_constraints, atran_models, oh_synthetic_spectra
from geminidr.interactive.fit import fit1d
from geminidr.interactive.fit.aperture import interactive_find_source_apertures
from geminidr.interactive.fit.tracing import interactive_trace_apertures
from geminidr.interactive.fit.wavecal import WavelengthSolutionVisualizer
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from gempy.library import astrotools as at
from gempy.library import tracing, transform, wavecal
from gempy.library.astrotools import array_from_list, transpose_if_needed
from gempy.library.config import RangeField
from gempy.library.fitting import fit_1D
from gempy.library.matching import KDTreeFitter, match_sources, fit_model
from gempy.library.spectral import Spek1D
from gempy.library.tracing import pinpoint_peaks
from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.utils.md5 import md5sum

from . import parameters_spect
from ..interactive.fit.help import CALCULATE_SENSITIVITY_HELP_TEXT, SKY_CORRECT_FROM_SLIT_HELP_TEXT, \
    NORMALIZE_FLAT_HELP_TEXT
from ..interactive.interactive import UIParameters

matplotlib.rcParams.update({'figure.max_open_warning': 0})

# ------------------------------------------------------------------------------


# noinspection SpellCheckingInspection
@parameter_override
@capture_provenance
class Spect(Resample):
    """
    This is the class containing all of the pre-processing primitives
    for the `Spect` level of the type hierarchy tree.
    """
    tagset = {"GEMINI", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_spect)
        self.generated_linelist = False

    def adjustWavelengthZeroPoint(self, adinputs=None, **params):
        """
        Find sky lines and match them to a linelist in order to shift the
        wavelength scale zero point slightly to account for flexure in the
        telescope.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 2D spectra.
        suffix : str
            Suffix to be added to output files
        center : None or int
            Central row/column for 1D extraction (None => use middle).
        shift : float, Default : None
            An optional shift to apply directly to the wavelength scale, in
            pixels. If not given, the shift will be calculated from the sky
            lines present in the image.
        verbose : bool, Default : False
            Print additional information on the fitting process.
        """

        def _add_shift_model_to_wcs(shift, dispaxis, ext):
            """ Create a model to shift the wavelength scale by and add to WCS

            This function creates a compound model of two Shift models in
            parallel, with the one in the `dispaxis` direction getting the value
            of `shift`. This allows the resulting compound model to applied to
            a WCS without any special handling for different spectral axis
            orientations.

            Parameters
            ----------
            shift : float
                The shift to apply, in pixels (will be converted to wavelength)
            dispaxis : int, 0 or 1
                The dispersion axis, in the Python sense
            ext : Astrodata extension
                The extension to apply the shift to.
            """
            dx, dy = 0, 0
            if dispaxis == 0:
                dy = shift
            elif dispaxis == 1:
                dx = shift
            else:
                raise ValueError("'dispaxis' must be 0 (vertical) or "
                                 "1 (horizontal)")

            # This should work for both orientations without having to code
            # them separately.
            model = models.Shift(dx) & models.Shift(dy)
            model.name = 'FLEXCORR'
            ext.wcs.insert_frame(ext.wcs.input_frame, model,
                                 cf.Frame2D(name="wavelength_scale_adjusted"))

        # Set up log
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        center = params["center"]
        shift = params["shift"]
        max_shift = params["debug_max_shift"]
        verbose = params["verbose"]

        # Check given shift, if there is one.
        if shift and abs(shift) > max_shift:
            raise ValueError("Provided shift is larger than parameter "
                             f"'debug_max_shift': |{shift:.3g}| > "
                             f"{max_shift:.3g}")

        for ad in adinputs:
            log.stdinfo(f"{ad.filename}:")

            for ext in ad:
                dispaxis = 2 - ext.dispersion_axis()  # Python sense
                row_or_col = 'row' if dispaxis == 1 else 'column'

                # If the user specifies a shift value, apply it and continue
                # (case of shift == 0 caught and handled above)
                if shift is not None:
                    if shift == 0:
                        msg = "    No wavelength shift from sky lines will be"\
                              " performed since shift=0"
                    else:
                        msg = "    Shifted wavelength scale for extension "\
                              f"{ext.id} by {shift:0.4g} pixels"
                    _add_shift_model_to_wcs(shift, dispaxis, ext)
                    log.stdinfo(msg)
                    continue

                # Otherwise, we'll need to automatically find the shift.
                # Values (generally) taken from the defaults for
                # determineWavelengthSolution, with some changes.
                config_dict = {
                        "order": 3,
                        "niter": 3,
                        "lsigma": 3.0,
                        "hsigma": 3.0,
                        "central_wavelength": None,
                        "interactive": False,
                        "center": center,
                        "nsum": 10,
                        "fwidth": None,
                        "min_snr": 10,
                        "min_sep": 2,
                        "weighting": "local",
                        "nbright": 0,
                        "dispersion": None,
                        "combine_method": "mean",
                        "absorption": False,
                        "debug_min_lines": 15,
                        "debug_alternative_centers": False,
                    }

                wave_scale = ext.wcs.output_frame.axes_names[0]
                if wave_scale == 'WAVE':
                    config_dict["in_vacuo"] = True
                    log.fullinfo("Calibrated in vacuum.")
                elif wave_scale == 'AWAV':
                    log.fullinfo("Calibrated in air.")
                    config_dict["in_vacuo"] = False
                else:
                    raise ValueError("Cannot interpret wavelength scale "
                                     f"for {ext.filename}:{ext.id} "
                                     f"(found '{wave_scale}'")

                # get_all_input_data() outputs several lines of information
                # which can be useful but confusing if many files are processed,
                # so use the verbose parameter to allow users to control it.
                # loglevel is the level at which the output should be logged,
                # so higher levels (e.g. stdinfo) print more to the console.
                loglevel = "stdinfo" if verbose else "fullinfo"
                try:
                    input_data = wavecal.get_all_input_data(
                        ext, self, config_dict, linelist=None,
                        bad_bits=DQ.not_signal, skylines=True,
                        loglevel=loglevel)
                except ValueError:
                    raise ValueError("Something went wrong in finding sky "
                                     "lines - check that the spectrum is being "
                                     f"taken in a {row_or_col} free of the "
                                     "object aperture, and change it with the "
                                     "`center` parameter if necessary")

                spectrum = input_data["spectrum"]
                init_models = input_data["init_models"]
                domain = init_models[0].meta["domain"]
                peaks, weights = input_data["peaks"], input_data["weights"]
                sky_lines = input_data["linelist"].wavelengths(
                    in_vacuo=config_dict["in_vacuo"], units='nm')
                sky_weights = input_data["linelist"].weights
                if sky_weights is None:
                    sky_weights = np.ones_like(sky_lines)
                    log.debug("No weights were found for the reference linelist")

                m_init = init_models[0]
                # Fix all parameters in the model so that they don't change
                # (only the Shift which will be added next).
                for p in m_init.param_names:
                    getattr(m_init, p).fixed = True

                # Add a bounded Shift model in front of the wavelength solution
                m_init = models.Shift(0, bounds={'offset': (-max_shift,
                                                            max_shift)}) | m_init
                m_init.meta["domain"] = domain

                fwidth = input_data["fwidth"]
                dw = np.diff(m_init(np.arange(spectrum.size))).mean()
                kdsigma = fwidth * abs(dw)
                k = 1 if kdsigma < 3 else 2

                fit_it = KDTreeFitter(sigma=2 * abs(dw), maxsig=5,
                                      k=k, method='Nelder-Mead')
                m_final = fit_it(m_init, peaks, sky_lines,
                                 in_weights=weights[config_dict["weighting"]],
                                 ref_weights=sky_weights, matches=None,
                                 options={'disp': True if verbose else False})

                # Apply the shift to the wavelength scale
                shift_final = m_final.offset_0.value
                log.stdinfo(f"    Shifted wavelength scale for "
                            f"extension {ext.id} by {shift_final:0.4g} "
                            f"pixels ({shift_final * dw:0.4g} nm)")
                _add_shift_model_to_wcs(shift_final, dispaxis, ext)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

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
        region: str / None
            pixel region for determining slit profile for cross-correlation
        tolerance : float
            Maximum distance from the header offset, for the correlation
            method (arcsec). If the correlation computed offset is too
            different from the header offset, then the latter is used.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        methods = (params["method"], params["fallback"])
        region = slice(*at.parse_user_regions(params["region"])[0])
        tolerance = params["tolerance"]
        integer_offsets = params["debug_block_resampling"]

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
            ref_profile = tracing.stack_slit(refad[0], section=region)
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
                    profile = tracing.stack_slit(ad[0], section=region)
                    corr = np.correlate(ref_profile, profile, mode='full')
                    expected_peak = corr.size // 2 + hdr_offset
                    peaks, snrs = tracing.find_wavelet_peaks(corr, widths=np.arange(3, 20),
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
                    if integer_offsets:
                        offset = np.round(offset)
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

    def attachWavelengthSolution(self, adinputs=None, **params):
        """
        Attach the distortion map (a Chebyshev2D model) and the mapping from
        distortion-corrected pixels to wavelengths (a Chebyshev1D model, when
        available after successful line matching) from a processed arc, or
        similar wavelength reference, to the WCS of the input data.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images.
        suffix : str
            Suffix to be added to output files.
        arc : :class:`~astrodata.AstroData` or str or None
            Arc(s) containing distortion map & wavelength calibration.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Modified input objects with the WCS updated for each extension.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        arc = params["arc"]

        # Get a suitable arc frame (with distortion map) for every science AD
        if arc is None:
            arc_list = self.caldb.get_processed_arc(adinputs)
        else:
            arc_list = (arc, None)

        fail = False

        adoutputs = []
        # Provide an arc AD object for every science frame, and an origin
        for ad, arc, origin in zip(*gt.make_lists(adinputs, *arc_list,
                                                  force_ad=(1,))):
            # We don't check for a timestamp since it's not unreasonable
            # to do multiple distortion corrections on a single AD object

            len_ad = len(ad)
            if arc is None:
                log.warning(f"{ad.filename}: no arc was specified. "
                            "Continuing.")
                if 'sq' in self.mode:
                    fail = True
                adoutputs.append(ad)
                continue

                # By analogy with distortionCorrect, we should probably still
                # attach the mosaic transform here if there's no arc and the
                # data are not already mosaicked, but it would be better to
                # separate that out into a different step anyway (and call it
                # here if necessary), so leave that for further refactoring
                # and just keep the original WCS for now (which currently won't
                # prevent distortionCorrect from mosaicking the data).

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the arc {arc.filename}"
                        f"{origin_str}")
            len_arc = len(arc)
            if len_arc not in (1, len_ad):
                log.warning(f"{ad.filename} has {len_ad} extensions but "
                            f"{arc.filename} had {len_arc} extensions so "
                            "cannot calibrate the distortion.")
                if 'sq' in self.mode:
                    fail = True
                adoutputs.append(ad)
                continue

            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            if arc.detector_x_bin() != xbin or arc.detector_y_bin() != ybin:
                log.warning(f"Science frame {ad.filename} and arc "
                            f"{arc.filename} have different binnings.")
                if 'sq' in self.mode:
                    fail = True
                adoutputs.append(ad)
                continue

            ad_detsec = ad.detector_section()

            # Check that the arc is at least as large as the science frame
            # We only do this for single-extension arcs now, which is true
            # for GMOSLongslit
            if len_arc == 1:
                arc_detsec = arc.detector_section()[0]
                detsec_array = np.asarray(ad_detsec)
                x1, _, y1, _ = detsec_array.min(axis=0)
                x2, _, y2, _ = detsec_array.max(axis=0)
                if (x1 < arc_detsec.x1 or x2 > arc_detsec.x2 or
                        y1 < arc_detsec.y1 or y2 > arc_detsec.y2):
                    log.warning(f"Science frame {ad.filename} is larger than "
                                f"the arc {arc.filename}")
                    fail = True
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
                    if 'distortion_corrected' not in wcs.available_frames:
                        distortion_models = []
                        break
                except AttributeError:
                    distortion_models = []
                    break

                m_distcorr = wcs.get_transform(wcs.input_frame,
                                               'distortion_corrected')
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
                if 'sq' in self.mode:
                    fail = True
                continue

            # Determine whether we're producing a single-extension AD
            # or keeping the number of extensions as-is
            if len_arc == 1:

                distortion_models *= len_ad
                wave_models *= len_ad
                wave_frames *= len_ad

                # Save spatial WCS from input ext we're transforming WRT:
                ref_idx = transform.find_reference_extension(ad)
                sky_models = [
                    am.get_named_submodel(ad[ref_idx].wcs.forward_transform,
                                          'SKY')
                ] * len_ad
                output_frames = [ad[ref_idx].wcs.output_frame.frames] * len_ad

                # For GMOS with one arc and lots of inputs. The output of either
                # branch of this code should be a gWCS that ends in
                # ("distortion_corrected", None) so that the final bit of code can
                # insert a "world" frame using a munged-together sky model and
                # wavelength model.
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
                                                     (cf.Frame2D(name='distortion_corrected'), None)])
                                break
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
                        # Shift the distortion-corrected co-ordinates back from
                        # the arc's ROI to the native one after transforming:
                        for ext in ad:
                            ext.wcs.insert_transform(
                                'distortion_corrected',
                                reduce(Model.__and__,
                                       [models.Shift(-offset) for
                                        offset in offsets]),
                                after=False
                            )
                        # len(arc)=1 so we only have one wave_model, but need to
                        # update the entry in the list, which gets used later
                        if wave_model is not None:
                            offset = offsets[ext.dispersion_axis()-1]
                            if offset != 0:
                                wave_model.name = None
                                wave_model = models.Shift(offset) | wave_model
                                wave_model.name = 'WAVE'
                                wave_models = [wave_model] * len_ad

                # Single-extension AD, with single Transform (non-GMOS code)
                else:
                    ad_detsec = ad.detector_section()[0]
                    if ad_detsec != arc_detsec:
                        if self.timestamp_keys['mosaicDetectors'] in ad.phu:
                            # Shouldn't this be allowed with a full-frame arc?
                            log.warning("Cannot calibrate distortions in "
                                        "mosaicked data unless arc has the "
                                        "same ROI. Continuing.")
                            if 'sq' in self.mode:
                                fail = True
                            adoutputs.append(ad)
                            continue
                        # No mosaicking, so we can just do a shift
                        m_shift = (models.Shift((ad_detsec.x1 - arc_detsec.x1) / xbin) &
                                   models.Shift((ad_detsec.y1 - arc_detsec.y1) / ybin))
                        m_distcorr = m_shift | m_distcorr
                    # Create a new pipeline for the gWCS here. We can't use
                    # insert_frame() because we need to chop off the "world"
                    # frame at the end (and split a frame from its transform).
                    # This should work whether or not one or more frames
                    # (e.g., "rectified") have been added after the input_frame.
                    new_pipeline = ad[0].wcs.pipeline[:-2] +\
                                   [(ad[0].wcs.pipeline[-2].frame, m_distcorr),
                                   (cf.Frame2D(name='distortion_corrected'), None)]
                    ad[0].wcs = gWCS(new_pipeline)

                if wave_model is None:
                    log.warning(f"{arc.filename} has no wavelength solution")
                    if 'sq' in self.mode:
                        fail = True

            else:
                log.warning("Distortion calibration with multiple-extension "
                            "arcs has not been tested.")

                sky_models, output_frames = [], []

                for i, (ext, ext_arc, dist_model) in enumerate(zip(ad, arc, distortion_models)):
                    # Save spatial WCS from input ext we're transforming WRT:
                    sky_models.append(
                        am.get_named_submodel(ext.wcs.forward_transform, 'SKY')
                    )
                    output_frames.append(ext.wcs.output_frame.frames)

                    # Shift science so its pixel coords match the arc's before
                    # applying the distortion correction
                    shifts = [c1 - c2 for c1, c2 in zip(ext.detector_section(),
                                                        ext_arc.detector_section())]
                    dist_model = (models.Shift(shifts[0] / xbin) &
                                  models.Shift(shifts[1] / ybin)) | dist_model
                    # This hasn't been tested, but should work in analogy with
                    # the code above. We can't use insert_frame() here either.
                    new_pipeline = ad[0].wcs.pipeline[:-2] +\
                                   [(ad[0].wcs.pipeline[-2].frame, m_distcorr),
                                   (cf.Frame2D(name='distortion_corrected'), None)]
                    ext.wcs = gWCS(new_pipeline)

                    if wave_model is None:
                        log.warning(f"{arc.filename} extension {ext.id} has "
                                    "no wavelength solution")
                        if 'sq' in self.mode:
                            fail = True

            for i, (ext, wave_model, wave_frame, sky_model, output_frame) in \
              enumerate(zip(ad, wave_models, wave_frames, sky_models,
                            output_frames)):
                t = wave_model & sky_model
                if ext.dispersion_axis() == 2:
                    t = models.Mapping((1, 0)) | t
                # We need to create a new output frame with a copy of the
                # ARC's SpectralFrame in case there's a change from air->vac
                # or vice versa
                new_output_frame = cf.CompositeFrame(
                    [copy(wave_frame) if isinstance(frame, cf.SpectralFrame)
                     else frame for frame in output_frame], name='world'
                )
                ext.wcs.insert_frame('distortion_corrected', t,
                                     new_output_frame)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad)
            if arc.path:
                add_provenance(ad, arc.filename, md5sum(arc.path) or "", self.myself())

        if fail:
            raise OSError("No suitable arc calibration for one or more "
                          "input(s)")

        return adoutputs

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

        resampling: float/None
            if not None, resample the specphot file to this wavelength
            interval (in nm) before calculating the sensitivity

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
        in_vacuo = params["in_vacuo"]
        bandpass = params["bandpass"]
        resample_interval = params["resampling"]
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
                specphot_name = get_specphot_name(ad)
                if specphot_name is None:
                    specphot_name = ad.object().lower().replace(' ', '')
                filename = f'{specphot_name}.dat'
                for module in (self.inst_lookups, gemini_lookups, 'geminidr.core.lookups'):
                    try:
                        path = import_module('.', module).__path__[0]
                    except (ImportError, ModuleNotFoundError):
                        continue
                    full_path = os.path.join(path, 'spectrophotometric_standards', filename)
                    try:
                        spec_table = self._get_spectrophotometry(
                            full_path, in_vacuo=in_vacuo)
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
                    spec_table = self._get_spectrophotometry(
                        datafile, in_vacuo=in_vacuo)
                except FileNotFoundError:
                    log.warning(f"Cannot find spectrophotometric data table {datafile}."
                                f" Unable to determine sensitivity for {ad.filename}")
                    continue
                except InconsistentTableError:
                    log.warning(f"Cannot read spectrophotometric data table {datafile}."
                                f" Unable to determine sensitivity for {ad.filename}")
                    continue

            exptime = ad.exposure_time()
            if 'WIDTH' not in spec_table.colnames:
                log.warning(f"Using default bandpass of {bandpass} nm")
                spec_table['WIDTH'] = bandpass * u.nm

            if resample_interval:
                spec_table = resample_spec_table(spec_table, resample_interval)

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

                extid = f"{ext.filename}:{ext.id}"
                if "AWAV" in ext.wcs.output_frame.axes_names:
                    wavecol_name = "WAVELENGTH_AIR"
                    log.debug(f"{extid} is calibrated to air wavelengths")
                elif "WAVE" in ext.wcs.output_frame.axes_names:
                    wavecol_name = "WAVELENGTH_VACUUM"
                    log.debug(f"{extid} is calibrated to vacuum wavelengths")
                else:
                    raise ValueError("Cannot interpret wavelength scale "
                                     f"for {extid}")

                # Compute values that are counts / (exptime * flux_density * bandpass)
                for w0, dw, fluxdens in zip(spec_table[wavecol_name].quantity,
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
                all_waves = [x[1].value for x in all_exts]
                all_zpt = [x[2].value for x in all_exts]
                all_weights = [x[3].value for x in all_exts]
                all_fp_init = []
                for i in range(len(all_exts)):
                    all_fp_init.append(fit_1D.translate_params(params))

                # build config for interactive
                config = self.params[self.myself()]
                config.update(**params)

                # Get filename to display in visualizer
                filename_info = getattr(ad, 'filename', '')

                uiparams = UIParameters(config)
                visualizer = fit1d.Fit1DVisualizer({"x": all_waves, "y": all_zpt, "weights": all_weights},
                                                   fitting_parameters=all_fp_init,
                                                   tab_name_fmt=lambda i: f"CCD {i+1}",
                                                   xlabel=f'Wavelength ({xunits})',
                                                   ylabel=f'Sensitivity ({yunits})',
                                                   domains=all_domains,
                                                   title="Calculate Sensitivity",
                                                   primitive_name="calculateSensitivity",
                                                   filename_info=filename_info,
                                                   help_text=CALCULATE_SENSITIVITY_HELP_TEXT,
                                                   ui_params=uiparams)
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

    def createNewAperture(self, adinputs=None, **params):
        """
        Create a new aperture, as an offset from another (given) aperture.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            A list of spectra with an APERTURE table.
        aperture : int
            Aperture number upon which to base new aperture.
        shift : float
            Shift (in pixels) to new aperture.
        aper_lower : float
            Distance in pixels from center to lower edge of new aperture.
        aper_upper : float
            Distance in pixels from center to upper edge of new aperture.
        suffix : str
            Suffix to be added to output files.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has a new
            aperture in its APERTURE table, created as an offset from an
            existing aperture.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        aperture = params["aperture"]
        shift = params["shift"]
        aper_lower = params["aper_lower"]
        aper_upper = params["aper_upper"]
        sfx = params['suffix']

        # First check that the given reference aperture is available in each
        # extension of all AstroData objects, no-op if not. Report all cases
        # where the reference aperture is missing.
        ok = True
        for ad in adinputs:
            for ext in ad:
                if aperture not in list(ext.APERTURE['number']):
                    log.warning(f"Aperture number {aperture} not found in "
                                f"extension {ext.id}.")
                    ok = False
        if not ok:
            log.warning(f"No new apertures will be created by {self.myself()}")
            return adinputs

        for ad in adinputs:
            for ext in ad:
                spataxis = ext.dispersion_axis() - 1  # Python sense
                too_low, too_high = (("left", "right") if spataxis == 1
                                     else ("bottom", "top"))

                # We know this exists from the check above.
                existing_apnums = list(ext.APERTURE['number'])
                apnum = existing_apnums.index(aperture)

                # Copy the appropriate row.
                new_row = deepcopy(ext.APERTURE[apnum])
                new_row['c0'] += shift

                apmodel = am.table_to_model(new_row)
                # Expect domain to be equal to the number of spectral pixels
                try:
                    center_pixels = apmodel(np.arange(*apmodel.domain))
                except TypeError:  # something wrong (e.g., domain is "None")
                    center_pixels = apmodel(np.arange(ext.shape[1-spataxis]))
                _min, _max = min(center_pixels), max(center_pixels)

                # Set user-given values for upper and lower aperture edges.
                # Validation should ensure they either both exist or are None.
                if aper_lower is not None and aper_upper is not None:
                    new_row['aper_lower'] = aper_lower
                    new_row['aper_upper'] = aper_upper
                aplo, aphi = new_row['aper_lower', 'aper_upper']

                new_apnum = min(set(range(1, max(existing_apnums) + 2)) -
                                set(existing_apnums))
                log.stdinfo(f"Adding new aperture {apnum} to {ad.filename} "
                            f"extension {ext.id}.")
                new_row['number'] = new_apnum
                ext.APERTURE.add_row(new_row)

                # Print warning if new aperture is off the array
                if _max + aphi < 0:
                    log.warning(f"New aperture is entirely off {too_low} of image.")
                elif _min + aplo < 0:
                    log.warning(f"New aperture is partially off {too_low} of image.")
                if _min + aplo > ext.data.shape[spataxis]:
                    log.warning(f"New aperture is entirely off {too_high} of image.")
                elif _max + aphi > ext.data.shape[spataxis]:
                    log.warning(f"New aperture is partially off {too_high} of image.")

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

        min_line_length: float
            Minimum length of traced feature (as a fraction of the tracing dimension
            length) to be considered as a useful line.

        debug_reject_bad: bool
            Reject lines with suspiciously high SNR (e.g. bad columns)? (Default: True)

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
        min_line_length = params["min_line_length"]
        debug_reject_bad = params["debug_reject_bad"]
        debug = params["debug"]

        orders = (max(spectral_order, 1), spatial_order)

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
                    data, _, _, _ = tracing.average_along_slit(ext, center=start, nsum=nsum)
                    fwidth = tracing.estimate_peak_width(data, boxcar_size=30)
                    log.stdinfo(f"Estimated feature width: {fwidth:.2f} pixels")

                if initial_peaks is None:
                    data, mask, variance, extract_slice = tracing.average_along_slit(ext, center=start, nsum=nsum)
                    log.stdinfo("Finding peaks by extracting {}s {} to {}".
                                format(direction, extract_slice.start + 1, extract_slice.stop))

                    # Find peaks; convert width FWHM to sigma
                    widths = 0.42466 * fwidth * np.arange(0.75, 1.26, 0.05)  # TODO!
                    initial_peaks, _ = tracing.find_wavelet_peaks(
                        data, widths=widths, mask=mask & DQ.not_signal,
                        variance=variance, min_snr=min_snr, reject_bad=debug_reject_bad)
                    log.stdinfo(f"Found {len(initial_peaks)} peaks")

                # The coordinates are always returned as (x-coords, y-coords)
                rwidth = 0.42466 * fwidth
                #.data[:,0:1500]
                ref_coords, in_coords = tracing.trace_lines(ext, axis=1 - dispaxis,
                                                            start=start, initial=initial_peaks,
                                                            rwidth=rwidth, cwidth=max(int(fwidth), 5), step=step,
                                                            nsum=nsum, max_missed=max_missed,
                                                            max_shift=max_shift * ybin / xbin,
                                                            viewer=self.viewer if debug else None,
                                                            min_line_length=min_line_length)

                # The model is computed entirely in the pixel coordinate frame
                # of the data, so it could be used as a gWCS object
                m_init = models.Chebyshev2D(x_degree=orders[1 - dispaxis],
                                            y_degree=orders[dispaxis],
                                            x_domain=[0, ext.shape[1]-1],
                                            y_domain=[0, ext.shape[0]-1])

                fixed_linear = (spectral_order == 0)
                model, m_final, m_inverse = create_distortion_model(
                    m_init, 1-dispaxis, in_coords, ref_coords, fixed_linear)

                # TODO: Some logging about quality of fit
                # print(np.min(diff), np.max(diff), np.std(diff))

                self.viewer.color = "red"
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
                                    (cf.Frame2D(name="distortion_corrected"),
                                     ext.wcs.pipeline[0][1])]
                                   + ext.wcs.pipeline[1:])

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def determineSlitEdges(self, adinputs=None, **params):
        """
        Finds the edges of the illuminated regions of the CCD and stores the
        Chebyshev polynomials used to fit them in a SLITEDGE table.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images.
        suffix : str
            Suffix to be added to output files.
        spectral_order : int, Default : 3
            Fitting order in the spectral direction (minimum of 1).
        debug : bool, Default: False
            Generate plots of several aspects of the fitting process.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with a `SLITEDGE` table attached
            to each extension.
        """
        # Set up log
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        sfx = params["suffix"]

        # Parse parameters
        debug = params['debug']
        spectral_order = params['spectral_order']
        edges1 = params['edges1']
        edges2 = params['edges2']
        if edges1 is not None and not isinstance(edges1, list):
            edges1 = [edges1]
        if edges2 is not None and not isinstance(edges2, list):
            edges2 = [edges2]

        # How far from the edge of the detector an edge must be to be traced.
        # I.e., peaks within this many pixels of the detector edge likely won't
        # be able to be traced.
        buffer = 8

        fit1d_params = fit_1D.translate_params({"function": "chebyshev",
                                                "order": spectral_order})
        for ad in adinputs:

            try:
                getattr(ad, "MDF")
            except AttributeError:
                log.warning(f"MDF not found for {ad.filename} - no "
                            "SLITEDGE table will be created.")
                continue

            log.stdinfo('Finding edges of illuminated region in flat for '
                        f'{ad.filename}')

            # Get the expected slit center and length for long slit.
            if (edges1 is not None) and (edges2 is not None):
                log.stdinfo('Using user-supplied edges.')
                exp_edges_1, exp_edges_2 = edges1, edges2
            elif (edges1 is None) and (edges2 is None):
                exp_edges_1, exp_edges_2 = self._get_slit_edge_estimates(ad)
                log.debug('Estimating fit edges...')
            else:
                log.warning("Both `edges1` and `edges2` parameters need to be "
                            "provided in order to use them. Using "
                            "edges from mask definition files.")
                exp_edges_1, exp_edges_2 = self._get_slit_edge_estimates(ad)

            slit_widths = [b - a for a, b in zip(exp_edges_1, exp_edges_2)]
            mdf_edge_guesses = [(a, b) for a, b in zip(exp_edges_1,
                                                       exp_edges_2)]

            edge1, edge2 = ("left", "right") if ad.dispersion_axis().pop() == 2\
                            else ("bottom", "top")
            log.fullinfo('Expected edge positions:\n'
                         f'{edge1.capitalize()} edges: {exp_edges_1}\n'
                         f'{edge2.capitalize()} edges: {exp_edges_2}\n')

            # This is the number of rows/columns to sum around the row with
            # the maxium flux to create the profile for finding edges, to
            # help eliminate cosmic rays. The row used for finding edges
            # will also be at least this far from the ends of the detector.
            offset = 20

            # How far to search (in pixels) to match expected and detected
            # peaks.
            search_rad = 60

            for ext in ad:

                dispaxis = 2 - ext.dispersion_axis()
                log.fullinfo(f'Dispersion axis is axis {dispaxis}.')

                # Find the row/column with the highest median flux, at least
                # `offset` pixels away from the edge of the detector.
                collapsed = np.median(
                    ext.data,
                    axis=1-dispaxis)
                cut = collapsed[offset:-offset].argmax()
                cut += offset
                row_or_col = 'row' if dispaxis == 0 else 'column'
                log.fullinfo(f'Creating profile around {row_or_col} {cut}.')

                # Take the first derivative of flux to find the slit edges.
                # Left edges will be peaks, right edges troughs, so make a
                # second negative copy to find right edges separately.
                diffarr_1 = np.diff(ext.data, axis=1-dispaxis)
                diffarr_2 = -diffarr_1

                # Take median of a small slice to smooth over cosmic rays:
                s = slice(cut-offset, cut+offset)
                if dispaxis == 0:
                    median_slice_1 = np.median(diffarr_1[s, :],
                                                axis=[dispaxis])
                    median_slice_2 = np.median(diffarr_2[s, :],
                                                axis=[dispaxis])
                elif dispaxis == 1:
                    median_slice_1 = np.median(diffarr_1[:, s],
                                                axis=[dispaxis])
                    median_slice_2 = np.median(diffarr_2[:, s],
                                                axis=[dispaxis])

                # Search for position of peaks in the first derivative of flux
                # perpendicular to the dispersion direction. Apply a 3.5-sigma
                # limit to avoid picking up too many noise peaks, but
                # match_sources later on should pick out the real ones.
                std = np.std(median_slice_1)
                if exp_edges_1[0] < 0:
                    # If the expected edge position if off the end of the
                    # detector, don't try to match as it might pick up
                    # spurious peaks close to the edge of the detector.
                    positions_1 = []
                else:
                    positions_1, _ = find_peaks(median_slice_1,
                                                height=3.5*std,
                                                distance=5)
                    # find_peaks returns integer values, so use pinpoint_peaks
                    # to better describe the positions.
                    positions_1, _ = tracing.pinpoint_peaks(median_slice_1,
                                                            peaks=positions_1)
                if exp_edges_2[0] > ext.shape[1-dispaxis]:
                    positions_2 = []
                else:
                    positions_2, _ = find_peaks(median_slice_2,
                                                height=3.5*std,
                                                distance=5)
                    positions_2, _ = tracing.pinpoint_peaks(median_slice_2,
                                                            peaks=positions_2)

                log.fullinfo('Found edge candidates at:\n'
                             f'{edge1.capitalize()}: {positions_1}\n'
                             f'{edge2.capitalize()}: {positions_2}')
                if debug:
                    # Print a diagnostic plot of the profile being fitted.
                    plt.plot(median_slice_1, label='1st-derivative of flux')
                    plt.plot(median_slice_2, label='Inverse')
                    plt.xlabel(f'{row_or_col.capitalize()} number')
                    plt.legend()

                # Check if both edges have been located; if the list returned
                # by find_peaks is empty, fill in the position by assuming
                # the other edge is at the length of the slit away. (This may
                # be off the edge of the detector.)
                if (len(positions_1) == 0) and (len(positions_2) == 0):
                    log.warning("No edges could be found for "
                                f"{ad.orig_filename} - no SLITEDGE table "
                                "will be attached.")
                    if debug:
                        plt.show()
                    continue

                # Lists to hold positions of edges found.
                edges_1, edges_2 = [], []

                # Handlle cases where one edge of the illuminated region is off
                # the side of the detector by constructing an "edge" based on
                # the expected breadth of the region. If both edges are on,
                # run match_sources on both to find the closest match to each.
                if len(slit_widths) == 1:
                    if (len(positions_1) == 0) and (len(positions_2) != 0):
                        edge_ids_2 = match_sources(positions_2,
                                                   exp_edges_2,
                                                   radius=search_rad)
                        for i, j in enumerate(edge_ids_2):
                            if j > -1:  # a matched position
                                edges_2.append(positions_2[i])
                                edges_1.append(
                                    positions_2[i] - slit_widths[0])

                    elif (len(positions_2) == 0) and (len(positions_1) != 0):
                        edge_ids_1 = match_sources(positions_1,
                                                   exp_edges_1,
                                                   radius=search_rad)
                        for i, j in enumerate(edge_ids_1):
                            if j > -1:  # a matched position
                                edges_1.append(positions_1[i])
                                edges_2.append(
                                    positions_1[i] + slit_widths[0])

                    elif (len(positions_2) != 0) and (len(positions_1) != 0):
                        edge_ids_2 = match_sources(positions_2,
                                                    exp_edges_2,
                                                    radius=search_rad)
                        edge_ids_1 = match_sources(positions_1,
                                                    exp_edges_1,
                                                    radius=search_rad)
                        edges_1 = [positions_1[i] for i, j in
                                    enumerate(edge_ids_1) if j > -1]
                        edges_2 = [positions_2[i] for i, j in
                                    enumerate(edge_ids_2) if j > -1]

                        if len(edges_1) == 0 and len(edges_2) != 0:
                            edges_1 = [r - slit_widths[0] for r in edges_2]
                        elif len(edges_2) == 0 and len(edges_1) != 0:
                            edges_2 = [l + slit_widths[0] for l in edges_1]

                # Generate pairs of slit edges.
                edge_pairs = [(a, b) for a, b in zip(edges_1, edges_2)]

                if not edge_pairs:
                    log.warning("No edges could be determined for "
                                f"{ad.filename} - no SLITEDGE table will be "
                                "attached. Consider setting positions of "
                                "edges manually using the `edges1` and "
                                "`edges2` parameters.")
                    if debug:
                        plt.show()
                    continue

                if debug:
                    for pair in edge_pairs:
                        for pos in pair:
                            plt.axvline(pos, color='Black', alpha=0.5,
                                        linestyle='--')
                    plt.legend()
                    plt.show()

                # Create the model to shift and scale the slit to the found
                # edges.
                if len(edge_pairs) == 1:
                    pair_center = (edge_pairs[0][0] + edge_pairs[0][1]) / 2
                    m_recenter = models.Shift(-pair_center,
                                              fixed={'offset': True})
                    m_shift = models.Shift(0, bounds={'offset': (-search_rad,
                                                                 search_rad)})
                    m_scale = models.Scale(1., bounds={'factor': (0.98, 1.02)})
                    m_init = m_recenter | m_shift | m_scale | m_recenter.inverse

                # Count the number of pairs of edges found.
                pair_num = 0
                models_dict = {}

                # Progressively increase the sigma parameter of the search,
                # since it being too narrow sometimes causes the fitting to
                # fail.
                for pair, guess in zip(edge_pairs, mdf_edge_guesses):
                    models_dict[pair_num] = {edge1: None, edge2: None}
                    log.fullinfo(f'Fitting guess at {guess} to {pair}.')

                    n_sigma = 1
                    max_sigma = 10
                    tolerance = 1.0
                    while True:
                        log.fullinfo(f'Trying fitting with sigma={n_sigma}')
                        if n_sigma > max_sigma:
                            raise RuntimeError("Unable to fit slit edges. "
                                               "Slit length may need adjusting "
                                               "using `edges1` and `edges2` "
                                               "parameters.")
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', AstropyUserWarning)
                            m_final = fit_model(m_init, pair, guess,
                                                sigma=n_sigma, verbose=debug)

                        n_sigma += 1
                        expected = m_final.inverse(guess)
                        if abs(expected[0] - pair[0]) < tolerance and\
                           abs(expected[1] - pair[1]) < tolerance:
                            break

                    log.fullinfo(f"Found a shift of {m_final.offset_1.value} "
                                 "and a scale of "
                                 f"{m_final.factor_2.value}.")
                    log.fullinfo("Looking for edges at "
                                 f"({expected[0]:.2f}, {expected[1]:.2f}).")

                    for loc, arr, edge in zip(expected,
                                              [diffarr_1, diffarr_2],
                                              (edge1, edge2)):

                        # The "- 1" here is because np.diff shrinks the array
                        # by 1 in taking the first derivative.
                        if (loc < buffer) or\
                           (loc > ext.data.shape[1-dispaxis] - buffer - 1):
                           log.debug(f"Not tracing at {loc:.2f} because "
                                     "it is off (or too close to) the "
                                     "detector edge.")
                           continue

                        # Trace the edge. max_shift is slightly larger than the
                        # default value of 0.05 since there's (theoretically) no
                        # chance of confusion with other lines, and some flats
                        # have enough 'tilt' to exceed that value.
                        try:
                            ref_coords, in_coords = tracing.trace_lines(
                                arr, dispaxis, start=cut,
                                initial=[loc],
                                variance=ext.variance,
                                max_missed=8,
                                step=20,
                                max_shift=0.08,
                                min_peak_value=5*ext.read_noise())
                        except ValueError:
                            # Unable to trace edge. If one edge can't be
                            # traced but the other can, the same model (shifted
                            # by the slit breadth) can be applied to the
                            # untraced edge, so just continue for now.
                            log.warning(f"Unable to trace edge at {loc:.2f}.")
                            continue

                        # This complicated bit of code parses out coordinates
                        # for the traced edges.
                        coords = np.array([list(c1) + list(c2)
                                           for c1, c2 in
                                           zip(ref_coords.T, in_coords.T)
                                           if abs(c1[dispaxis] - loc) < 1.])

                        if len(coords) == 0:
                            # Then no edge has been able to be fitted, probably
                            # because it's off the end of the CCD or very
                            # close. Just continue for now and copy the model
                            # from the other edge of the pair at the end.
                            log.warning(f"No edge was found at {loc:.2f}.")
                            continue

                        else:
                            values = np.array(
                                sorted(coords,
                                       key=lambda c: c[1 - dispaxis])).T
                            # Sorting this way returns a len-4 list, where the
                            # first two entries are the (sorted) ref_coords.

                            in_coords_new = values[2:]
                            # Log the trace.
                            min_value = in_coords_new[1 - dispaxis].min()
                            max_value = in_coords_new[1 - dispaxis].max()
                            log.debug(f"Edge at {loc:.2f} traced from "
                                      f"{min_value} to {max_value}.")

                            # Perform the fit of the coordinates for the traced
                            # edges. Use log-weighting to help ensure valid
                            # points are all considered in the trace.
                            weights = np.log(collapsed[
                                in_coords_new[1 - dispaxis].astype(int)])

                            # Create a plot of weights for inspection.
                            if debug:
                                plt.plot(in_coords_new[1-dispaxis], weights,
                                         label='Weights')
                                plt.xlabel(f'{row_or_col.capitalize()} number')
                                plt.legend()
                                plt.show()

                            # Perform the fit.
                            _fit_1d = fit_1D(
                                in_coords_new[dispaxis],
                                weights=weights,
                                domain=[0, ext.shape[1 - dispaxis] - 1],
                                points=in_coords_new[1 - dispaxis],
                                plot=debug,
                                **fit1d_params,)
                            model_fit = am.model_to_table(_fit_1d.model)

                            if _fit_1d.rms > 0.5:
                                log.warning(f"RMS of fit to {edge} edge is "
                                            f"{_fit_1d.rms:.2f} pixels. "
                                            "Consider increasing the order of "
                                            "the fit with the 'spectral_order' "
                                            "parameter")
                            else:
                                log.fullinfo(f"RMS of fit to {edge} edge is "
                                            f"{_fit_1d.rms:.2f} pixels")

                            models_dict[pair_num][edge] = model_fit
                            models_dict[pair_num][f"coords_{edge}"] = values

                    # Increment the number of pairs processed.
                    pair_num += 1

                for key, pair in models_dict.items():

                    if pair[edge1] is None and pair[edge2] is not None:
                        pair[edge1] = deepcopy(pair[edge2])
                        pair[edge1]['c0'] -= slit_widths[key]
                        log.debug(f"Copying {edge2} edge to {edge1}.")

                    if pair[edge2] is None and pair[edge1] is not None:
                        pair[edge2] = deepcopy(pair[edge1])
                        pair[edge2]['c0'] += slit_widths[key]
                        log.debug(f"Copying {edge1} edge to {edge2}.")

                # With all edges fitted (or not), create the SLITEDGE table.
                edge_num = 0
                slit_table = []
                make_table = []
                for key, pair in models_dict.items():
                    if pair[edge1] is None and pair[edge2] is None:
                        e_1, e_2 = m_final.inverse(mdf_edge_guesses[key])
                        log.warning("Couldn't fit either edge for the pair "
                                    "with edges at "
                                    f"{edge1}={e_1:.2f} and {edge2}={e_2:.2f}.\n"
                                    "This may be because the slit length "
                                    "estimate is incorrect. You may be able "
                                    "to fix this by manually giving the edges "
                                    "with the `edges1` and `edges2` parameters. "
                                    "No SLITEDGE table will be created.")
                        make_table.append(False)
                        continue

                    # If both edges have models, create a row for the table.
                    for edge in (edge1, edge2):
                        row = Table(np.array([1, edge_num]),
                                names=('slit', 'edge'))
                        row = hstack([row, pair[edge]], join_type='inner')
                        slit_table.append(row)

                        edge_num += 1
                        make_table.append(True)

                    # Even though both models exist, it's possible only one edge
                    # was actually traced (with the other model being a copy of
                    # it), so get the actual coords traced with a default of
                    # None in case nothing was traced.
                    coords_one = pair.get(f'coords_{edge1}', None)
                    coords_two = pair.get(f'coords_{edge2}', None)

                    # Assume only one edge traced for now, so spatial order 0
                    # for the model to rectify the slit. This means that the
                    # shift isn't a function of location in the spatial direction
                    # (i.e., the same shift should be applied to all pixels in
                    # the same row/column).
                    spatial_order = 0

                    if (coords_one is not None and coords_two is not None):
                        log.fullinfo("Getting coordinates from both traced edges.")
                        in_coords = [np.concatenate([a, b]) for a, b in zip(
                            coords_one[2:], coords_two[2:])]
                        ref_coords = [np.concatenate([a, b]) for a, b in zip(
                            coords_one[:2], coords_two[:2])]

                        # If both edges have been traced, use order 1 for the
                        # spatial order of the distortion model.
                        spatial_order = 1
                    else:
                        # If only one edge has been traced, get the values from it.
                        log.fullinfo("Only one was edge was traced, get "
                                     "coordinates from it.")
                        coords = coords_one if coords_two is None else coords_two
                        in_coords = coords[2:]
                        ref_coords = coords[:2]

                        # Find the value of the trace closest to the midpoint of
                        # the detector and set that as the reference pixel - in
                        # essense "rotating" the image around that point (rather
                        # than one of the endpoints of the trace).
                        half_detector = ext.shape[1 - dispaxis] // 2
                        dists = np.array([abs(n - half_detector)
                                          for n in ref_coords[1 - dispaxis]])
                        midpoint = in_coords[dispaxis][dists.argmin()]
                        ref_coords[dispaxis] = np.full_like(ref_coords[dispaxis],
                                                            midpoint)

                    if dispaxis == 0:
                        x_ord, y_ord = 1, spectral_order
                    else:
                        x_ord, y_ord = spectral_order, 1

                    m_init_2d = models.Chebyshev2D(
                        x_degree=x_ord, y_degree=y_ord,
                        x_domain=[0, ext.shape[1]-1],
                        y_domain=[0, ext.shape[0]-1])

                    # Create the distortion model from the available coords.
                    # Currently this is set up for a single slit (i.e. longslit)
                    # and will need some thinking/refactoring for XD, MOS.
                    log.stdinfo("Creating distortion model for slit rectification")
                    fixed = (spatial_order == 0)
                    model, m_final_2d, m_inverse_2d = create_distortion_model(
                        m_init_2d, dispaxis, in_coords, ref_coords, fixed)
                    model.name = 'RECT'

                # Attach the table to the extension as a new plane if none of
                # the edge pairs failed to be fit.
                if all(make_table):
                    ext.SLITEDGE = vstack(slit_table)
                    log.debug('Appending the table below as "SLITEDGE".')
                    log.debug(ext.SLITEDGE)

                    # Put the slit rectification model as the first step in the
                    # WCS if one already exists.
                    if ext.wcs is None:
                        ext.wcs = gWCS([(ext.wcs.input_frame, model),
                                        (cf.Frame2D(name="rectified"), None)])
                    else:
                        ext.wcs.insert_frame(ext.wcs.input_frame, model,
                                             cf.Frame2D(name="rectified"))
                    log.debug("The WCS for this extension is:")
                    log.debug(ext.wcs)

            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects optical distortion in science frames, using a distortion map
        (a Chebyshev2D model, usually from a processed arc) that has previously
        been attached to each input's WCS by attachWavelengthSolution.

        If the input image requires mosaicking, then this is done as part of
        the resampling, to ensure one, rather than two, interpolations.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images with appropriately-calibrated WCS.
        suffix : str
            Suffix to be added to output files.
        order : int (0 - 5)
            Order of interpolation when resampling.
        subsample : int
            Pixel subsampling factor.
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Modified input objects with distortion correct applied.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        order = params["order"]
        subsample = params["subsample"]
        do_cal = params["do_cal"]
        dq_threshold = params["dq_threshold"]

        if do_cal == 'skip':
            log.warning('Distortion correction has been turned off.')
            return adinputs

        fail = False

        adoutputs = []
        for ad in adinputs:

            # We don't check for a timestamp since it's not unreasonable
            # to do multiple distortion corrections on a single AD object

            for ext in ad:
                try:
                    idx = ext.wcs.available_frames.index('distortion_corrected')
                except (ValueError, AttributeError):
                    have_distcorr = False
                else:
                    have_distcorr = idx > 0
                if not have_distcorr:
                    log.warning('No distortion transformation attached to'
                                f' {ad.filename}, extension {ext.id}')
                    break

                # The resampling routine currently relies on a no-op forward
                # distortion model to size the output correctly (while using
                # the proper inverse for evaluating the sample points), so we
                # replace that part of the WCS with Identity here. This hack
                # gets uglier because any origin shift between arc & science
                # ROIs that has been prefixed to the distortion model needs to
                # be preserved; get rid of this at a later iteration by
                # including ROI shifts in their own frame(s).

                new_pipeline = []
                # Step through the steps in the pipeline, and replace the
                # forward transform with Identity(2) for the frame names given
                # in order to keep the output image size the same (the actual
                # transform of importance is the inverse transform, which
                # isn't touched).
                for index, step in enumerate(ext.wcs.pipeline[:idx]):
                    if ext.wcs.pipeline[index+1].frame.name in (
                            'distortion_corrected', 'rectified'):
                        prev_frame, m_distcorr = step

                        # The model must have a Mapping prior to the Chebyshev2D
                        # model(s) since coordinates have to be duplicated. Find this
                        for i in range(m_distcorr.n_submodels):
                            if isinstance(m_distcorr[i], models.Mapping):
                                break
                        else:
                            raise ValueError("Cannot find Mapping")

                        # Now determine the extent of the submodel that
                        # encompasses the overall 2D distortion, which will be
                        # a 2D->2D model
                        for j in range(i + 1, m_distcorr.n_submodels + 1):
                            try:
                                msub = m_distcorr[i:j]
                            except IndexError:
                                continue
                            if msub.n_inputs == msub.n_outputs == 2:
                                break
                        else:
                            raise ValueError("Cannot find distortion model")

                        # Name it so we can replace it
                        m_distcorr[i:j].name = "DISTCORR"
                        m_dummy = models.Identity(2)
                        m_dummy.inverse = msub.inverse
                        new_m_distcorr = m_distcorr.replace_submodel("DISTCORR",
                                                                     m_dummy)
                        new_pipeline.append((prev_frame, new_m_distcorr))
                    else:  # Keep the step unchanged.
                        new_pipeline.append(step)

                # Now recreate the WCS using the new pipeline.
                new_pipeline.extend(ext.wcs.pipeline[idx:])
                ext.wcs = gWCS(new_pipeline)

            if not have_distcorr:
                # TODO: Think about this when we have MOS/XD/IFU
                if 'sq' in self.mode or do_cal == 'force':
                    fail = True
                elif len(ad) == 1:
                    adoutputs.append(ad)
                else:
                    # In further refactoring, the mosaic WCS should get added
                    # at an earlier stage, separately from resampling.
                    log.warning('Image will be mosaicked.')
                    adoutputs.extend(self.mosaicDetectors([ad]))
                continue

            # Do all the extension WCSs contain a mosaic frame, allowing us to
            # resample them into a single mosaic at the same time as correcting
            # distortions (they won't have if the arc wasn't mosaicked)?
            mosaic = all('mosaic' in ext.wcs.available_frames if ext.wcs is not
                         None else False for ext in ad)

            if mosaic:
                ad_out = transform.resample_from_wcs(
                    ad, 'distortion_corrected', order=order,
                    subsample=subsample, parallel=False,
                    threshold=dq_threshold
                )
            else:
                for i, ext in enumerate(ad):
                    if i == 0:
                        ad_out = transform.resample_from_wcs(
                            ext, 'distortion_corrected', order=order,
                            subsample=subsample, parallel=False,
                            threshold=dq_threshold
                        )
                    else:
                        ad_out.append(
                            transform.resample_from_wcs(ext,
                                                        'distortion_corrected',
                                                        order=order,
                                                        subsample=subsample,
                                                        parallel=False,
                                                        threshold=dq_threshold)
                        )

            # The WCS gets updated by resample_from_wcs. We should also make it
            # save the (inverted) WCS pipeline components prior to resampling
            # somehow, to allow mapping rectified co-ordinates back to detector
            # pixels for calibration & inspection purposes.

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(),
                            keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        if fail:
            raise OSError("One or more input(s) missing distortion "
                          "calibration; run attachWavelengthSolution first")

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

        combine_method: {"mean", "median"}
            Method to use for combining rows/columns when extracting 1D-spectrum.
            Default: "mean"
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

        absorption : bool
            If feature type is absorption (default: "False")

        interactive : bool
            Use the interactive tool?

        resolution: int/None
            Resolution (as l/dl), to which to convolve ATRAN spectrum,
            for ATRAN linelist and reference plot generation.
            If None, the default value for the instrument/mode is used.

        wv_band: {'20', '50', '80', '100', 'header'}
            Water vapour content (as percentile) to be used for ATRAN model
            selection. If "header", then the value from the header is used.

        num_atran_lines: int/None
            Maximum number of lines with largest weigths (within a wvl bin) to be
            included in the generated ATRAN line list.

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
        # This parameter won't be available in some children classes
        absorption = params.get("absorption", False)

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
                self.generated_linelist = False
                log.stdinfo(f"Read arc line list {arc_file}")

        for ad in adinputs:
            # In the case of wavecal from telluric absorption in object spectrum,
            # use the location of the brightest aperture as central row/column for
            # 1D extraction.
            if absorption:
                if params["center"] is None:
                    try:
                        aptable = ad[0].APERTURE
                        params["center"] = int(aptable['c0'].data[0])
                    except (AttributeError, KeyError):
                        log.error("Could not find aperture locations in "
                                    f"{ad.filename} - continuing")
                        continue

        # Pass the primitive configuration to the interactive object.
        config = copy(self.params[self.myself()])
        config.update(**params)

        for ad in adinputs:
            log.stdinfo(f"Determining wavelength solution for {ad.filename}")
            uiparams = UIParameters(
                    config, reinit_params=["center", "nsum", "min_snr", "min_sep",
                                           "fwidth", "central_wavelength", "dispersion",
                                                       "in_vacuo"])
            if self.generated_linelist:
                # Add some extra parameters to the UI when the linelist gets generated on-the-fly
                linelist_pars = {"atran_linelist_pars": ["num_atran_lines", "resolution", "wv_band"]}
                uiparams.reinit_params.append(linelist_pars)

            uiparams.fields["center"].max = min(
                ext.shape[ext.dispersion_axis() - 1] for ext in ad)

            # In case when absorption lines are used for wavelength calibration,
            # we set the data to negative to make absorption lines into emission
            # lines, and perform all calculations on this negative data.
            if absorption:
                calc_ad = deepcopy(ad)
                for i, data in enumerate(ad.data):
                   calc_ad[i].data = -data
            else:
                calc_ad = ad

            if interactive:
                all_fp_init = [fit_1D.translate_params(
                    {**params, "function": "chebyshev"})] * len(ad)
                # This feels like I shouldn't have to do it here
                domains = []
                for ext in ad:
                    axis = 0 if ext.data.ndim == 1 else 2 - ext.dispersion_axis()
                    domains.append([0, ext.shape[axis] - 1])
                reconstruct_points = partial(wavecal.create_interactive_inputs, calc_ad, p=self,
                            linelist=linelist, bad_bits=DQ.not_signal)


                visualizer = WavelengthSolutionVisualizer(
                    reconstruct_points, all_fp_init,
                    modal_message="Re-extracting 1D spectra",
                    tab_name_fmt=lambda i: f"Slit {i+1}",
                    xlabel="Fitted wavelength (nm)", ylabel="Non-linear component (nm)",
                    domains=domains,
                    absorption=absorption,
                    title="Wavelength Solution",
                    primitive_name=self.myself(),
                    filename_info=ad.filename,
                    enable_regions=False, plot_ratios=False, plot_height=350,
                    ui_params=uiparams)
                geminidr.interactive.server.interactive_fitter(visualizer)
                for ext, fit1d, image, other in zip(ad, visualizer.results(),
                                                    visualizer.image, visualizer.meta):
                    fit1d.image = image
                    wavecal.update_wcs_with_solution(ext, fit1d, other, config)
            else:
                for ext, calc_ext in zip(ad, calc_ad):
                    if len(ad) > 1:
                        log.stdinfo(f"Determining solution for extension {ext.id}")

                    input_data, fit1d, acceptable_fit = wavecal.get_automated_fit(
                        calc_ext, uiparams, p=self, linelist=linelist, bad_bits=DQ.not_signal)
                    if not acceptable_fit:
                        log.warning("No acceptable wavelength solution found")
                    else:
                        wavecal.update_wcs_with_solution(ext, fit1d, input_data, config)
                        wavecal.save_fit_as_pdf(
                            input_data["spectrum"], fit1d.points[~fit1d.mask],
                            fit1d.image[~fit1d.mask], ad.filename)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def extractSpectra(self, adinputs=None, **params):
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
        method : {'standard', optimal', 'default'}
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
                             (self.timestamp_keys['subtractSky'] not in ad.phu and
                              self.timestamp_keys['skyCorrectFromSlit'] not in ad.phu))
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
                    apmask = np.logical_or.reduce(
                        [ap.aperture_mask(ext, width=width, grow=grow)
                         for ap in apertures])
                    # Add a mask of non-pixels (unilluminated/no-data). But
                    # we only want to add rows that are completely data-free
                    # or else the GMOS chip gaps (for example) will cause
                    # sky apertures to be rejected
                    if ext.mask is not None:
                        if dispaxis == 1:
                            apmask = (apmask.T | np.all(
                                ext.mask & (DQ.unilluminated | DQ.no_data),
                                axis=1)).T
                        else:
                            apmask |= np.all(ext.mask & (DQ.unilluminated |
                                                         DQ.no_data), axis=0)

                # Calculate world coords at middle of each dispersed spectrum
                pix_coords = [[0.5 * (length-1)] * len(apertures)
                              for length in ext.shape[::-1]]
                pix_coords[dispaxis] = [ap.center for ap in apertures]
                wcs_coords = ext.wcs(*pix_coords)
                sky_axes = None
                if isinstance(ext.wcs.output_frame, cf.CompositeFrame):
                    for frame in ext.wcs.output_frame.frames:
                        if isinstance(frame, cf.CelestialFrame):
                            try:
                                sky_axes = [frame.axes_order[frame.axes_names.index(axis)]
                                            for axis in ('lon', 'lat')]
                            except IndexError:
                                pass
                            break

                if method == "default":
                    this_method = "optimal" if 'STANDARD' in ad.tags else "aperture"
                else:
                    this_method = method

                for apnum, (aperture, *coords) in enumerate(zip(apertures, *wcs_coords), start=1):
                    log.stdinfo(f"    Extracting spectrum from aperture {apnum}")
                    self.viewer.width = 2
                    self.viewer.color = colors[(apnum-1) % len(colors)]
                    ndd_spec = aperture.extract(
                        ext, width=width, method=this_method, viewer=self.viewer if debug else None)

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
                        in_frame = astrodata.wcs.pixel_frame(naxes=1)
                        out_frame = cf.SpectralFrame(unit=u.nm, name='world',
                                                     axes_names=axes_names)
                        ext_spec.wcs = gWCS([(in_frame, wave_model),
                                             (out_frame, None)])
                    ext_spec.hdr[ad._keyword_for('aperture_number')] = apnum
                    center = aperture.model.c0.value
                    ext_spec.hdr['XTRACTED'] = (
                        center, f"Spectrum extracted from {direction} {int(center+0.5)}")
                    for i, kw in enumerate(['XTRACTLO', 'XTRACTHI']):
                        ext_spec.hdr[kw] = (aperture.last_extraction[i],
                                            self.keyword_comments[kw])
                    if sky_axes:
                        for i, kw in zip(sky_axes, ['XTRACTRA', 'XTRACTDE']):
                            ext_spec.hdr[kw] = (coords[i], self.keyword_comments[kw])

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

    def findApertures(self, adinputs=None, **params):
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
        min_snr : float
            minimum S/N ratio for detecting peaks
        use_snr : bool
            Convert data to SNR per pixel before collapsing and peak-finding?
        threshold : float (0 - 1)
            parameter describing either the height above background (relative
            to peak) at which to define the edges of the aperture.
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
            'max_apertures', 'min_sky_region', 'percentile', 'section',
            'threshold', 'min_snr', 'use_snr', 'max_separation')}

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
                    ext_oriented = ext.__class__(
                        nddata=ext.nddata.T, phu=ad.phu, is_single=True)
                else:
                    ext_oriented = ext

                if interactive:
                    # build config for interactive
                    config = self.params[self.myself()]
                    config.update(**params)
                    reinit_params = ["percentile", "min_sky_region", "use_snr", "min_snr", "section", "max_apertures",
                                     "threshold", "max_separation"]
                    title_overrides = {
                        "percentile": "Percentile (use mean if no value)",
                        "min_sky_region": "Min sky region",
                        "use_snr": "Use S/N ratio in spatial profile?",
                        "min_snr": "SNR threshold for peak detection",
                        "max_apertures": "Max Apertures (empty means no limit)",
                        "threshold": "Threshold",
                        "max_separation": "Maximum separation from target",
                    }
                    ui_params = UIParameters(config, reinit_params=reinit_params, extras={},
                                             title_overrides=title_overrides,
                                             placeholders={"section": "e.g. 100:900,1500:2000"})

                    # pass "direction" purely for logging purposes
                    filename = ad.filename
                    if not filename:
                        filename = ad.orig_filename
                    locations, all_limits = interactive_find_source_apertures(
                        ext_oriented, ui_params=ui_params, filename=filename, **aper_params,
                        direction="column" if dispaxis == 0 else "row")
                else:
                    locations, all_limits, _, _ = tracing.find_apertures(
                        ext_oriented, **aper_params)

                if locations is None or len(locations) == 0:
                    # Delete existing APERTURE table
                    if 'APERTURE' in ext.tables:
                        del ext.APERTURE
                    continue

                apmodels, sizes = [], []
                for i, (loc, limits) in enumerate(zip(locations, all_limits), start=1):
                    apmodels.append(models.Chebyshev1D(
                        degree=0, domain=[0, npix-1], c0=loc))
                    lower, upper = limits - loc
                    log.stdinfo(f"Aperture {i} found at {loc:.2f} "
                                f"({lower:.2f}, +{upper:.2f})")
                    if lower > 0 or upper < 0:
                        log.warning("Problem with automated sizing of "
                                    f"aperture {i}")
                    sizes.append((lower, upper))
                ext.APERTURE = make_aperture_table(apmodels, limits=sizes)

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

        spectral_order, spatial_order : int or None, optional
            Order for fitting and subtracting object continuum and sky line
            models, prior to running the main cosmic ray detection algorithm.
            When None, defaults are used, according to the image size (as in
            the IRAF task gemcrspec). To control which fits are performed, use
            the bkgmodel parameter.

       bkgmodel : {'both', 'object', 'skyline', 'none'}, optional
           Set which background model(s) to use, between 'object', 'skyline',
           'both', or 'none'. Different data may get better results with
           different background models.
           'both': Use both object and sky line models.
           'object': Use object model only.
           'skyline': Use sky line model only.
           'none': Don't use a background model.
           Default: 'skyline'.

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
        x_order_in = params.pop('spectral_order')
        y_order_in = params.pop('spatial_order')
        bkgmodel = params.pop('bkgmodel')

        fit_1D_params = dict(
            plot=debug,
            niter=params.pop('bkgfit_niter'),
            sigma_lower=params.pop('bkgfit_lsigma'),
            sigma_upper=params.pop('bkgfit_hsigma'),
        )

        log.fullinfo("Input parameters:\n")
        log.fullinfo(f"  spectral_order: {x_order_in}")
        log.fullinfo(f"  spatial_order: {y_order_in}")
        log.fullinfo(f"  bkgmodel: {bkgmodel}")
        log.fullinfo(f"  sigclip: {params['sigclip']}")
        log.fullinfo(f"  sigfrac: {params['sigfrac']}\n")

        for ad in adinputs:
            is_in_adu = ad[0].is_in_adu()
            if not is_in_adu:
                # astroscrappy takes data in adu
                for ext in ad:
                    ext.divide(ext.gain())

            # tile extensions by CCD to limit the number of edges
            array_info = gt.array_information(ad)
            ad_tiled = self.tileArrays([ad], tile_all=False)[0]

            # Create a modified version of the debug plot for inspection
            fig, axes = plt.subplots(5, 3, sharex=True, sharey=True,
                                     tight_layout=True)
            if debug:
                # This counter-intuitive step prevents the empty figure from
                # being shown by calls to pyplot.show(), but still allows it to
                # be drawn to and modified (somehow...).
                plt.close(fig)

            for i, ext in enumerate(ad_tiled):
                dispaxis = 2 - ext.dispersion_axis()

                # Use default orders from gemcrspec (from Bryan):
                # ny, nx = ext.shape
                # spectral_order = 9 if x_order_in is None else x_order_in
                # spatial_order = ((2 if ny < 50 else 3 if ny < 80 else 5)
                #                 if y_order_in is None else y_order_in)
                spectral_order = x_order_in
                spatial_order = y_order_in

                if ext.mask is not None:
                    data = np.ma.array(ext.data, mask=ext.mask != 0)
                    mask = (ext.mask & bitmask) > 0
                    weights = (ext.mask == 0).astype(int)
                else:
                    data = ext.data
                    mask = None
                    weights = None

                # Set up the background and models to be blank initially:
                background = np.zeros(ext.shape)

                # Fit the object spectrum:
                if bkgmodel in ('both', 'object'):
                    objfit = fit_1D(data,
                                    function='legendre',
                                    axis=dispaxis,
                                    order=spectral_order,
                                    weights=weights,
                                    **fit_1D_params).evaluate()
                else:
                    objfit = np.zeros(ext.shape)
                if debug:
                    ext.OBJFIT = objfit.copy()

                background += objfit

                # If fitting both models, subtracting objfit from the data
                # ensures sky background isn't fitted twice:
                skyfit_input = data - objfit

                # Fit sky lines:
                if bkgmodel in('both', 'skyline'):
                    skyfit = fit_1D(skyfit_input,
                                    function='legendre',
                                    axis=1 - dispaxis,
                                    order=spatial_order,
                                    weights=weights,
                                    **fit_1D_params).evaluate()
                else:
                    skyfit = np.zeros(ext.shape)
                if debug:
                    ext.SKYFIT = skyfit

                background += skyfit

                # Run astroscrappy's detect_cosmics. We use the variance array
                # because it takes into account the different read noises if
                # the data has been tiled
                crmask, _ = detect_cosmics(ext.data,
                                           inmask=mask,
                                           inbkg=background,
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
                    plot_cosmics(ext, objfit, skyfit, crmask)

                plot_cosmics(ext, objfit, skyfit, crmask, axes=axes[:, i])

                # Free up memory.
                skyfit, objfit, skyfit_input = None, None, None

            # Save the figure
            figy, figx = ext.data.shape
            fig.set_size_inches(figx*3/300, figy*5/300)
            figname, _ = os.path.splitext(ad.orig_filename)
            figname = figname + '_flagCosmicRays.pdf'
            # This context manager prevents two harmless RuntimeWarnings from
            # image normalization if bkgmodel != 'both' (due to empty panels)
            # which we don't want to worry users with.
            with np.errstate(divide='ignore', invalid='ignore'):
                fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)

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
                if 'sq' in self.mode or do_cal == 'force':
                    raise OSError("No processed stndard listed for "
                                  f"{ad.filename}")
                else:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no standard was specified")
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
            if std.path:
                add_provenance(ad, std.filename, md5sum(std.path) or "", self.myself())

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
                                                trim_spectral=False)[0]
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            adoutputs.append(ad_out)

        return adoutputs


    def maskBeyondSlit(self, adinputs=None, **params):
        """
        This primitive masks unilluminated regions defined by a mask definition
        file (MDF).

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Spectra with unilluminated regions.
        suffix : str
            Suffix to append to the filename.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Spectra with regions outside the illuminated region masked.

        """
        # Set up log
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        sfx = params["suffix"]

        # Parse parameters
        debug = params['debug']

        for ad in adinputs:
            log.stdinfo(f"Masking unilluminated regions in {ad.filename}")

            for ext in ad:

                # If there's no SLITEDGE table from determineSlitEdge, we can't
                # create a mask, so just pass.
                try:
                    slittab = ext.SLITEDGE
                except AttributeError:
                    log.warning(f"No SLITEDGE table found for {ad.filename} - "
                                "no masking was performed.")
                    continue

                dispaxis = 2 - ext.dispersion_axis()
                # Create pairs of slit edge models by zipping consecutive pairs
                # of entries from the table.
                pairs = [(m, n) for m, n in zip(islice(slittab, 0, None, 2),
                                                islice(slittab, 1, None, 2))]

                if len(pairs) == 1:  # longslit data

                    model1 = am.table_to_model(pairs[0][0])
                    model2 = am.table_to_model(pairs[0][1])

                    height = ext.data.shape[0 - dispaxis]
                    width = ext.data.shape[1 - dispaxis]

                    # Create a NumPy mesh grid to hold the mask.
                    x, y = np.mgrid[0:width, 0:height]

                    if dispaxis == 0:
                        y1 = model1(y)
                        y2 = model2(y)

                        # Mask outside the two edges of the (single) longslit.
                        mask = (x < y1) | (x > y2)

                    elif dispaxis == 1:
                        x1 = model1(x)
                        x2 = model2(x)

                        mask = (y < x1) | (y > x2)

                    ext.mask |= mask.T * DQ.unilluminated

                    if debug:
                        # Show a plot of the DQ plane after applying the mask
                        plt.subplot(111)
                        plt.imshow(ext.mask, origin='lower', cmap='gray')
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.show()

                # TODO: Handle MOS data.

            # Update the filename.
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs


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
            number of rows/columns to average (about "center")
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
        interactive : bool
            set to activate an interactive preview to fine tune the input parameters
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        threshold = params["threshold"]
        interactive_reduce = params["interactive"]

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

            masked_data_arr = list()
            x_arr = list()
            weights_arr = list()
            threshold_mask_arr = list()
            saved_thresholds = list()  # for clipping values in the final eval, save the calculated thresholds when
                                       # we reconstruct points

            # This will loop over MOS slits or XD orders
            def reconstruct_points(ui_params):
                masked_data_arr.clear()
                x_arr.clear()
                weights_arr.clear()
                threshold_mask_arr.clear()
                saved_thresholds.clear()

                for ext in admos:
                    dispaxis = 2 - ext.dispersion_axis()  # python sense
                    direction = "row" if dispaxis == 1 else "column"

                    data, mask, variance, extract_slice = tracing.average_along_slit(
                        ext, center=ui_params.center, nsum=ui_params.nsum)
                    log.stdinfo("Extracting 1D spectrum from {}s {} to {}".
                                format(direction, extract_slice.start + 1, extract_slice.stop))
                    mask |= (DQ.no_data * (variance == 0))  # Ignore var=0 points
                    slices = _ezclump((mask & (DQ.no_data | DQ.unilluminated)) == 0)

                    masked_data = np.ma.masked_array(data, mask=mask)
                    weights = np.sqrt(np.where(variance > 0, 1. / variance, 0.))
                    # uncomment this to use if we want to calculate the waves as our x inputs
                    # and wire it up appropriately
                    # center = (extract_slice.start + extract_slice.stop) // 2
                    # waves = ext.wcs(range(len(masked_data)),
                    #                 np.full_like(masked_data, center))[0]

                    # We're only going to do CCD-to-CCD normalization if we've
                    # done the mosaicking in this primitive; if not, we assume
                    # the user has already taken care of it (if it's required).
                    nslices = len(slices)
                    if nslices > 1 and mosaicked:
                        coeffs = np.ones((nslices - 1,))
                        boundaries = list(slice_.stop for slice_ in slices[:-1])
                        result = optimize.minimize(QESpline, coeffs, args=(waves, masked_data,
                                                                           weights, boundaries,
                                                                           20),
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
                    x_arr.append(np.arange(len(masked_data)))
                    masked_data_arr.append(masked_data)
                    weights_arr.append(weights)
                    maxy = masked_data.max()
                    threshold_mask_arr.append(np.where(masked_data/maxy < ui_params.threshold, 1, 0))
                    saved_thresholds.append(maxy * ui_params.threshold)
                return { "y": masked_data_arr, "x": x_arr,
                         "weights": weights_arr, "threshold_mask": threshold_mask_arr }

            config = self.params[self.myself()]
            config.update(**params)
            dispaxis = 2 - ad[0].dispersion_axis()
            line = 'row' if dispaxis == 1 else 'column'
            title_overrides = {"center": f"Central {line} to extract",
                               "nsum": f"Number of {line} to average",
                               "threshold": "Threshold for unilluminated pixels"}
            uiparams = UIParameters(config, reinit_params=["center", "nsum", "threshold"],
                                    title_overrides=title_overrides)

            # let's update the max center to something reasonable
            npix = ad[0].shape[1 - dispaxis]
            uiparams.fields['center'].max = npix
            uiparams.fields['nsum'].max = npix

            xaxis_label = 'x (pixels)' if dispaxis == 1 else 'y (pixels)'

            data = reconstruct_points(uiparams)

            fit1d_arr = list()

            if interactive_reduce:
                all_domains = list()
                all_fp_init = list()
                for ext, x in zip(admos, x_arr):
                    pixels = np.arange(ext.shape[1])
                    all_domains.append([min(pixels), max(pixels)])
                    dispaxis = 2 - ext.dispersion_axis()
                    all_fp_init.append(fit_1D.translate_params(params))

                config = self.params[self.myself()]
                config.update(**params)

                if ad.filename:
                    filename_info = ad.filename
                else:
                    filename_info = ''

                visualizer = fit1d.Fit1DVisualizer(reconstruct_points,
                                                   all_fp_init,
                                                   tab_name_fmt="Array {}",
                                                   xlabel=xaxis_label, ylabel='counts',
                                                   domains=all_domains,
                                                   title="Normalize Flat",
                                                   primitive_name="normalizeFlat",
                                                   filename_info=filename_info,
                                                   enable_user_masking=False,
                                                   enable_regions=True,
                                                   help_text=NORMALIZE_FLAT_HELP_TEXT,
                                                   recalc_inputs_above=False,
                                                   # modal_message="Recalculating",
                                                   ui_params=uiparams)
                geminidr.interactive.server.interactive_fitter(visualizer)
                fit1d_arr = visualizer.results()
            else:
                for ext, masked_data, x, weights, threshold_value \
                        in zip(admos, masked_data_arr, x_arr, weights_arr,
                               saved_thresholds):
                    masked_data.mask |= np.where(masked_data.data < threshold_value, True, False)
                    fitted_data = fit_1D(masked_data, points=x, weights=weights,
                                         **fit1d_params)
                    fit1d_arr.append(fitted_data)

            for ext, fitted_data, x, threshold_mask, masked_data, threshold_value \
                    in zip(admos, fit1d_arr, x_arr, threshold_mask_arr, masked_data_arr, saved_thresholds):
                if not mosaicked:
                    # In the case where this was run interactively, the resulting fit has pre-masked points (x).
                    # This happens before the interactive code builds the fit_1D.  Using the default evaluate()
                    # points on these fit_1Ds will send a trimmed list of points, resulting in trimmed output which
                    # we don't want - and also is inconsistent with the non-interactive code where the array
                    # was masked but the x values were not.
                    #
                    # Instead, we want to call to evaluate with an explicit set of points using our pre-masked copy
                    # of the x values to get a consistent and correctly-sized output.
                    fdeval = fitted_data.evaluate(points=x)
                    flat_data = np.tile(fdeval, (ext.shape[1-dispaxis], 1))
                    flat_mask = at.transpose_if_needed(
                        np.tile(np.where(threshold_mask > 0,
                                         DQ.unilluminated, DQ.good),
                                (ext.shape[1-dispaxis], 1)).astype(DQ.datatype),
                        transpose=(dispaxis == 0))[0]
                    ext.divide(at.transpose_if_needed(flat_data, transpose=(dispaxis == 0))[0])

                    ext.data[flat_mask>0] = threshold_value

                    if ext.mask is None:
                        ext.mask = flat_mask
                    else:
                        ext.mask |= flat_mask

                    pass

            # If we've mosaicked, there's only one extension
            # We forward transform the input pixels, take the transformed
            # coordinate along the dispersion direction, and evaluate the
            # spline there.
            if mosaicked:
                raise NotImplementedError("Mosaicked data handling not supported for core normalizeFlat")
                #origin = admos.nddata[0].meta.pop('transform')['origin']
                #origin_shift = reduce(Model.__and__, [models.Shift(-s) for s in origin[::-1]])
                for ext, wcs in zip(ad, orig_wcs):
                    ypix, xpix = np.mgrid[:ext.shape[0], :ext.shape[1]]
                    waves = wcs(xpix, ypix)[0]
                    flat_data = np.array([fit1d.evaluate(w) for w in waves])
                    #t = ext.wcs.get_transform(ext.wcs.input_frame, "mosaic") | origin_shift
                    #geomap = transform.GeoMap(t, ext.shape, inverse=True)
                    #flat_data = fit1d.evaluate(geomap.coords[dispaxis])
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
        trim_spatial : bool
            Output data will cover the intersection (rather than union) of
            the inputs' spatial coverage?
        trim_spectral: bool
            Output data will cover the intersection (rather than union) of
            the inputs' wavelength coverage?
        force_linear : bool
            Force a linear output wavelength solution?
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.

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
        trim_spatial = params["trim_spatial"]
        trim_spectral = params["trim_spectral"]
        force_linear = params["force_linear"]
        dq_threshold = params["dq_threshold"]

        # Check that all ad objects are either 1D or 2D
        ndim = {len(ext.shape) for ad in adinputs for ext in ad}
        if len(ndim) == 0:
            log.warning('Input list empty. Doing nothing.')
            return adinputs
        elif len(ndim) != 1:
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
            dispaxis = {ad[0].dispersion_axis() for ad in adinputs}
            if len(dispaxis) > 1:  # this shouldn't happen!
                raise ValueError('Not all inputs have the same dispersion axis')
            dispaxis_wcs = dispaxis.pop() - 1  # for gWCS axes
            dispaxis = ndim - 1 - dispaxis_wcs  # python sense
            # Store these values for later!
            refad = adinputs[0]
            ref_coords = refad[0].wcs(*((dim-1)/2 for dim in refad[0].shape))
            ref_pixels = [np.asarray(ad[0].wcs.invert(*ref_coords)[::-1])
                          for ad in adinputs]
            # Locations in frame of reference AD. The spectral axis is
            # unimportant here.
            all_corners = [(np.array(at.get_corners(ad[0].shape)) -
                            r + ref_pixels[0]).T.astype(int)
                           for ad, r in zip(adinputs, ref_pixels)]

        # If only one variable is missing we compute it from the others
        nparams = 4 - [w1, w2, dw, npix].count(None)
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
                    elif trim_spectral:
                        w1out = max(w1out, model_info['w1'])
                    else:
                        w1out = min(w1out, model_info['w1'])

                if w2 is None:
                    if w2out is None:
                        w2out = model_info['w2']
                    elif trim_spectral:
                        w2out = min(w2out, model_info['w2'])
                    else:
                        w2out = max(w2out, model_info['w2'])
            info.append(adinfo)

        if trim_spectral:
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

        # Now let's think about the spatial direction
        if ndim > 1:
            if trim_spatial:
                if ndim == 2:
                    mins = [min(ac[dispaxis_wcs]) for ac in all_corners]
                    maxs = [max(ac[dispaxis_wcs]) for ac in all_corners]
                    origin = [max(mins)] * 2
                    output_shape = [min(maxs) - max(mins) + 1] * 2
                else:  # TODO: revisit!
                    # for cubes, treat the imaging plane like the Image version
                    # and trim to the reference, not the intersection
                    origin = [0] * ndim
                    output_shape = list(refad[0].shape)
            else:
                origin = np.concatenate(all_corners, axis=1).min(axis=1)
                output_shape = list(np.concatenate(all_corners, axis=1).max(axis=1) - origin + 1)
            output_shape[dispaxis] = npixout
            origin[dispaxis] = 0
        else:
            origin = (0,)
            output_shape = (npixout,)

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
                    spatial_offset = reduce(
                        Model.__and__, [models.Shift(r0 - ref_pixels[i][j])
                                        for j, r0 in enumerate(ref_pixels[0]) if j != dispaxis])
                    if dispaxis == 0:
                        resampling_model = spatial_offset & wave_resample
                    else:
                        resampling_model = wave_resample & spatial_offset

                this_conserve = conserve_or_interpolate(ext, user_conserve=conserve,
                                        flux_calibrated=flux_calibrated, log=log)

                if i == 0 and not linearize:
                    log.fullinfo(f"{ad.filename}: No interpolation")
                msg = "Resampling"
                if linearize:
                    msg += " and linearizing"
                log.stdinfo(f"{msg} {extn}: w1={w1out:.3f} w2={w2out:.3f} "
                            f"dw={dwout:.3f} npix={npixout}")

                # If we resample to a coarser pixel scale, we may
                # interpolate over features. We avoid this by subsampling
                # back to the original pixel scale (approximately).
                input_dw = info[i][iext]['dw']
                subsample = int(np.ceil(abs(dwout / input_dw) - 0.1))
                attributes = [attr for attr in ('data', 'mask', 'variance')
                              if getattr(ext, attr) is not None]

                resampled_frame = copy(ext.wcs.input_frame)
                resampled_frame.name = 'resampled'
                ext.wcs = gWCS([(ext.wcs.input_frame, resampling_model),
                                (resampled_frame, new_wcs_model),
                                (ext.wcs.output_frame, None)])

                new_ext = transform.resample_from_wcs(
                    ext, 'resampled', subsample=subsample,
                    attributes=attributes, conserve=this_conserve,
                    origin=origin, output_shape=output_shape,
                    threshold=dq_threshold)
                if iext == 0:
                    ad_out = new_ext
                else:
                    ad_out.append(new_ext[0])

                # We attempt to modify the APERTURE table (if it exists) so
                # that it's still relevant. This involved applying a shift
                # and redefining the domain to the pixel range that corresponds
                # to the same wavelength range as before. This is still not
                # perfect though, since the location of a specific wavelength
                # within the domain (the normalized coordinate) will have
                # changed slightly. The solution to this is to INITIALLY define
                # the APERTURE model as a function of wavelength, not pixel.
                # Currently this is accurate to <0.1 pixel for GMOS.
                # TODO? Define APERTURE as a function of wavelength, not pixel.
                if ndim == 2 and hasattr(ext, 'APERTURE'):
                    offset = spatial_offset.offset.value
                    log.fullinfo("Shifting aperture locations by "
                                 f"{offset:.2f} pixels")
                    apmodels = [am.table_to_model(row) for row in ext.APERTURE]
                    for model in apmodels:
                        model.c0 += offset
                        model.domain = wave_resample(model.domain)
                    ad_out[-1].APERTURE = make_aperture_table(
                        apmodels, existing_table=ext.APERTURE)

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
        :meth:`~geminidr.core.primitives_spect.Spect.findApertures`,
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        debug_plot = params["debug_plot"]
        fit1d_params = fit_1D.translate_params(params)
        interactive = params["interactive"]

        def calc_sky_coords(ad: AstroData, apgrow=0, interactive_mode=False):
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
            apgrow : float
                Aperture avoidance distance (pixels)
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
                csc_sky_mask = (np.zeros_like(csc_ext.data, dtype=bool)
                                if csc_ext.mask is None else
                                (csc_ext.mask & DQ.not_signal).astype(bool))

                # for interactive mode, we aggregate an aperture mask separately
                # for the UI
                csc_aperture_mask = (np.zeros_like(csc_ext.data, dtype=bool))

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
                        csc_aperture_mask |= aperture_mask

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
                if csc_ext.mask is not None:
                    no_data = (np.bitwise_and.reduce(csc_ext.mask, axis=csc_axis) &
                               DQ.no_data).astype(bool)
                    if csc_axis == 0:
                        csc_sky_mask ^= no_data
                    else:
                        csc_sky_mask ^= no_data[:, None]

                csc_ext.data, csc_sky_mask, csc_sky_weights = \
                    transpose_if_needed(csc_ext.data, csc_sky_mask, csc_sky_weights, transpose=csc_axis != 0)
                if interactive_mode:
                    csc_aperture_mask = \
                        transpose_if_needed(csc_aperture_mask, transpose=csc_axis != 0)[0]
                    yield csc_ext, csc_sky_mask, csc_sky_weights, csc_aperture_mask
                else:
                    yield csc_ext, csc_sky_mask | csc_aperture_mask, csc_sky_weights

        def recalc_fn(ad: AstroData, ui_parms: UIParameters):
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
            ui_parms : :class:`~geminidr.interactive.interactive.UIParameters`
                configuration for the primitive, including extra controls

            Returns
            -------
            :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, :class:`~numpy.ndarray`
                Yields a list of tupes with the pixel, sky, sky_weights

            See Also
            --------
            :meth:`~geminidr.core.primitives_spect.Spect.skyCorrectFromSlit.calc_sky_coords`
            """
            # pylint: disable=unused-argument
            c = max(0, ui_parms.values['col'] - 1)
            apgrow = ui_parms.values['aperture_growth']
            # TODO alternatively, save these 3 arrays for faster recalc
            # here I am rerunning all the above calculations whenever a col select is made
            data = {"x": [], "y": [], "weights": [], "aperture_mask": []}
            for rc_ext, rc_sky_mask, rc_sky_weights, rc_aper_mask in \
                    calc_sky_coords(ad, apgrow=apgrow, interactive_mode=True):
                # TODO: FIX THIS TO PAY ATTENTION TO ORIENTATION!!!!!
                if rc_ext.dispersion_axis() == 1:
                    data["weights"].append(None if rc_sky_weights is None else rc_sky_weights[:, c])
                    data["aperture_mask"].append(rc_aper_mask[:, c])
                    rc_sky = np.ma.masked_array(rc_ext.data[:, c], mask=rc_sky_mask[:, c])
                else:
                    data["weights"].append(None if rc_sky_weights is None else rc_sky_weights[c])
                    data["aperture_mask"].append(rc_aper_mask[c])
                    rc_sky = np.ma.masked_array(rc_ext.data[c], mask=rc_sky_mask[c])
                data["x"].append(np.arange(rc_sky.size))
                data["y"].append(rc_sky)
            return data

        final_parms = list()
        apgrow = None  # for saving selected aperture_grow values, if interactive

        if interactive:
            apgrow = list()
            # build config for interactive
            config = self.params[self.myself()]
            config.update(**params)

            # Create a 'col' parameter to add to the UI so the user can select the column they
            # want to fit.
            # We pass a default column at the 1/3 mark, since dead center is flat
            axis = adinputs[0].dispersion_axis()[0] - 1  # python sense
            ncols = adinputs[0].shape[0][1 if axis == 0 else 0]
            reinit_params = ["col", "aperture_growth"]
            reinit_extras = {"col": RangeField(doc="Column of data", dtype=int, default=int(ncols / 2),
                                               min=1, max=ncols)}

            # Build the set of input shapes and count the total extensions while we are at it
            for ad in adinputs:
                all_shapes = []
                count = 0
                for ext in ad:
                    axis = ext.dispersion_axis() - 1  # python sense
                    count = count+1
                    all_shapes.append((0, ext.shape[axis]))  # extracting single line for interactive

                # Get filename to display in visualizer
                filename_info = getattr(ad, 'filename', '')

                # get the fit parameters
                fit1d_params = fit_1D.translate_params(params)
                ui_params = UIParameters(config, reinit_params=reinit_params, extras=reinit_extras)
                visualizer = fit1d.Fit1DVisualizer(lambda ui_params: recalc_fn(ad, ui_params),
                                                   fitting_parameters=[fit1d_params]*count,
                                                   tab_name_fmt=lambda i: f"Slit {i+1}",
                                                   xlabel='Row',
                                                   ylabel='Signal',
                                                   domains=all_shapes,
                                                   title="Sky Correct From Slit",
                                                   primitive_name="skyCorrectFromSlit",
                                                   filename_info=filename_info,
                                                   help_text=SKY_CORRECT_FROM_SLIT_HELP_TEXT,
                                                   plot_ratios=False,
                                                   enable_user_masking=False,
                                                   recalc_inputs_above=True,
                                                   ui_params=ui_params,
                                                   reinit_live=True)
                geminidr.interactive.server.interactive_fitter(visualizer)

                # Pull out the final parameters to use as inputs doing the real fit
                fit_results = visualizer.results()
                final_parms_exts = list()
                apgrow.append(ui_params.values['aperture_growth'])
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
            if apgrow:
                # get value set in the interactive tool
                apg = apgrow[idx]
            else:
                # get value for aperture growth from config
                apg = params["aperture_growth"]
            for ext, sky_mask, sky_weights in calc_sky_coords(ad, apgrow=apg):
                axis = 0  # Note: transposed already
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
        :meth:`~geminidr.core.primitives_spect.Spect.findApertures`

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

                    title_overrides = {
                        'max_missed': 'Max Missed',
                        'max_shift':  'Max Shifted',
                        'nsum':       'Lines to sum',
                        'step':       'Tracing step',
                    }
                    ui_params = UIParameters(_config,
                                             reinit_params=["max_missed", "max_shift", "nsum", "step"],
                                             title_overrides=title_overrides)
                    aperture_models = interactive_trace_apertures(
                        ext, fit1d_params, ui_params=ui_params)
                else:
                    dispaxis = 2 - ext.dispersion_axis()  # python sense
                    aperture_models = []

                    # For efficiency, we would like to trace all sources
                    #  simultaneously (like we do with arc lines), but we need
                    #  to start somewhere the source is bright enough, and there
                    #  may not be a single location where that is true for all
                    #  sources
                    all_ref_coords = np.array([])
                    for i, loc in enumerate(locations):
                        c0 = int(loc + 0.5)
                        spectrum = ext.data[c0, nsum:-nsum] if dispaxis == 1 else ext.data[nsum:-nsum, c0]
                        if ext.mask is None:
                            start = np.argmax(at.boxcar(spectrum, size=20)) + nsum
                        else:
                            good = ((ext.mask[c0, nsum:-nsum] if dispaxis == 1 else
                                     ext.mask[nsum:-nsum, c0]) & DQ.not_signal) == 0

                            start = nsum + np.arange(spectrum.size)[good][np.argmax(
                                at.boxcar(spectrum[good], size=20))]
                        log.stdinfo(f"{ad.filename}: Starting trace of "
                                    f"aperture {i+1} at pixel {start+1}")

                        # The coordinates are always returned as (x-coords, y-coords)
                        ref_coords, in_coords = tracing.trace_lines(
                            ext, axis=dispaxis, start=start, initial=[loc],
                            rwidth=None, cwidth=5, step=step, nsum=nsum,
                            max_missed=max_missed, initial_tolerance=None,
                            max_shift=max_shift, viewer=self.viewer if debug else None)
                        if ref_coords.size:
                            if all_ref_coords.size:
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
                        if in_coords.size:
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

                ext.APERTURE = make_aperture_table(aperture_models,
                                                   existing_table=aptable)

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
        xunits: str
            units of the x (wavelength/frequency) column
        yunits: str
            units of the data column

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
        xunits = None if params["wave_units"] is None else u.Unit(params["wave_units"])
        yunits = None if params["data_units"] is None else u.Unit(params["data_units"])

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

                output_frame = ext.wcs.output_frame
                xdata = (ext.wcs(range(ext.data.size)) *
                         (output_frame.unit[0] or u.nm))
                if xunits is not None and xunits != xdata.unit:
                    xdata = xdata.to(xunits)
                data_unit = u.Unit(ext.hdr.get("BUNIT"))
                ydata = ext.data * data_unit
                equivalencies = u.spectral_density(xdata)
                if yunits is not None:
                    try:
                        ydata = ydata.to(yunits, equivalencies=equivalencies)
                    except u.core.UnitConversionError:
                        try:
                            ydata = (ydata / (ad.exposure_time() * u.s)).to(
                                yunits, equivalencies=equivalencies)
                        except u.core.UnitConversionError:
                            log.warning(f"Cannot convert spectrum in {ad.filename}:"
                                        f"{ext.id} from {ydata.unit} to {yunits}")
                            yunits = data_unit
                else:
                    yunits = data_unit

                t = Table((xdata.value, ydata.value),
                          names=("wavelength", "data"),
                          units=(xdata.unit, ydata.unit))
                t.meta['comments'] = [f"Wavelength in {xdata.unit}, "
                                      f"Data in {ydata.unit}"]
                if write_dq:
                    t.add_column(ext.mask, name="dq")
                if write_var:
                    stddev = np.sqrt(ext.variance) * data_unit
                    try:
                        stddev = stddev.to(yunits, equivalencies=equivalencies)
                    except u.core.UnitConversionError:
                        stddev = (stddev / (ad.exposure_time() * u.s)).to(
                            yunits, equivalencies=equivalencies)
                    var = stddev * stddev
                    t.add_column(var.value, name="variance")
                    t["variance"].unit = var.unit
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

    def _get_arc_linelist(self, ext, waves=None):
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


    def _get_slit_edge_estimates(self, ad):
        """Return a list of pairs of slit edges in the given observation.

        Parameters
        ----------
        ad : AstroData instance
            The object to find the slit edges in.

        Returns
        -------
        tuple
            A 2-length tuple of lists of expected pixel postions for the left and
            right edges of slits.
        """
        # TODO: Currently this only handles longslit MDFs.
        exp_edges_1, exp_edges_2 = [], []

        # Get values from the MDF attached to the ad object.
        if ad.dispersion_axis()[0] == 2:
            center = ad.MDF['x_ccd'][0]
        elif ad.dispersion_axis()[0] == 1:
            center = ad.MDF['y_ccd'][0]
        half_slit_px = ad.MDF['slitlength_pixels'][0] / 2.

        # Calculate slit edges based on center and slit length.
        exp_edges_1.append(center - (half_slit_px))
        exp_edges_2.append(center + (half_slit_px))

        return exp_edges_1, exp_edges_2


    def _get_spectrophotometry(self, filename, in_vacuo=False):
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
        in_vacuo: bool/None
            are the wavelengths in the spectrophotometry file in vacuo?

        Returns
        -------
        Table:
            the spectrophotometric data, with columns 'WAVELENGTH_AIR',
            'WAVELENGTH_VACUUM', 'WIDTH', and 'FLUX'

        Raises
        ------
        FileNotFoundError: if file does not exist
        InconsistentTableError: if the file can't be read as ASCII
        """
        log = self.log

        # HST/calspec files have all sorts of UnitsWarnings because of
        # incorrect names like "ANGSTROMS" and "FLAM"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", u.UnitsWarning)
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

        wavecol = spec_table["WAVELENGTH"].quantity
        if in_vacuo is None:
            in_vacuo = min(wavecol) < 300 * u.nm

        if in_vacuo:
            spec_table["WAVELENGTH_VACUUM"] = spec_table["WAVELENGTH"]
            spec_table["WAVELENGTH_AIR"] = vac_to_air(wavecol)
        else:
            spec_table["WAVELENGTH_AIR"] = spec_table["WAVELENGTH"]
            spec_table["WAVELENGTH_VACUUM"] = air_to_vac(wavecol)
        del spec_table["WAVELENGTH"]

        # If we don't have a flux column, create one
        if not 'FLUX' in spec_table.colnames:
            # Use ".data" here to avoid "mag" being in the unit
            spec_table['FLUX'] = (10 ** (-0.4 * (spec_table['MAGNITUDE'].data + 48.6))
                                  * u.Unit("erg cm-2 s-1") / u.Hz)
        return spec_table

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0, slit_length=None,
                        slit_width=None, max_perpendicular_offset=None):
        """
        Checks whether the fields of view of two AD objects overlap
        sufficiently to be considerd part of a single ExposureGroup.
        This method, implemented at the Spect level, assumes that both AD
        objects have a single extension and, although it allows the length
        and width of the slit to be passed, these parameters will not be
        passed in normal usage. If not passed, the assumption is that the
        slit is as long as the spatial axis of the image, and the width is
        obtained from the descriptor. If either of these conditions is not
        met, this method should be defined in a subclass and, if desired,
        super() this method.

        Parameters
        ----------
        ad1: AstroData
            one of the input AD objects
        ad2: AstroData
            the other input AD object
        frac_FOV: float (0 < frac_FOV <= 1)
            fraction of the field of view for an overlap to be considered. If
            frac_FOV=1, *any* overlap is considered to be OK
        slit_length: float/None
            length of the slit (in arcsec)
        slit_width: float
            width of the slit (in arcsec)
        max_perpendicular_offset: float
            maximum allowable offset perpendicular to the slit

        Returns
        -------
        bool: do the fields overlap sufficiently?
        """
        if slit_length is None:
            slit_length = (ad1[0].shape[ad1[0].dispersion_axis()-1] *
                           ad1.pixel_scale())
        if slit_width is None:
            slit_width = ad1.slit_width()
        max_perpendicular_offset = max(max_perpendicular_offset or 0, slit_width)

        # I'm not sure where to put the abstraction here. This function in
        # gemini_tools calls one in astrotools, so maybe the gemini_tools
        # function should be brought here. We'll want to keep an eye on this
        # as things develop.
        dist_para, dist_perp = gt.offsets_relative_to_slit(ad1[0], ad2[0])
        return (abs(dist_para) <= frac_FOV * slit_length and
                abs(dist_perp) <= max_perpendicular_offset)


    def transferDistortionModel(self, adinputs=None, suffix=None, source=None):
        """
        This primitive transfers distortion_model from the AD(s) in another ("source")
        stream to the ADs in this stream if there was none, to match the WCS structure
        of processedArc. If the number of ADs in the source stream is larger than
        in the current stream, it's assumed that there was frame stacking done in the recipe,
        and the distortion_model gets transferred only if the AD's ORIGNAME keywords are matching.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        source: str
            name of stream containing ADs whose "distortion_model" you want
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        if source not in self.streams.keys():
            log.info(f"Stream {source} does not exist so nothing to transfer")
            return adinputs

        source_length = len(self.streams[source])
        source_files = self.streams[source]
        if not source_length > len(adinputs):
            # If number of files in the source stream is larger than in the current stream, assume
            # that there was frame stacking done in the recipe.
            # Do model copying only for the ads that have matching original filenames.
            source_files = []
            for ad1 in adinputs:
                orig_filename = ad1.phu.get('ORIGNAME')
                for ad2 in self.streams[source]:
                    if ad2.phu.get('ORIGNAME') == orig_filename:
                        source_files.append(ad2)
                        break
                else:
                    # Didn't find a match
                    log.warning(f"No matching frame found for "
                                f"{orig_filename} in the source stream")

        elif source_length < len(adinputs):
            log.warning("Incompatible stream lengths: "
                        f"{len(adinputs)} and {source_length}")
            return adinputs

        log.stdinfo(f"Transferring distortion model from stream '{source}'")

        # Copy distortion model from ad2 to ad1
        for ad1, ad2 in zip(*gt.make_lists(adinputs, source_files)):
            fail = False
            distortion_models = []
            for ext1, ext2 in zip(ad1, ad2):
                wcs1 = ext1.wcs
                wcs2 = ext2.wcs
                if 'distortion_corrected' in wcs1.available_frames:
                    log.warning(f"{ad1.filename}: already contains distortion model. "
                            " Continuing.")
                    fail = True
                    break
                try:
                    if 'distortion_corrected' not in wcs2.available_frames:
                        log.warning("Could not find a 'distortion_corrected' frame "
                            f"in {ad2.filename} extension {ext2.id} - "
                            "continuing")
                        fail = True
                        break
                except AttributeError:
                    fail = True
                    break
                else:
                    if 'rectified' not in wcs2.available_frames:
                        m_distcorr = wcs2.get_transform(wcs2.input_frame, 'distortion_corrected')
                    else:
                        m_distcorr = wcs2.get_transform("rectified", 'distortion_corrected')
                    distortion_models.append(m_distcorr)
            if not fail:
                for ext, dist in zip(ad1, distortion_models):
                    ext.wcs.insert_frame(ext.wcs.input_frame, dist, cf.Frame2D(name="distortion_corrected"))

                ad1.update_filename(suffix=suffix, strip=True)
        return adinputs

    def _uses_atran_linelist(self, cenwave, absorption):
        """
        Returns True if the observation can use ATRAN line list for the wavecal.
        By default ATRAN line lists are used in two cases:
            1) wavecal from sky absorption lines in object spectrum (in XJHK-bands);
            2) wavecal from sky emission lines in L- and M-band.
        In these cases (unless the user is providing her own linelist) we
        generate the line list on-the-fly from an ATRAN model with the parameters
        closest to the frame's observing conditions, and convolve it to the
        resolution of the observation.

        Parameters
        ----------
        cenwave: float
            central wavelength in um
        absorption: bool
            is wavecal done from absorption lines?
        """
        return True if (absorption is True) or (cenwave >= 2.8) else False

    def _get_atran_linelist(self, ext, config):
        """
        Returns default filename of the line list generated from an ATRAN
        model spectrum with model parameters matching the observing conditions
        and spectral characteristics of the frame (or those, specified by the user).
        Also returns a dictionary with the data necessary for the reference plot (so
        that we don't have to re-calculate the linelist for making the reference plot).

        Parameters
        ----------
        ext: single-slice AstroData
            the extension
        config: Config-like object containing parameters

        Returns
        -------
        atran_filename: str
            full path to the generated ATRAN line list
        refplot_dict: dict
            a dictionary containing data for the reference plot
        """
        lookup_dir = os.path.dirname(import_module('.__init__',
                                           self.inst_lookups).__file__)
        working_dir = os.getcwd()

        refplot_dict = None
        model_params = self._get_model_params(ext, config)

        atran_linelist = 'atran_linelist_{}_{:.0f}-{:.0f}_wv{:.0f}_r{:.0f}.dat'\
            .format(model_params["site"],model_params["start_wvl"],
                    model_params["end_wvl"],model_params["wv_content"]*1000,
                    model_params["resolution"])

        atran_filename = os.path.join(lookup_dir, atran_linelist)
        # If there is no pre-calculated ATRAN linelist in the default lookups directory,
        # create one and save it in the current working directory:
        if not os.path.exists(atran_filename):
            atran_filename = os.path.join(working_dir, atran_linelist)
            self.log.stdinfo(f"Generating a linelist from ATRAN synthetic spectrum")
            _, refplot_dict = self._make_atran_linelist(ext, filename=atran_filename,
                                                        model_params=model_params)
        self.log.stdinfo(f"Using linelist {atran_filename}")
        return wavecal.LineList(atran_filename), refplot_dict

    def _make_atran_linelist(self, ext, filename, model_params):
        """
        Generates line list from a convolved ATRAN spectrum: finds peaks,
        assings weights, divides spectrum into 10 wavelength bins,
        selects nlines//10 best lines within each bin, pinpoints the peaks.
        Saves the linelist to a file, and creates a disctionary with
        convolved spectrum and line list data to use later for the reference plot.

        Parameters
        ----------
        ext: single-slice AstroData
            the extension
        filename: str or None
            full name of the linelist to be saved on disk. If None, the linelist is not saved.
        model_params: dict
         A dictionary of observation parameters generated by _get_model_params() function.

        Returns
        -------
        atran_linelist: 1d array
            A list of ATRAN line wavelengths in air/vacuum (nm)
        refplot_dict: dict
            a dictionary containing data for the reference plot
        """

        atran_spec, sampling = self._get_convolved_atran(ext, model_params=model_params)

        inverse_atran_spec = 1 - atran_spec[:,1]
        fwhm = (model_params["cenwave"] / model_params["resolution"]) * (1 / sampling)
        peaks, properties = find_peaks(inverse_atran_spec, prominence=0.001, width=(None,5*fwhm))
        weights = properties["prominences"] / properties["widths"]
        peaks_with_weights = np.vstack([peaks,weights]).T

        def trim_peaks(peaks, npix, nbins, nlargest, sort=True):
            """
            Filters the peaks list, binning it over the range of the whole signal, preserving only the
            N-largest ones on each bin

            peaks: a 2-column array. The first column is pixel index, the second is signal value
            npix: total nunmber of pixels
            nbins: number of desired bins
            nlargest: number of largest peaks to extract from each bin

            Returns:
              An array of the same format
            """
            pixels_per_bin = npix // nbins
            padded_npix = npix
            if npix % nbins != 0:
                # Not evenly divisible. Adjust!
                pixels_per_bin += 1
                padded_npix = pixels_per_bin * nbins
            weights = np.zeros((padded_npix,), dtype=peaks.dtype)
            # Put the weights in their place and "bin" the weights array
            weights[peaks[:,0].astype(int)] = peaks[:,1]
            binned = weights.reshape((nbins, pixels_per_bin))
            indices = np.arange(padded_npix).reshape((nbins, pixels_per_bin))

            result = []
            for row, ind in zip(binned, indices):
                sorting = np.argpartition(row, -nlargest)
                sorted_row = row[sorting]
                nonzeros = sorted_row > 0
                trimmed_indices = ind[sorting][nonzeros]
                if len(trimmed_indices) > 0:
                    result.extend(zip(list(trimmed_indices[-nlargest:]), list(sorted_row[nonzeros][-nlargest:])))

            if sorted:
                return np.array(sorted(result), dtype=peaks.dtype)
            else:
                return np.array(result, dtype=peaks.dtype)

        # For the final line list select n // 10 peaks with largest weights
        # within each of 10 wavelenght bins.
        nbins = 10
        best_peaks = trim_peaks(peaks_with_weights, inverse_atran_spec.shape[0], nbins=nbins,
                                nlargest=model_params["nlines"]//nbins, sort=True)
        # Pinpoint peak positions
        final_peaks, peak_values = pinpoint_peaks(inverse_atran_spec, peaks=best_peaks[:,0],
                                                  halfwidth=1, keep_bad=True)

        atran_linelist = np.empty([len(final_peaks),3])
        # wavelenghts
        atran_linelist[:,0] = final_peaks
        # intensities
        atran_linelist[:,1] = peak_values
        # weights
        atran_linelist[:,2] = best_peaks[:,1]
        atran_linelist = atran_linelist[~np.isnan(atran_linelist).any(axis=1), :]

        atran_linelist[:,0] = atran_spec[[int(x) for x in atran_linelist[:,0]]][:,0] + \
                              [p%1 * sampling for p in atran_linelist[:,0]]
        atran_linelist = atran_linelist[np.argsort(atran_linelist[:,0])]

        if model_params["absorption"]:
            atran_linelist[:,1] = 1 - atran_linelist[:,1]

        header = (f"Sky emission line list: {model_params['start_wvl']:.0f}-{model_params['end_wvl']:.0f}nm   \n"
            "Generated from ATRAN synthetic spectrum (Lord, S. D., 1992, NASA Technical Memorandum 103957) \n"
            "Model parameters: \n"
            f"Obs altitude: {model_params['alt']}, Obs latitude: 39 degrees,\n"
            f"Water vapor overburden: {model_params['wv_content']*1000:.0f} microns, Number of atm. layers: 2,\n"
            "Zenith angle: 48 deg, Wavelength range: 1.0 - 6.0 microns, Smoothing R:0 \n"
            "units nanometer\n"
            "wavelengths IN VACUUM")
        if filename is not None:
            np.savetxt(filename, atran_linelist[:,0], fmt=['%.3f'], header = header)
        else:
            pass

        # Create data for reference plot using the convolved spectrum and the line
        # list generated here
        refplot_dict = self._make_refplot_data(ext=ext, refplot_spec=atran_spec,
                                refplot_linelist=atran_linelist[:,[0,1]], model_params=model_params)
        return atran_linelist[:,0], refplot_dict

    def _get_convolved_atran(self, ext, model_params):
        """
        Reads a section of an appropriate ATRAN model spectrum file,
        convolves the spectrum to the resolution of the observation.

        Parameters
        ----------
        ext: single-slice AstroData
            the extension
        model_params: dict
         A dictionary of observation parameters generated by _get_model_params() function.

        Returns
        -------
        spectrum : ndarray
            2d array of wavelengths (nm) and atmospheric transmission values
        sampling: float
            sampling (nm) of the ATRAN model spectrum
        """
        path = list(atran_models.__path__).pop()

        atran_model_filename = os.path.join(path,
                    'atran_{}_850-6000nm_wv{:.0f}_za48_r0.dat'
                                            .format(model_params["site"],
                                                    model_params["wv_content"]*1000))
        # sampling of the ATRAN model spectra (nm)
        sampling = 0.01
        # Starting wavelength of the ATRAN spectrum (nm)
        spec_start_wvl = 850
        start_row = int((model_params["start_wvl"] - spec_start_wvl) / sampling)
        nrows = int(model_params["spec_range"] / sampling)
        # Read the model spectrum within observation's spec. range
        atran_spec = np.loadtxt(atran_model_filename, skiprows=start_row,
                                      max_rows=nrows+1)
        # Smooth the atran spectrum with a Gaussian with a constant FWHM value,
        # where FWHM = cenwave / resolution
        fwhm = (model_params["cenwave"]/ model_params["resolution"]) * (1 / sampling)
        gauss_kernel = Gaussian1DKernel(fwhm / 2.35) # std = FWHM / 2.35
        atran_spec[:,1] = convolve(atran_spec[:,1], gauss_kernel,
                                     boundary='extend')

        return atran_spec, sampling


    def _show_refplot(self, ext):
        """
        Reference plots are implemented for the wavecal from sky lines
        """
        return False if 'ARC' in ext.tags else True

    def _make_refplot_data(self, ext, refplot_linelist, config=None, model_params=None,
                           refplot_spec=None, refplot_name=None, refplot_y_axis_label=None):
        """
        Generate data for the reference plot (reference spectrum, reference plot name, and
        the label for the y-axis), depending on the type of spectrum used for wavelength calibration,
        using the supplied line list.

        Parameters
        ----------
        ext: single-slice AstroData
            the extension
        model_params: dict or None
         A dictionary of observation parameters generated by _get_model_params() function.
         Either model_params or config must be not None
        config: Config-like object containing parameters or None
            Either model_params or config must be not None
        refplot_linelist: LineList object or a 2-column nd-array with wavelengths
                (nm) and line intensities
        refplot_spec: 2d-array or None
            two-column nd.array containing reference spectrum wavelengths (nm) and intensities
        refplot_name: str or None
            reference plot name string
        refplot_y_axis_label: str or None
            reference plot y-axis label string

        Returns
        -------
        dict : all the information needed to construct reference spectrum plot:
        "refplot_spec" : two-column nd.array containing reference spectrum wavelengths
                        (nm) and intensities
        "refplot_linelist" : two-column nd.array containing line wavelengths (nm) and
                            intensities
        "refplot_name" : reference plot name string
        "refplot_y_axis_label" : reference plot y-axis label string
        """

        if config is None and model_params is None:
            raise ValueError('Either "config" or "model_params" must be not None')
        if model_params is None:
            model_params = self._get_model_params(ext, config)
        resolution = model_params["resolution"]
        absorption = model_params["absorption"]

        if self._show_refplot(ext):
            # Use ATRAN models for generating reference plots
            # for wavecal from sky absorption lines in science spectrum,
            # and for wavecal from sky emission lines in L- and M-bands
            if self._uses_atran_linelist(absorption=absorption, cenwave=ext.central_wavelength(asMicrometers=True)):
                if refplot_spec is None:
                    refplot_spec, _ = self._get_convolved_atran(ext, model_params)
                if refplot_y_axis_label is None:
                    if absorption:
                        refplot_y_axis_label = "Atmospheric transmission"
                    else:
                        refplot_y_axis_label = "Inverse atm. transmission"
                        refplot_spec[:,1] = 1 - refplot_spec[:,1]
                if refplot_name is None:
                    refplot_name = 'ATRAN spectrum (Alt={}, WV={:.1f}mm, AM=1.5, R={:.0f})'\
                        .format(model_params["alt"],model_params["wv_content"],resolution)
            else:
                # Use a set of pre-calculated synthetic spectra for the reference plots
                # for wavecal from the OH emission sky lines
                if refplot_spec is None:
                    path = list(oh_synthetic_spectra.__path__).pop()

                    if resolution >= 15000:
                        R = 20000
                    elif (7500 <= resolution < 15000):
                        R = 10000
                    elif (3000 <= resolution < 7500):
                        R = 5000
                    elif (1000 <= resolution < 3000):
                        R = 1500
                    elif (resolution < 1000):
                        R = 500
                    oh_spec_filename = os.path.join(path, "oh_conv_{}_XJHK.dat".format(R))
                    refplot_spec = np.loadtxt(oh_spec_filename, usecols=(0,1))

                    dispersion_axis = 2 - ext.dispersion_axis()
                    # The next bit estimates the spectral range that wasn't masked
                    # by the default illumination mask (that mask off second order
                    # in some GNIRS and F2 modes).
                    mask = at.transpose_if_needed(ext.mask, transpose=(dispersion_axis == 0))[0]
                    center = ext.shape[dispersion_axis]//2
                    center_col = mask[center][:]
                    new = np.ones(center_col.shape)
                    new[center_col >= 64] = 0
                    ind = np.nonzero(new)
                    first_non_zero_index = ind[0][0]
                    last_non_zero_index = ind[0][-1]
                    # Number of illuminated pixels
                    npix = last_non_zero_index - first_non_zero_index
                    spec_range = 2 * self._get_cenwave_accuracy(ext) +\
                                 abs(ext.dispersion(asNanometers=True)) * npix
                    center_shift = (center - (npix//2 + first_non_zero_index)) * ext.dispersion(asNanometers=True)
                    start_wvl = model_params["cenwave"] - (0.5 * spec_range) - center_shift
                    end_wvl = start_wvl + spec_range

                    refplot_spec = refplot_spec[np.logical_and(refplot_spec[:,0] >= start_wvl,
                                                             refplot_spec[:,0] <= end_wvl)]
                    if refplot_y_axis_label is None:
                        refplot_y_axis_label = "Intensity"
                    if refplot_name is None:
                        refplot_name = 'Synthetic spectrum of night-sky OH emission (R={:.0f})'.format(R)

        if isinstance(refplot_linelist, wavecal.LineList):
            # If the provided linelist is a LineList-type object, then it wasn't
            # generated on-the-fly and its format has to be adjusted.
            # For the refplot use only line wavelengths, and determine
            # reference spectrum intensities at the positions of lines in the linelist
            # (this is needed to determine the line label positions).
            line_intens = []
            line_wvls = refplot_linelist.wavelengths(in_vacuo=model_params["in_vacuo"], units="nm")
            for n, line in enumerate(line_wvls):
                subtracted = refplot_spec[:,0] - line
                min_index = np.argmin(np.abs(subtracted))
                line_intens.append(refplot_spec[min_index,1])
            refplot_linelist = np.array([line_wvls, line_intens]).T
            refplot_linelist = refplot_linelist[np.logical_and(refplot_linelist[:,0] >= start_wvl,
                                                               refplot_linelist[:,0] <= end_wvl)]

        return {"refplot_spec": refplot_spec, "refplot_linelist": refplot_linelist,
                "refplot_name": refplot_name, "refplot_y_axis_label": refplot_y_axis_label}


    def _get_model_params(self, ext, config):
        """
        Get the observation parameters needed to select the appropriate
        ATRAN (or other) model.

        Parameters
        ----------
        ext: single-slice AstroData
            the extension
        config: Config-like object containing parameters

        Returns
        ----------
        dict : all the parameters needed for ATRAN model selection and line
        list generation:
        "resolution" : resolution to which ATRAN model spectrum should be convolved (int)
        "cenwave" : *actual* central wavelength of the observation (float)
        "in_vacuo" : config["in_vacuo"]
        "nlines" : maximum number of highest-weight lines in the ATRAN spectrum to
            be included in the generated linelist (int)
        "absorption" : are absorption features used for wavecal? (bool)
        wv_content: float
            WV band constraints (in mm of precipitable H2O at zenith)
        site: str
            observation site
        alt: str
            observatory altitude
        start_wvl: float
             start wavelength of the spectrum (nm) with a small buffer
        end_wvl: float
            start wavelength of the spectrum (nm) with a small buffer
        spec_range: float
            spectral range with buffer (nm)
        """

        if config.get("resolution", None) is None:
            resolution = round(self._get_resolution(ext),  -1)
        else:
            resolution = config["resolution"]

        if config["central_wavelength"] is None:
            cenwave = self._get_actual_cenwave(ext, asNanometers=True)
        else:
            cenwave = config["central_wavelength"]

        in_vacuo = config["in_vacuo"]
        num_atran_lines = config.get("num_atran_lines", 50)
        absorption = config.get("absorption", False)

        dispaxis = 2 - ext.dispersion_axis()  # python sense
        npix = ext.shape[dispaxis]
        dcenwave = self._get_cenwave_accuracy(ext)

        spec_range = 2 * dcenwave + abs(ext.dispersion(asNanometers=True)) * npix
        start_wvl = cenwave - (0.5 * spec_range)
        end_wvl = start_wvl + spec_range
        observatory = ext.phu['OBSERVAT']
        param_wv_band = config.get("wv_band", "header")
        if param_wv_band == "header":
            wv_band = ext.raw_wv()
        else:
            wv_band = int(param_wv_band)

        if wv_band is None:
            req_wv = ext.requested_wv()
            self.log.stdinfo(f"Unknown RAWWV for this observation; \n"
                f" using the constraint value for the requested WV band: '{req_wv}'-percentile")
            wv_band = req_wv

        if wv_band == 100:
            # a WV value to use for the case of RAWWV='Any'
            if observatory == 'Gemini-North':
                wv_content = 5.
            elif observatory == 'Gemini-South':
                wv_content = 10.
        else:
            wv_content = qa_constraints.wvBands.get(observatory).get(str(wv_band))

        if observatory == 'Gemini-North':
            site = 'mk'
            alt = "13825ft"
        elif observatory == 'Gemini-South':
            site = 'cp'
            alt = "8980ft"
        return {"site": site, "alt": alt, "start_wvl": start_wvl, "end_wvl": end_wvl,
                "spec_range": spec_range, "wv_content": wv_content,
                "resolution": resolution, "cenwave": cenwave, "in_vacuo": in_vacuo,
                "absorption": absorption, "nlines": num_atran_lines}


    def _get_resolution(self, ext):
        # Estimated resolwing power of a spectrum at it's actual central wavelenght,
        # assuming that it depends on 1/(slit width). This is true only for GNIRS (and GMOS?),
        # for other instruments there should be an instrument-specific implementation.
        resolution_1pix_slit = self._get_actual_cenwave(ext) / ext.dispersion()
        slit_width_pix = ext.slit_width()/ext.pixel_scale()
        return abs(resolution_1pix_slit // slit_width_pix)

    def _get_actual_cenwave(self, ext, asMicrometers=False,
                            asNanometers=False, asAngstroms=False):
        """
        For some instruments (NIRI, F2) wavelenght at the central pixel
        can differ significantly from the descriptor value.

        Parameters
        ----------
        asMicrometers : bool
            If True, return the wavelength in microns
        asNanometers : bool
            If True, return the wavelength in nanometers
        asAngstroms : bool
            If True, return the wavelength in Angstroms

        Returns
        -------
        float
            Actual cenral wavelenght
        """
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        output_units = "meters" # By default
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        cenwave = ext.central_wavelength()
        actual_cenwave = gmu.convert_units('meters', cenwave, output_units)

        return actual_cenwave

    def _get_cenwave_accuracy(self, ext):
        # TODO: remove this
        dispaxis = 2 - ext.dispersion_axis()
        npix = ext.shape[dispaxis]
        w1, w2 = am.get_named_submodel(ext.wcs.forward_transform, "WAVE").copy()([0, npix-1])
        m_wave = models.Chebyshev1D(degree=1, c0=0.5*(w1+w2), c1=0.5*(w2-w1))
        self._apply_wavelength_model_bounds(m_wave, ext)
        return 0.5 * abs(np.diff(m_wave.c0.bounds)[0])

    def _apply_wavelength_model_bounds(self, model=None, ext=None):
        # Apply bounds to an astropy.modeling.models.Chebyshev1D to indicate
        # the range of parameter space to explore
        for i, (pname, pvalue) in enumerate(zip(model.param_names, model.parameters)):
            if i == 0:  # central wavelength
                prange = 10
            elif i == 1:  # half the wavelength extent (~dispersion)
                prange = 0.05 * abs(pvalue)
            else:  # higher-order terms
                prange = 20
            getattr(model, pname).bounds = (pvalue - prange, pvalue + prange)

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


def create_distortion_model(m_init, transform_axis, in_coords, ref_coords,
                            fixed_linear):
    """
    This helper function creates a distortion model given an input model,
    input and reference coordinates,

    Parameters
    ----------
    m_init : astropy.modeling.models.Chebyshev2D
        A blank initial model covering the region of interest.
    transform_axis : int (0 or 1)
        The axis to transform the image along; alternatively, the axis along
        which tracing.trace_lines() performed the trace which produced the
        `in_coords` and `ref_coords` used. (This is in the Python sense, 0 or 1,
        not the value returned by the dispaxis() descriptor).
    in_coords, ref_coords : iterable
        Lists of coordinates for the model fitting.
    fixed_linear : bool
        Fix the linear term? (For instance, if creating a distortion model for
        rectifying a slit when only one edge of a flat is traceable.)

    Returns
    -------
    model : astropy.modeling.models.Mapping
        The output model mapping.
    m_final, m_inverse : astropy.modeling.fitting.FittingWithOutlierRemoval
        Models describing the forward and inverse transformations used in
        `model`.
    """
    # Rather than fit to the reference coords, fit to the
    # *shift* we want in the spectral direction and then add a
    # linear term which will produce the desired model. This
    # allows us to use spectral_order=0 if there's only a single
    # traceable line.
    if transform_axis == 0:
        domain_start, domain_end = m_init.x_domain
        param_names = [f'c1_{i}' for i in range(m_init.y_degree + 1)]
    else:
        domain_start, domain_end = m_init.y_domain
        param_names = [f'c{i}_1' for i in range(m_init.x_degree + 1)]
    domain_centre = 0.5 * (domain_start + domain_end)
    if fixed_linear:
        for pn in param_names:
            getattr(m_init, pn).fixed = True
    shifts = ref_coords[transform_axis] - in_coords[transform_axis]

    # Find model to transform actual (x,y) locations to the
    # value of the reference pixel along the dispersion axis
    fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                               sigma_clip, sigma=3)
    m_final, _ = fit_it(m_init, *in_coords, shifts)
    m_inverse, _ = fit_it(m_init, *ref_coords, -shifts)
    for m in (m_final, m_inverse):
        m.c0_0 += domain_centre
        # param_names[0] will be 'c1_0' (tranforms_axis == 0) or 'c0_1'.
        getattr(m, param_names[0]).value += domain_end - domain_centre

    if transform_axis == 0:
        model = models.Mapping((0, 1, 1)) | (m_final & models.Identity(1))
        model.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
    else:
        model = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_final)
        model.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)

    return model, m_final, m_inverse


def make_aperture_table(apmodels, existing_table=None, limits=None):
    """
    Create a new APERTURE table from a list of aperture trace models. These
    can either be updated models to apply to an existing table, or a new
    table (in which case the limits must be provided).

    Parameters
    ----------
    apmodels: list of Chebyshev1D models
        the models defining the apertures
    existing_table: Table/None
        an existing Table (if updating apertures)
    limits: list of 2-tuples/None
        aperture limits (if creating a new table)

    Returns
    -------
    Table: the new APERTURE table
    """
    all_tables = []
    length = len(limits if existing_table is None else existing_table)
    if len(apmodels) != length:
        raise ValueError(f"Mismatch between apmodels length ({len(apmodels)})"
                         f" and iterator length ({length})")

    iterator = iter(limits if existing_table is None else existing_table)
    for apmodel, item in zip(apmodels, iterator):
        aptable = am.model_to_table(apmodel)
        if existing_table is None:
            aptable["aper_lower"] = item[0]
            aptable["aper_upper"] = item[1]
        else:
            aptable["aper_lower"] = item["aper_lower"]
            aptable["aper_upper"] = item["aper_upper"]
        all_tables.append(aptable)

    # If the traces have different orders, there will be missing
    # values that vstack will mask, so we have to set those to zero
    new_aptable = vstack(all_tables, metadata_conflicts="silent")
    if existing_table is None:
        new_aptable["number"] = np.arange(len(new_aptable), dtype=np.int32) + 1
    else:
        new_aptable["number"] = existing_table["number"]
    colnames = new_aptable.colnames
    new_col_order = (["number"] + sorted(c for c in colnames
                                         if c.startswith("c")) +
                     ["aper_lower", "aper_upper"])
    for col in colnames:
        if isinstance(new_aptable[col], MaskedColumn):
            new_aptable[col] = new_aptable[col].filled(fill_value=0)
    return new_aptable[new_col_order]



def QESpline(coeffs, waves, data, weights, boundaries, order):
    """
    Fits a cubic spline to data, allowing scaling renormalizations of
    contiguous subsets of the data.

    Parameters
    ----------
    coeffs : array_like
        Scaling factors for CCDs 2+.
    waves : array
        Wavelengths
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
    spline = am.UnivariateSplineWithOutlierRemoval(waves, scaled_data,
                                                   order=order, w=scaled_weights, niter=1, grow=0)
    result = np.ma.masked_where(spline.mask, np.square((spline.data - scaled_data) *
                                                       scaled_weights)).sum() / (~spline.mask).sum()
    return result


def resample_spec_table(spec_table, resample_interval):
    # If we need to do this, we probably have ground-based
    # spectrophotometry, so interpolate in WAVELENGTH_AIR
    waves = spec_table["WAVELENGTH_AIR"].to(u.nm).value
    new_waves_air = np.arange(min(waves), max(waves + 0.001),
                              resample_interval)
    new_waves_vac = air_to_vac(new_waves_air * u.nm)
    new_fluxes = np.interp(new_waves_air, waves,
                           spec_table["FLUX"]) * spec_table["FLUX"].unit
    width_unit = spec_table["WIDTH"].unit
    new_width = (resample_interval * u.nm).to(width_unit).value
    new_widths = np.array([spec_table["WIDTH"][np.argmin(abs(waves-w))]
                           if np.any(abs(waves-w)<1e-6) else new_width
                           for w in new_waves_air]) * width_unit
    t = Table([new_waves_air * u.nm, new_waves_vac, new_fluxes, new_widths],
              names=["WAVELENGTH_AIR", "WAVELENGTH_VAC", "FLUX", "WIDTH"])
    return t


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


def plot_cosmics(ext, objfit, skyfit, crmask, axes=None):
    from astropy.visualization import ZScaleInterval, imshow_norm

    if axes is None:
        fig, axes = plt.subplots(1, 5, figsize=(15, 5*2),
                                 sharex=True,
                                 sharey=True,
                                 tight_layout=True)
    imgs = (ext.data, objfit, skyfit,
            ext.data - (objfit + skyfit), crmask)
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

    if axes is None:
        plt.show()
        plt.close(fig)
