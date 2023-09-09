#
#                                                                  gemini_python
#
#                                                      primitives_ghost_spect.py
# ------------------------------------------------------------------------------
import os
import numpy as np
import math
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
import astropy.coordinates as astrocoord
from astropy.time import Time
from astropy.io import fits
from astropy.io.ascii.core import InconsistentTableError
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
from astropy.modeling import fitting, models
from astropy.modeling.tabular import Tabular1D
from scipy import interpolate
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from pysynphot import observation, spectrum
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from importlib import import_module

import astrodata

from geminidr.core.primitives_spect import Spect
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library import tracing, astrotools as at
from gempy.gemini import gemini_tools as gt

from .polyfit import GhostArm, Extractor, SlitView
from .polyfit.polyspect import WaveModel

from .primitives_ghost import GHOST, filename_updater

from . import parameters_ghost_spect

from .lookups import polyfit_lookup, line_list, keyword_comments, targetn_dict

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------

GEMINI_SOUTH_LOC = astrocoord.EarthLocation.from_geodetic((-70, 44, 12.096),
                                                          (-30, 14, 26.700),
                                                          height=2722.,
                                                          ellipsoid='WGS84')
BAD_FLAT_FLAG = 16

# FIXME: This should go somewhere else, but where?
from scipy.ndimage import median_filter
def convolve_with_mask(data, mask, rectangle_width = (100,20)):
    """Helper function to convolve a masked array with a uniform rectangle after median
    filtering to remove cosmic rays.
    """
    #Create our rectangular function
    rectangle_function = np.zeros_like(data)
    rectangle_function[:rectangle_width[0], :rectangle_width[1]] = 1.0
    rectangle_function = np.roll(rectangle_function, int(-rectangle_width[
        0] / 2), axis=0)
    rectangle_function = np.roll(rectangle_function, int(-rectangle_width[1]/2),
                                 axis=1)
    rectangle_fft = np.fft.rfft2(rectangle_function)

    #Median filter in case of cosmic rays
    filt_data = median_filter(data,3)

    #Now convolve. The mask is never set to exactly zero in order to avoid divide
    #by zero errors outside the mask.
    convolved_data = np.fft.irfft2(np.fft.rfft2(filt_data * (mask + 1e-4))*rectangle_fft)
    convolved_data /= np.fft.irfft2(np.fft.rfft2(mask + 1e-4)*rectangle_fft)
    return convolved_data


@parameter_override
class GHOSTSpect(GHOST):
    """
    Primitive class for processing GHOST science data.

    This class contains the primitives necessary for processing GHOST science
    data, as well as all related calibration files from the main spectrograph
    cameras. Slit viewer images are processed with another primitive class
    (:class:`ghostdr.ghost.primitives_ghost_slit.GHOSTSlit`).
    """

    """Applicable tagset"""
    tagset = set(["GEMINI", "GHOST"])  # NOT SPECT because of bias/dark

    def __init__(self, adinputs, **kwargs):
        super(GHOSTSpect, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_ghost_spect)
        self.keyword_comments.update(keyword_comments.keyword_comments)

    def addDQ(self, adinputs=None, **params):
        """
        Quick'n'dirty addition of known bad pixels to FLATs only
        """
        log = self.log
        adinputs = super().addDQ(adinputs, **params)
        for ad in adinputs:
            if 'FLAT' in ad.tags:
                xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
                arm = ad.arm()
                ut_date = ad.ut_date().strftime("%Y-%m-%d")
                bpm_module = import_module(f'.BPM.badpix_{arm}', self.inst_lookups)
                try:
                    regions = [v for k, v in sorted(bpm_module.bpm_dict.items())
                               if k < ut_date][-1]
                except IndexError:
                    log.warning(f"Cannot find BPM information for {ad.filename}")
                    continue
                log.stdinfo(f"Adding bad pixels to mask of {ad.filename}")
                # TODO: rewrite with Section and contains, etc.
                for (x1, x2, y1, y2) in regions:
                    for i, (ext, datsec, detsec) in enumerate(
                            zip(ad, ad.data_section(), ad.detector_section())):
                        ix1 = (max(x1, detsec.x1) - detsec.x1) // xbin
                        ix2 = (min(x2, detsec.x2-1) - detsec.x1) // xbin + 1
                        iy1 = (max(y1, detsec.y1) - detsec.y1) // ybin
                        iy2 = (min(y2, detsec.y2-1) - detsec.y1) // ybin + 1
                        if 0 <= ix1 < ix2 and 0 <= iy1 < iy2:
                            ext.mask[iy1+datsec.y1:iy2+datsec.y1,
                            ix1+datsec.x1:ix2+datsec.x1] |= DQ.bad_pixel

        return adinputs

    def addWavelengthSolution(self, adinputs=None, **params):
        """
        Compute and append a wavelength solution for the data.

        The GHOST instrument is designed to be very stable over a long period
        of time, so it is not strictly necessary to take arcs for every
        observation. The alternative is use the arcs taken most recently
        before and after the observation of interest, and compute an
        average of their wavelength solutions.

        The average is weighted by
        the inverse of the time between each arc observation and science
        observation. E.g., if the 'before' arc is taken 12 days before the
        science observation, and the 'after' arc is taken 3 days after the
        science observation, then the 'after' arc will have a weight of 80%
        in the final wavelength solution (12/15), and the 'before' arc 20%
        (3/15).

        In the event that either a 'before' arc can't be found but an 'after'
        arc can, or vice versa, the wavelength solution from the arc that was
        found will be applied as-is. If neither a 'before' nor 'after' arc can
        be found, an IOError will be raised.

        It is possible to explicitly pass which arc files to use as
        the ``arc`` parameter. This should be a list of two-tuples, with each
        tuple being of the form
        ``('before_arc_filepath', 'after_arc_filepath')``. This list must be
        the same length as the list of ``adinputs``, with a one-to-one
        correspondence between the two lists.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        arc: list of two-tuples
            A list of two-tuples, with each tuple corresponding to an element of
            the ``adinputs`` list. Within each tuple, the two elements are the
            designated 'before' and 'after' arc for that observation.
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # No attempt to check if this primitive has already been run -
        # new arcs may be available which we wish to apply. Any old WAVL
        # extensions will simply be removed.

        # CJS: Heavily edited because of the new AD way
        # Get processed slits, slitFlats, and flats (for xmod)
        # slits and slitFlats may be provided as parameters
        # arc_list = params["arcs"]
        arc_before_file = params["arc_before"]
        arc_after_file = params["arc_after"]
        # if arc_list is None:
        #     # CJS: This populates the calibrations cache (dictionary) with
        #     # "processed_slit" filenames for each input AD
        #     self.getProcessedArc(adinputs)
        #     # This then gets those filenames
        #     arc_list = [self._get_cal(ad, 'processed_arc')
        #                  for ad in adinputs]
        #     log.stdinfo(arc_list)

        # for ad, arcs in zip(
        #         *gt.make_lists(adinputs, arc_list, force_ad=True)):
        for i, ad in enumerate(adinputs):

            found_arcs = False

            #if arc_list:
            #    try:
            #        arc_before, arc_after = arc_list[i]
            #        found_arcs = True
            #    except (TypeError, ValueError):
            #        pass

            if arc_before_file or arc_after_file:
                arc_before = arc_before_file
                arc_after = arc_after_file
                found_arcs = True

            # self.getProcessedArc(ad, howmany=2)
            # if not found_arcs:
            #     try:
            #         arcs_calib = self._get_cal(ad, 'processed_arc', )
            #         log.stdinfo('Found following arcs: {}'.format(
            #             ', '.join([_ for _ in arcs_calib])
            #         ))
            #         arc_before, arc_after = self._get_cal(ad, 'processed_arc',)
            #     except (TypeError, ValueError):
            #         # Triggers if only one arc, or more than two
            #         arc_before = self._get_cal(ad, 'processed_arc',)[0]
            #         arc_after = None

            if not found_arcs:
                # Fetch the arc_before and arc_after in sequence
                arc_before = self._request_bracket_arc(ad, before=True)
                arc_after = self._request_bracket_arc(ad, before=False)

            if arc_before is None and arc_after is None:
                raise IOError('No valid arcs found for {}'.format(ad.filename))

            log.stdinfo('Arcs for {}: \n'
                        '   before: {}\n'
                        '    after: {}'.format(ad.filename,
                                               arc_before, arc_after))

            # Stand up a GhostArm instance for this ad
            gs = GhostArm(arm=ad.arm(), mode=ad.res_mode(),
                          detector_x_bin=ad.detector_x_bin(),
                          detector_y_bin=ad.detector_y_bin())

            if arc_before is None:
                # arc = arc_after
                arc_after = astrodata.open(arc_after)
                wfit = gs.evaluate_poly(arc_after[0].WFIT)
                ad.phu.set('ARCIM_A', os.path.abspath(arc_after.path),
                           "'After' arc image")
            elif arc_after is None:
                # arc = arc_before
                arc_before = astrodata.open(arc_before)
                wfit = gs.evaluate_poly(arc_before[0].WFIT)
                ad.phu.set('ARCIM_B', os.path.abspath(arc_before.path),
                           "'Before' arc image")
            else:
                # Need to weighted-average the wavelength fits from the arcs
                # Determine the weights (basically, the inverse time between
                # the observation and the arc)
                arc_after = astrodata.open(arc_after)
                arc_before = astrodata.open(arc_before)
                wfit_b = gs.evaluate_poly(arc_before[0].WFIT)
                wfit_a = gs.evaluate_poly(arc_after[0].WFIT)
                weight_b = np.abs((arc_before.ut_datetime() -
                                   ad.ut_datetime()).total_seconds())
                weight_a = np.abs((arc_after.ut_datetime() -
                                   ad.ut_datetime()).total_seconds())
                weight_a, weight_b = 1. / weight_a, 1 / weight_b
                log.stdinfo('Cominbing wavelength solutions with weights '
                            '%.3f, %.3f' %
                            (weight_a / (weight_a + weight_b),
                             weight_b / (weight_a + weight_b),
                             ))
                # Compute weighted mean fit
                wfit = wfit_a * weight_a + wfit_b * weight_b
                wfit /= (weight_a + weight_b)
                ad.phu.set('ARCIM_A', os.path.abspath(arc_after.path),
                           self.keyword_comments['ARCIM_A'])
                ad.phu.set('ARCIM_B', os.path.abspath(arc_before.path),
                           self.keyword_comments['ARCIM_B'])
                ad.phu.set('ARCWT_A', weight_a,
                           self.keyword_comments['ARCWT_A'])
                ad.phu.set('ARCWT_B', weight_b,
                           self.keyword_comments['ARCWT_B'])

            # rebin the wavelength fit to match the rest of the extensions
            for _ in range(int(math.log(ad.detector_x_bin(), 2))):
                wfit = wfit[:, ::2] + wfit[:, 1::2]
                wfit /= 2.0

            for ext in ad:
                ext.WAVL = wfit

            # FIXME Wavelength unit needs to be in output ad

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def applyFlatBPM(self, adinputs=None, **params):
        """
        Find the flat relevant to the file(s) being processed, and merge the
        flat's BPM into the target file's.

        GHOST does not use flat subtraction in the traditional sense; instead,
        the extracted flat profile is subtracted from the extracted object
        profile. This means that the BPM from the flat needs to be applied to
        the object file before profile extraction, and hence well before actual
        flat correction is performed.

        The BPM flat is applied by ``bitwise_or`` combining it into the main
        adinput(s) BPM.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str/None
            Name (full path) of the flatfield to use. If None, try:
        flatstream: str/None
            Name of the stream containing the flatfield as the first
            item in the stream. If None, the calibration service is used
        write_result: bool
            Denotes whether or not to write out the result of profile
            extraction to disk. This is useful for both debugging, and data
            quality assurance.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # No attempt to check if this primitive has already been run -
        # re-applying a flat BPM should have no adverse effects, and the
        # primitive simply skips if no flat is found.

        # CJS: extractProfile() contains comments explaining what's going on here
        flat_list = params["flat"]
        flat_stream = params["flat_stream"]
        if flat_list is None:
            if flat_stream is not None:
                flat_list = self.streams[flat_stream][0]
            else:
                self.getProcessedFlat(adinputs)
                flat_list = [self._get_cal(ad, 'processed_flat')
                            for ad in adinputs]

        for ad, flat in zip(*gt.make_lists(adinputs, flat_list, force_ad=True)):
            if flat is None:
                log.warning("No flat identified/provided for {} - "
                            "skipping".format(ad.filename))
                continue

            # Re-bin the flat if necessary
            # We only need the mask, but it's best to use the full rebin
            # helper function in case the mask rebin code needs to change
            if flat.detector_x_bin() != ad.detector_x_bin(
            ) or flat.detector_y_bin() != ad.detector_y_bin():
                xb = ad.detector_x_bin()
                yb = ad.detector_y_bin()
                flat = self._rebin_ghost_ad(flat, xb, yb)
                # Re-name the flat so we don't blow away the old one on save
                flat_filename_orig = flat.filename
                flat.filename = filename_updater(flat,
                                                 suffix='_rebin%dx%d' %
                                                        (xb, yb,),
                                                 strip=True)
                flat.write(overwrite=True)

            # CJS: Edited here to require that the science and flat frames'
            # extensions are the same shape. The original code would no-op
            # with a warning for each pair that didn't, but I don't see how
            # this would happen in normal operations. The clip_auxiliary_data()
            # function in gemini_tools may be an option here.
            try:
                gt.check_inputs_match(adinput1=ad, adinput2=flat,
                                      check_filter=False)
            except ValueError:
                log.warning("Input mismatch between flat and {} - "
                            "skipping".format(ad.filename))
                continue

            # We don't want to copy over non-linear/saturated pixels from
            # the flat's DQ since these will be from CR hits and will have
            # been handled by the median combining
            for ext, flat_ext in zip(ad, flat):
                if ext.mask is None:
                    ext.mask = flat_ext.mask & DQ.not_signal
                else:
                    ext.mask |= flat_ext.mask & DQ.not_signal

            ad.phu.set('FLATBPM', os.path.abspath(flat.path),
                       self.keyword_comments['FLATBPM'])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            if params["write_result"]:
                ad.phu.set('PROCIMG', os.path.abspath(ad.path),
                           keyword_comments.keyword_comments['PROCIMG'])
                ad.write(overwrite=True)

        return adinputs

    def barycentricCorrect(self, adinputs=None, **params):
        """
        Perform barycentric correction of the wavelength extension in the input
        files.

        Barycentric correction is performed by multiplying the wavelength
        (``.WAVL``) data extension by a correction factor. This factor can be
        supplied manually, or can be left to be calculated based on the
        headers in the AstroData input.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        correction_factor: float
            Barycentric correction factor to be applied. Defaults to None, at
            which point a computed value will be applied. The computed value
            is based on the recorded position of the Gemini South observatory.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        velocity = params["velocity"]

        if velocity == 0.0:
            log.stdinfo("A radial velocity of 0.0 has been provided - no "
                        "barycentric correction will be applied")
            return adinputs

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by barycentricCorrect".
                            format(ad.filename))
                continue

            if not hasattr(ad[0], 'WAVL'):
                log.warning("No changes will be made to {}, since it contains "
                            "no wavelength information".
                            format(ad.filename))
                continue

            # Get or compute the correction factor
            if velocity is None:
                ra, dec = ad.ra(), ad.dec()
                if ra is None or dec is None:
                    print("Unable to compute barycentric correction for "
                          f"{ad.filename} (no sky pos data) - skipping")
                    rv = None
                else:
                    self.log.stdinfo(f"Computing SkyCoord for {ra}, {dec}")
                    sc = astrocoord.SkyCoord(ra, dec, unit=(u.deg, u.deg,))

                    # Compute central time of observation
                    dt_midp = ad.ut_datetime() + timedelta(
                        seconds=ad.exposure_time() / 2.0)
                    dt_midp = Time(dt_midp)
                    bjd = dt_midp.jd1 + dt_midp.jd2

                    # Vanilla AstroPy Implementation
                    rv = sc.radial_velocity_correction(
                        'barycentric', obstime=dt_midp,
                        location=GEMINI_SOUTH_LOC).to(u.km / u.s)
            else:
                rv = velocity * u.km / u.s
                bjd = None

            if rv is not None:
                log.stdinfo("Applying radial velocity correction of "
                            f"{rv.value} km/s to {ad.filename}")
                cf = float(1 + rv / const.c)  # remove u.dimensionless_unscaled
                for ext in ad:
                    ext.WAVL *= cf

                # Only one correction per AD right now
                ad.hdr['BERV'] = (rv.value, "Barycentric correction applied (km s-1)")
                if bjd is not None:
                    ad.hdr['BJD'] = (bjd, "Barycentric Julian date")

                # Timestamp and update filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def clipSigmaBPM(self, adinputs=None, **params):
        """
        Perform a sigma-clipping on the input data frame.

        This is a primitive wrapper for the :func:`astropy.stats.sigma_clip`
        method. The ``sigma`` and ``iters`` parameters are passed through to the
        corresponding keyword arguments.

        Parameters
        ----------
        sigma: float/None
            The sigma value to be used for clipping.
        bpm_value: int/None
            The integer value to be applied to the data BPM where the sigma
            threshold is exceeded. Defaults to 1 (which is the generic bad
            pixel flag). Note that the final output BPM is made using a
            bitwise_or operation.
        iters : int/None
            Number of sigma clipping iterations to perform. Default is None,
            which will continue sigma clipping until no further points are
            masked.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sigma = params["sigma"]
        bpm_value = params["bpm_value"]
        iters = params["iters"]

        for ad in adinputs:

            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by clipSigmaBPM".
                            format(ad.filename))
                continue

            for ext in ad:
                extver = ext.hdr['EXTVER']
                if ext.mask is not None:
                    # MCW 190218: Form a masked array to operate on
                    masked_data = np.ma.masked_where(ext.mask != 0,
                                                     ext.data, copy=True)
                    # Perform the sigma clip
                    clipd = sigma_clip(
                        # ext.data,
                        masked_data,
                        sigma=sigma, maxiters=iters, copy=True)
                    # Convert the mask from the return into 0s and 1s and
                    # bitwise OR into the ext BPM
                    clipd_mask = clipd.mask.astype(ext.mask.dtype)
                    ext.mask |= clipd_mask * bpm_value

                    log.stdinfo('   {}:{}: nPixMasked: {:9d} / {:9d}'.format(
                        ad.filename, extver, np.sum(clipd_mask), ext.data.size))

                    # Original implementaion
                    # mean_data = np.mean(ext.data)
                    # sigma_data = np.std(ext.data)
                    # mask_map = (np.abs(ext.data-mean_data) > sigma*sigma_data)
                    # if bpm_value:  # might call with None for diagnosis
                    #     ext.mask[mask_map] |= bpm_value
                    #
                    # log.stdinfo('   {}:{}: nPixMasked: {:9d} / {:9d}'.format(
                    #     ad.filename, extver, np.sum(mask_map), ext.data.size))
                else:
                    log.warning('No DQ plane in {}:{}'.format(ad.filename,
                                                              extver))

            # Timestamp; DO NOT update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        return adinputs

    def darkCorrect(self, adinputs=None, **params):
        """
        Dark-correct GHOST observations.

        This primitive, at its core, simply copies the standard
        DRAGONS darkCorrect (part of :any:`Preprocess`). However, it has
        the ability to examine the binning mode of the requested dark,
        compare it to the adinput(s), and re-bin the dark to the
        correct format.

        To do this, this version of darkCorrect takes over the actual fetching
        of calibrations from :meth:`subtractDark`,
        manipulates the dark(s) as necessary,
        saves the updated dark to the present working directory, and then
        passes the updated list of dark frame(s) on to :meth:`subtractDark`.

        As a result, :any:`IOError` will be raised if the adinputs do not
        all share the same binning mode.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        dark: str/list
            name(s) of the dark file(s) to be subtracted
        do_cal: str
            controls the behaviour of this primitive
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if params['do_cal'] == 'skip':
            log.warning("Dark correction has been turned off.")
            return adinputs

        sfx = params["suffix"]

        # Check if all the inputs have matching detector_x_bin and
        # detector_y_bin descriptors
        if not(all(
                [_.detector_x_bin() == adinputs[0].detector_x_bin() for
                 _ in adinputs])) or not(all(
            [_.detector_y_bin() == adinputs[0].detector_y_bin() for
             _ in adinputs]
        )):
            log.stdinfo('Detector x bins: %s' %
                        str([_.detector_x_bin() for _ in adinputs]))
            log.stdinfo('Detector y bins: %s' %
                        str([_.detector_y_bin() for _ in adinputs]))
            raise IOError('Your input list of files contains a mix of '
                          'different binning modes')

        adinputs_orig = list(adinputs)
        if isinstance(params['dark'], list):
            params['dark'] = [params['dark'][i] for i in range(len(adinputs))
                              if not adinputs[i].phu.get(timestamp_key)]
        adinputs = [_ for _ in adinputs if not _.phu.get(timestamp_key)]
        if len(adinputs) != len(adinputs_orig):
            log.stdinfo('The following files have already been processed by '
                        'darkCorrect and will not be further modified: '
                        '{}'.format(', '.join([_.filename for _ in adinputs_orig
                                               if _ not in adinputs])))

        if params['dark']:
            pass
        else:
            # All this line seems to do is check the valid darks can be found
            # for the adinputs
            self.getProcessedDark(adinputs)

        # Here we need to ape the part of subtractDark which creates the
        # dark_list, then re-bin as required, and send the updated dark_list
        # through to subtractDark
        # This is preferable to writing our own subtractDark, as it should
        # be stable against algorithm changes to dark subtraction
        dark_list = params["dark"] if params["dark"] else [
            self._get_cal(ad, 'processed_dark') for ad in adinputs]

        # We need to make sure we:
        # - Provide a dark AD object for each science frame;
        # - Do not unnecessarily re-bin the same dark to the same binning
        #   multiple times
        dark_list_out = []
        dark_processing_done = {}
        for ad, dark in zip(*gt.make_lists(adinputs, dark_list,
                                           force_ad=True)):
            if dark is None:
                if 'qa' in self.mode:
                    log.warning("No changes will be made to {}, since no "
                                "dark was specified".format(ad.filename))
                    dark_list_out.append(None)
                    continue
                else:
                    raise IOError("No processed dark listed for {}".
                                  format(ad.filename))

            if dark.detector_x_bin() == ad.detector_x_bin() and \
                    dark.detector_y_bin() == ad.detector_y_bin():
                log.stdinfo('Binning for %s already matches input file' %
                            dark.filename)
                dark_list_out.append(dark.filename)
            else:
                xb = ad.detector_x_bin()
                yb = ad.detector_y_bin()
                dark = self._rebin_ghost_ad(dark, xb, yb)
                # Re-name the dark so we don't blow away the old one on save
                dark_filename_orig = dark.filename
                dark.filename = filename_updater(dark,
                                                 suffix='_rebin%dx%d' %
                                                        (xb, yb, ),
                                                 strip=True)
                dark.write(overwrite=True)
                dark_processing_done[
                    (dark_filename_orig, xb, yb)] = dark.filename
                dark_list_out.append(dark.filename)
                log.stdinfo('Wrote out re-binned dark %s' % dark.filename)

            # Check the inputs have matching binning, and shapes
            # Copied from standard darkCorrect (primitives_preprocess)
            # TODO: Check exposure time?
            try:
                gt.check_inputs_match(ad, dark, check_filter=False)
            except ValueError:
                # Else try to extract a matching region from the dark
                log.warning('AD inputs did not match - attempting to clip dark')
                dark = gt.clip_auxiliary_data(ad, aux=dark, aux_type="cal")

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, dark, check_filter=False)

            log.stdinfo("Subtracting the dark ({}) from the input "
                        "AstroData object {}".
                        format(dark.filename, ad.filename))
            ad.subtract(dark)

            # Record dark used, timestamp, and update filename
            ad.phu.set('DARKIM',
                       # os.path.abspath(dark.path),
                       dark.filename,
                       self.keyword_comments["DARKIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs_orig

    def extractProfile(self, adinputs=None, **params):
        """
        Extract the object profile from a slit or flat image.

        This is a primtive wrapper for a collection of :any:`polyfit <polyfit>`
        calls. For each AstroData input, this primitive:

        - Instantiates a :class:`polyfit.GhostArm` class for the input, and
          executes :meth:`polyfit.GhostArm.spectral_format_with_matrix`;
        - Instantiate :class:`polyfit.SlitView` and :class:`polyfit.Extractor`
          objects for the input
        - Extract the profile from the input AstroData, using calls to
          :meth:`polyfit.Extractor.one_d_extract` and
          :meth:`polyfit.Extractor.two_d_extract`.
        
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        slit: str/None
            Name of the (processed & stacked) slit image to use for extraction
            of the profile. If not provided/set to None, the primitive will
            attempt to pull a processed slit image from the calibrations
            database (or, if specified, the --user_cal processed_slit
            command-line option)
        slitflat: str/None
            Name of the (processed) slit flat image to use for extraction
            of the profile. If not provided, set to None, the RecipeSystem
            will attempt to pull a slit flat from the calibrations system (or,
            if specified, the --user_cal processed_slitflat command-line
            option)
        flat: str/None
            Name of the (processed) flat image to use for extraction
            of the profile. If not provided, set to None, the RecipeSystem
            will attempt to pull a slit flat from the calibrations system (or,
            if specified, the --user_cal processed_flat command-line
            option)
        ifu1: str('object'|'sky'|'stowed')/None
            Denotes the status of IFU1. If None, the status is read from the
            header.
        ifu2: str('object'|'sky'|'stowed')/None
            Denotes the status of IFU2. If None, the status is read from the
            header.
        sky_subtract: bool
            subtract the sky from the object spectra?
        seeing: float/None
            can be used to create a synthetic slitviewer image (and synthetic
            slitflat image) if, for some reason, one is not available
        flat_correct: bool
            remove the reponse function from each order before extraction?
        snoise: float
            linear fraction of signal to add to noise estimate for CR flagging
        sigma: float
            number of standard deviations for identifying discrepant pixels
        weighting: str ("uniform"/"optimal")
            weighting scheme for extraction
        writeResult: bool
            Denotes whether or not to write out the result of profile
            extraction to disk. This is useful for both debugging, and data
            quality assurance.
        extract2d: bool
            perform 2D extraction to account for slit tilt?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        flat_correct = params["flat_correct"]
        ifu1 = params["ifu1"]
        ifu2 = params["ifu2"]
        seeing = params["seeing"]
        snoise = params["snoise"]
        sigma = params["sigma"]
        debug_pixel = (params["debug_order"], params["debug_pixel"])
        add_cr_map = params["debug_cr_map"]
        optimal_extraction = params["weighting"] == "optimal"
        ftol = params["tolerance"]
        apply_centroids = params["apply_centroids"]
        timing = params["debug_timing"]

        # This primitive modifies the input AD structure, so it must now
        # check if the primitive has already been applied. If so, it must be
        # skipped.
        adinputs_orig = list(adinputs)
        adinputs = [_ for _ in adinputs if not _.phu.get(timestamp_key)]
        if len(adinputs) != len(adinputs_orig):
            log.stdinfo('extractProfile is skipping the following files, which '
                        'already have extracted profiles: '
                        '{}'.format(','.join([_.filename for _ in adinputs_orig
                                              if _ not in adinputs])))

        # This check is to head off problems where the same flat is used for
        # multiple ADs and gets binned but then needs to be rebinned (which
        # isn't possible because the unbinned flat has been overwritten)
        binnings = set(ad.binning() for ad in adinputs)
        if len(binnings) > 1:
            raise ValueError("Not all input files have the same binning")

        # CJS: Heavily edited because of the new AD way
        # Get processed slits, slitFlats, and flats (for xmod)
        # slits and slitFlats may be provided as parameters
        slit_list = params["slit"]
        # log.stdinfo('slit_list before processing:')
        # log.stdinfo('   {}'.format(slit_list))
        if slit_list is not None and isinstance(slit_list, list):
            slit_list = [slit_list[i] for i in range(len(slit_list))
                         if adinputs_orig[i] in adinputs]
        if slit_list is None:
            # CJS: This populates the calibrations cache (dictionary) with
            # "processed_slit" filenames for each input AD
            self.getProcessedSlit(adinputs)
            # This then gets those filenames
            slit_list = [self._get_cal(ad, 'processed_slit')
                         for ad in adinputs]
        # log.stdinfo('slit_list after processing:')
        # log.stdinfo('   {}'.format(slit_list))

        slitflat_list = params["slitflat"]
        if slitflat_list is not None and isinstance(slitflat_list, list):
            slitflat_list = [slitflat_list[i] for i in range(len(slitflat_list))
                             if adinputs_orig[i] in adinputs]
        if slitflat_list is None:
            self.getProcessedSlitFlat(adinputs)
            slitflat_list = [self._get_cal(ad, 'processed_slitflat')
                             for ad in adinputs]

        flat_list = params['flat']
        if flat_list is None:
            self.getProcessedFlat(adinputs)
            flat_list = [self._get_cal(ad, 'processed_flat')
                         for ad in adinputs]

        # TODO: Have gt.make_lists handle multiple auxiliary lists?
        # CJS: Here we call gt.make_lists. This has only been designed to work
        # with one auxiliary list at present, hence the three calls. This
        # produces two lists of AD objects the same length, one of the input
        # ADs and one of the auxiliary files, from the list
        # of filenames (or single passed parameter). Importantly, if multiple
        # auxiliary frames are the same, then the file is opened only once and
        # the reference to this AD is re-used, saving speed and memory.
        _, slit_list = gt.make_lists(adinputs, slit_list, force_ad=True)
        _, slitflat_list = gt.make_lists(adinputs, slitflat_list, force_ad=True)
        _, flat_list = gt.make_lists(adinputs, flat_list, force_ad=True)

        for ad, slit, slitflat, flat in zip(adinputs, slit_list,
                                            slitflat_list, flat_list):
            # CJS: failure to find a suitable auxiliary file (either because
            # there's no calibration, or it's missing) places a None in the
            # list, allowing a graceful continuation.
            if flat is None:  # can't do anything as no XMOD
                raise RuntimeError(f"No processed flat listed for {ad.filename}")

            if slitflat is None:
                # TBD: can we use a synthetic slitflat (findApertures in
                # makeProcessedFlat would need one too)
                raise RuntimeError(f"No processed slitflat listed for {ad.filename}")
                if slit is None:
                    slitflat_filename = "synthetic"
                    slitflat_data = None
                else:
                    raise RuntimeError(f"{ad.filename} has a processed slit "
                                       "but no processed slitflat")
            else:
                slitflat_filename = slitflat.filename
                slitflat_data = slitflat[0].data

            if slit is None:
                if seeing is None:
                    raise RuntimeError(f"No processed slit listed for {ad.filename}"
                                       "and no seeing estimate has been provided")
                else:
                    slit_filename = f"synthetic (seeing {seeing})"
                    slit_data = None
            else:
                slit_filename = slit.filename
                slit_data = slit[0].data

            # CJS: Changed to log.debug() and changed the output
            log.stdinfo("Slit parameters: ")
            log.stdinfo(f"   processed_slit: {slit_filename}")
            log.stdinfo(f"   processed_slitflat: {slitflat_filename}")
            log.stdinfo(f"   processed_flat: {flat.filename}")

            res_mode = ad.res_mode()
            arm = GhostArm(arm=ad.arm(), mode=res_mode,
                           detector_x_bin=ad.detector_x_bin(),
                           detector_y_bin=ad.detector_y_bin())

            ifu_status = ["stowed", "sky", "object"]
            if ifu1 is None:
                try:
                    ifu1 = ifu_status[ad.phu['TARGET1']]
                except (KeyError, IndexError):
                    raise RuntimeError(f"{self.myself()}: ifu1 set to 'None' "
                                       f"but 'TARGET1' keyword missing/incorrect")
            if ifu2 is None:
                if res_mode == 'std':
                    try:
                        ifu2 = ifu_status[ad.phu['TARGET2']]
                    except (KeyError, IndexError):
                        raise RuntimeError(f"{self.myself()}: ifu2 set to 'None' "
                                           f"but 'TARGET1' keyword missing/incorrect")
                else:
                    ifu2 = "object" if ad.phu.get('THXE') == 1 else "stowed"

            log.stdinfo(f"\nIFU status: {ifu1} and {ifu2}")
            ifu_stowed = [obj for obj, ifu in enumerate([ifu1, ifu2])
                          if ifu == "stowed"]

            # CJS 20220411: We need to know which IFUs contain an object if
            # we need to make a synthetic slit profile so this has moved up
            if 'ARC' in ad.tags:
                # Extracts entire slit first, and then the two IFUs separately
                objs_to_use = [[], list(set([0, 1]) - set(ifu_stowed))]
                use_sky = [True, False]
                find_crs = [False, False]
                sky_correct_profiles = False
            else:
                ifu_selection = [obj for obj, ifu in enumerate([ifu1, ifu2])
                                 if ifu == "object"]
                objs_to_use = [ifu_selection]
                use_sky = [params["sky_subtract"] if ifu_selection else True]
                find_crs = [True]
                sky_correct_profiles = True

            # CJS: Heavy refactor. Return the filename for each calibration
            # type. Eliminates requirement that everything be updated
            # simultaneously.
            # key = self._get_polyfit_key(ad)
            # log.stdinfo("Polyfit key selected: {}".format(key))
            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                log.warning("Cannot open required initial model files for {};"
                            " skipping".format(ad.filename))
                continue

            arm.spectral_format_with_matrix(flat[0].XMOD, wpars[0].data,
                        spatpars[0].data, specpars[0].data, rotpars[0].data)
            sview_kwargs = {} if slit is None else {"binning": slit.detector_x_bin()}
            sview = SlitView(slit_data, slitflat_data,
                             slitvpars.TABLE[0], mode=res_mode,
                             microns_pix=4.54 * 180 / 50,
                             stowed=ifu_stowed, **sview_kwargs)

            # There's no point in creating a fake slitflat first, since that
            # will case the code to use it to determine the fibre positions,
            # but the fibre positions are just the defaults
            if slit is None:
                log.stdinfo(f"Creating synthetic slit image for seeing={seeing}")
                ifus = []
                if ifu1 == 'object':
                    ifus.append('ifu0' if res_mode == 'std' else 'ifu')
                # "ifu2" is the ThXe cal fiber in HR and has no continuum
                if ifu2 == 'object' and res_mode == 'std':
                    ifus.append('ifu1')
                slit_data = sview.fake_slitimage(
                    flat_image=slitflat_data, ifus=ifus, seeing=seeing)
            elif not ad.tags.intersection({'FLAT', 'ARC'}):
                # TODO? This only models IFU0 in SR mode
                slit_models = sview.model_profile(slit_data, slitflat_data)
                log.stdinfo("")
                for k, model in slit_models.items():
                    fwhm, apfrac = model.estimate_seeing()
                    log.stdinfo(f"Estimated seeing in the {k} arm: {fwhm:5.3f}"
                                f" ({apfrac*100:.1f}% aperture throughput)")
            if slitflat is None:
                log.stdinfo("Creating synthetic slitflat image")
                slitflat_data = sview.fake_slitimage()

            extractor = Extractor(arm, sview, badpixmask=ad[0].mask,
                                  vararray=ad[0].variance)
                        
            # FIXME: This really could be done as part of flat processing!
            correction = None
            if flat_correct:
                try:
                    blaze = flat[0].BLAZE
                except AttributeError:
                    log.warning(f"Flatfield {flat.filename} has no BLAZE, so"
                                "cannot perform flatfield correction")
                else:
                    ny, nx = blaze.shape
                    xbin = ad.detector_x_bin()
                    binned_blaze = blaze.reshape(ny, nx // xbin, xbin).mean(axis=-1)
                    correction = np.where(binned_blaze < 0.0001, 0, 1. / binned_blaze)

            for i, (o, s, cr) in enumerate(zip(objs_to_use, use_sky, find_crs)):
                if o:
                    log.stdinfo(f"\nExtracting objects {str(o)}; sky subtraction {str(s)}")
                elif use_sky:
                    log.stdinfo("\nExtracting entire slit")
                else:
                    raise RuntimeError("No objects for extraction and use_sky=False")

                extracted_flux, extracted_var = extractor.new_extract(
                    data=ad[0].data.copy(),
                    correct_for_sky=sky_correct_profiles,
                    use_sky=s, used_objects=o, find_crs=cr,
                    snoise=snoise, sigma=sigma,
                    debug_pixel=debug_pixel,
                    correction=correction, optimal=optimal_extraction,
                    apply_centroids=apply_centroids, ftol=ftol,
                    timing=timing
                )

                # Append the extraction as a new extension... we don't
                # replace ad[0] since that will still be needed if we have
                # another iteration of the extraction loop
                ad.append(extracted_flux)
                ad[-1].mask = (extracted_var == 0).astype(DQ.datatype)
                ad[-1].variance = extracted_var
                ad[-1].nddata.meta['header'] = ad[0].hdr.copy()
                if add_cr_map and extractor.badpixmask is not None:
                    ad[-1].CR = extractor.badpixmask & DQ.cosmic_ray
                ad[-1].hdr['DATADESC'] = (
                    'Order-by-order processed science data - '
                    'objects {}, subtraction = {}'.format(
                        str(o), str(s)),
                    self.keyword_comments['DATADESC'])

            del ad[0]   # Remove original echellogram data
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            ad.phu.set("FLATIM", flat.filename, self.keyword_comments["FLATIM"])
            if params["write_result"]:
                ad.write(overwrite=True)

        return adinputs_orig

    def interpolateAndCombine(self, adinputs=None, **params):
        """
        Combine the independent orders from the input ADs into a single,
        over-sampled spectrum.

        The wavelength scale of the output is determined by finding the
        wavelength range of the input, and generating a new
        wavelength sampling in accordance with the ``scale`` and
        ``oversample`` parameters.

        The output spectrum is constructed as follows:

        - A blank spectrum, corresponding to the new wavelength scale, is
          initialised;
        - For each order of the input AstroData object:

            - The spectrum order is re-gridded onto the output wavelength scale;
            - The re-gridded order is averaged with the final output spectrum
              to form a new output spectrum.

          This process continues until all orders have been averaged into the
          final output spectrum.

        Note that the un-interpolated data is kept - the interpolated data
        is appended to the end of the file as a new extension.

        Parameters
        ----------
        scale : str
            Denotes what scale to generate for the final spectrum. Currently
            available are:
            ``'loglinear'``
            Default is ``'loglinear'``.
        oversample : int or float
            The factor by which to (approximately) oversample the final output
            spectrum, as compared to the input spectral orders. Defaults to 1.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        adoutputs = []
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by interpolateAndCombine".
                            format(ad.filename))
                adoutputs.append(ad)
                continue

            adout = astrodata.create(ad.phu)
            for ext in ad:
                # Determine the wavelength bounds of the file
                min_wavl, max_wavl = np.min(ext.WAVL), np.max(ext.WAVL)
                logspacing = np.median(
                    np.log(ext.WAVL[:, 1:]) - np.log(ext.WAVL[:, :-1])
                )
                # Form a new wavelength scale based on these extremes
                if params['scale'] == 'loglinear':
                    wavl_grid = np.exp(
                        np.linspace(np.log(min_wavl), np.log(max_wavl),
                                    num=int(
                                        (np.log(max_wavl) - np.log(min_wavl)) /
                                        (logspacing / float(params['oversample']))
                                    ))
                    )
                else:
                    raise ValueError('interpolateAndCombine does not understand '
                                     'the scale {}'.format(params['scale']))

                # Create a final spectrum and (inverse) variance to match
                # (One plane per object)
                no_obj = ext.data.shape[-1]
                # We can easily get underflows for the VAR in np.float32
                # for flux-calibrated data
                spec_final = np.zeros(wavl_grid.shape + (no_obj, ), dtype=np.float64)
                var_final = np.zeros_like(spec_final)
                mask_final = np.zeros_like(spec_final, dtype=DQ.datatype)

                # Loop over each input order, making the output spectrum the
                # result of the weighted average of itself and the order
                # spectrum
                for order in range(ext.data.shape[0]):
                    for ob in range(ext.data.shape[-1]):
                        log.debug(f'Re-gridding order {order:2d}, obj {ob:1d}')
                        flux_for_adding = np.interp(wavl_grid,
                                                    ext.WAVL[order],
                                                    ext.data[order, :, ob],
                                                    left=0, right=0)
                        # again, float64 to avoid VAR underflows
                        ivar_for_adding = np.interp(wavl_grid,
                                                    ext.WAVL[order],
                                                    at.divide0(1.0,
                                                    ext.variance[order, :, ob].astype(np.float64)),
                                                    left=0, right=0)
                        # Ensure we don't interpolate masked pixels
                        try:
                            mask_for_adding = np.interp(wavl_grid,
                                                        ext.WAVL[order],
                                                        ext.mask[order, :, ob] & DQ.not_signal,
                                                        left=0, right=0)
                        except TypeError:  # ext.mask is None
                            pass
                        else:
                            ivar_for_adding[mask_for_adding > 0] = 0
                        spec_comp, ivar_comp = np.ma.average(
                            np.asarray([spec_final[:, ob], flux_for_adding]),
                            weights=np.asarray([at.divide0(1.0, var_final[:, ob]),
                                                ivar_for_adding]),
                            returned=True, axis=0,
                        )
                        spec_final[:, ob] = deepcopy(spec_comp)
                        var_final[:, ob] = deepcopy(1.0 / ivar_comp)

                # import pdb;
                # pdb.set_trace()

                # Can't use .reset without looping through extensions
                mask_final[np.logical_or.reduce([np.isnan(spec_final),
                                                 np.isinf(var_final),
                                                 var_final == 0])] = DQ.bad_pixel
                spec_final[np.isnan(spec_final)] = 0
                var_final[np.isinf(var_final)] = 0
                adout.append(astrodata.NDAstroData(data=spec_final.astype(np.float32),
                                                   mask=mask_final,
                                                meta={'header': deepcopy(ext.hdr)}))
                adout[-1].variance = var_final.astype(np.float32)
                adout[-1].WAVL = wavl_grid
                adout[-1].hdr['DATADESC'] = ('Interpolated data',
                                             self.keyword_comments['DATADESC'])

            # Timestamp & update filename
            gt.mark_history(adout, primname=self.myself(), keyword=timestamp_key)
            adout.update_filename(suffix=params["suffix"], strip=True)
            adoutputs.append(adout)

        return adoutputs

    def findApertures(self, adinputs=None, **params):
        """
        Locate the slit aperture, parametrized by an :any:`polyfit` model.

        The primitive locates the slit apertures within a GHOST frame,
        and inserts a :any:`polyfit` model into a new extension on each data
        frame. This model is placed into a new ``.XMOD`` attribute on the
        extension.
        
        Parameters
        ----------
        slitflat: str or :class:`astrodata.AstroData` or None
            slit flat to use; if None, the calibration system is invoked
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        make_pixel_model = params.get('make_pixel_model', False)

        # Make no attempt to check if primitive has already been run - may
        # have new calibrators we wish to apply.

        # CJS: See comment in extractProfile() for handling of calibrations
        flat_list = params["slitflat"]
        if flat_list is None:
            self.getProcessedSlitFlat(adinputs)
            flat_list = [self._get_cal(ad, 'processed_slitflat')
                         for ad in adinputs]

        for ad, slit_flat in zip(*gt.make_lists(adinputs, flat_list,
                                                force_ad=True)):
            if not {'PREPARED', 'GHOST', 'FLAT'}.issubset(ad.tags):
                log.warning("findApertures is only run on prepared flats: "
                            "{} will not be processed".format(ad.filename))
                continue

            try:
                poly_xmod = self._get_polyfit_filename(ad, 'xmod')
                log.stdinfo('Found xmod: {}'.format(poly_xmod))
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                log.stdinfo('Found spatmod: {}'.format(poly_spat))
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                log.stdinfo('Found slitvmod: {}'.format(slitv_fn))
                xpars = astrodata.open(poly_xmod)
                spatpars = astrodata.open(poly_spat)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                log.warning("Cannot open required initial model files for {};"
                            " skipping".format(ad.filename))
                continue

            arm = ad.arm()
            res_mode = ad.res_mode()
            ghost_arm = GhostArm(arm=arm, mode=res_mode)

            # Create an initial model of the spectrograph
            xx, wave, blaze = ghost_arm.spectral_format(xparams=xpars[0].data)

            slitview = SlitView(slit_flat[0].data, slit_flat[0].data,
                                slitvpars.TABLE[0], mode=res_mode,
                                microns_pix=4.54*180/50,
                                binning=slit_flat.detector_x_bin())

            # Convolve the flat field with the slit profile
            flat_conv = ghost_arm.slit_flat_convolve(
                ad[0].data,
                slit_profile=slitview.slit_profile(arm=arm),
                spatpars=spatpars[0].data, microns_pix=slitview.microns_pix,
                xpars=xpars[0].data
            )

            # Fit the initial model to the data being considered
            fitted_params = ghost_arm.fit_x_to_image(flat_conv,
                                                     xparams=xpars[0].data,
                                                     decrease_dim=8,
                                                     sampling=2,
                                                     inspect=False)

            # CJS: Append the XMOD as an extension. It will inherit the
            # header from the science plane (including irrelevant/wrong
            # keywords like DATASEC) but that's not really a big deal.
            # (The header can be modified/started afresh if needed.)
            ad[0].XMOD = fitted_params

            #MJI: Compute a pixel-by-pixel model of the flat field from the new XMOD and
            #the slit image.
            if make_pixel_model:
                # FIXME: MJI Copied directly from extractProfile. Is this compliant?
                try:
                    poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                    poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                    poly_spec = self._get_polyfit_filename(ad, 'specmod')
                    poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                    wpars = astrodata.open(poly_wave)
                    spatpars = astrodata.open(poly_spat)
                    specpars = astrodata.open(poly_spec)
                    rotpars = astrodata.open(poly_rot)
                except IOError:
                    log.warning("Cannot open required initial model files "
                                "for {}; skipping".format(ad.filename))
                    continue

                #Create an extractor instance, so that we can add the pixel model to the 
                #data.
                ghost_arm.spectral_format_with_matrix(ad[0].XMOD, wpars[0].data,
                            spatpars[0].data, specpars[0].data, rotpars[0].data)
                extractor = Extractor(ghost_arm, slitview, badpixmask=ad[0].mask,
                                      vararray=ad[0].variance)
                pixel_model = extractor.make_pixel_model()
                ad[0].PIXELMODEL = pixel_model

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        return adinputs

    def fitWavelength(self, adinputs=None, **params):
        """
        Fit wavelength solution to a GHOST ARC frame.

        This primitive should only be applied to a reduce GHOST ARC frame. Any
        other files passed through this primitive will be skipped.

        This primitive works as follows:
        - :class:`polyfit.ghost.GhostArm` and `polyfit.extract.Extractor`
          classes are instantiated and configured for the data;
        - The ``Extractor`` class is used to find the line locations;
        - The ``GhostArm`` class is used to fit this line solution to the data.

        The primitive will use the arc line files stored in the same location
        as the initial :module:`polyfit` models kept in the ``lookups`` system.

        This primitive uses no special parameters.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        min_snr = params["min_snr"]
        sigma = params["sigma"]
        max_iters = params["max_iters"]
        radius = params["radius"]
        plot1d = params["plot1d"]
        plotrms = params["plotrms"]
        plot2d = params["debug_plot2d"]

        # import pdb; pdb.set_trace()

        # Make no attempt to check if primitive has already been run - may
        # have new calibrators we wish to apply.

        flat_list = params['flat']
        if not flat_list:
            self.getProcessedFlat(adinputs)
            flat_list = [self._get_cal(ad, 'processed_flat') for ad in adinputs]

        for ad, flat in zip(*gt.make_lists(adinputs, flat_list, force_ad=True)):
            if self.timestamp_keys["extractProfile"] not in ad.phu:
                log.warning("extractProfile has not been run on {} - "
                            "skipping".format(ad.filename))
                continue

            if flat is None:
                log.warning("Could not find processed_flat calibration for "
                            "{} - skipping".format(ad.filename))
                continue

            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
            except IOError:
                log.warning("Cannot open required initial model files for {};"
                            " skipping".format(ad.filename))
                continue

            # CJS: line_list location is now in lookups/__init__.py
            arclinefile = os.path.join(os.path.dirname(polyfit_lookup.__file__),
                                       line_list)
            arcwaves, arcfluxes = np.loadtxt(arclinefile, usecols=[1, 2]).T

            arm = GhostArm(arm=ad.arm(), mode=ad.res_mode())

            # Find locations of all significant peaks in all orders
            nm, ny, _ = ad[0].data.shape
            all_peaks = []
            pixels = np.arange(ny)
            for m_ix, flux in enumerate(ad[0].data[:, :, 0]):
                try:
                    variance = ad[0].variance[m_ix, :, 0]
                except TypeError:  # variance is None
                    nmad = median_filter(
                        abs(flux - median_filter(flux, size=2*radius+1)),
                        size=2*radius+1)
                    variance = nmad * nmad
                peaks = tracing.find_peaks(flux.copy(), widths=np.arange(2.5, 4.5, 0.1),
                                           variance=variance, min_snr=min_snr, min_sep=5,
                                           pinpoint_index=None, reject_bad=False)
                fit_g = fitting.LevMarLSQFitter()
                these_peaks = []
                for x in peaks[0]:
                    good = np.zeros_like(flux, dtype=bool)
                    good[max(int(x - radius), 0):int(x + radius + 1)] = True
                    g_init = models.Gaussian1D(mean=x, amplitude=flux[good].max(),
                                               stddev=1.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        g = fit_g(g_init, pixels[good], flux[good])
                    if g.stddev.value < 5:  # avoid clearly bad fits
                        these_peaks.append(g)
                all_peaks.append(these_peaks)

            # Find lines based on the extracted flux and the arc wavelengths.
            # Note that "inspect=True" also requires and input arc file, which has
            # the non-extracted data. There is also a keyword "plots".
            #lines_out = extractor.find_lines(ad[0].data, arcwaves,
            #                                 arcfile=ad[0].data,
            #                                 plots=params['plot_fit'])

            wmod_shape = wpars[0].data.shape
            m_init = WaveModel(model=wpars[0].data, arm=arm)
            arm.spectral_format_with_matrix(
                flat[0].XMOD, m_init.parameters.reshape(wmod_shape),
                spatpars[0].data, specpars[0].data, rotpars[0].data)
            extractor = Extractor(arm, None)  # slitview=None for this usage

            for iter in (0, 1):
                # Only cross-correlate on the first pass
                lines_out = extractor.match_lines(all_peaks, arcwaves,
                                                  hw=radius, xcor=not bool(iter),
                                                  log=(self.log, None)[iter])
                #lines_out is now a long vector of many parameters, including the
                #x and y position on the chip of each line, the order, the expected
                #wavelength, the measured line strength and the measured line width.
                #fitted_params, wave_and_resid = arm.read_lines_and_fit(
                #    wpars[0].data, lines_out)
                y_values, waves, orders = lines_out[:, 1], lines_out[:, 0], lines_out[:, 3]
                fit_it = fitting.FittingWithOutlierRemoval(
                    fitting.LevMarLSQFitter(), sigma_clip, sigma=sigma,
                    maxiters=max_iters)
                m_final, mask = fit_it(m_init, y_values, orders, waves)
                arm.spectral_format_with_matrix(
                    flat[0].XMOD, m_final.parameters.reshape(wmod_shape),
                    spatpars[0].data, specpars[0].data, rotpars[0].data)
                m_init = m_final

            fitted_waves = m_final(y_values, orders)
            rms = np.std((fitted_waves - waves)[~mask])
            nlines = y_values.size - mask.sum()
            log.stdinfo(f"Fit residual RMS (Angstroms): {rms:7.4f} ({nlines} lines)")
            if np.any(mask):
                log.stdinfo("The following lines were rejected:")
                for yval, order, wave, fitted, m in zip(
                        y_values, orders, waves, fitted_waves, mask):
                    if m:
                        log.stdinfo(f"    Order {int(order):2d} pixel {yval:6.1f} "
                                    f"wave {wave:10.4f} (fitted wave {fitted:10.4f})")

            if plot2d:
                plt.ioff()
                fig, ax = plt.subplots()
                xpos = lambda y, ord: arm.szx // 2 + np.interp(
                    y, pixels, arm.x_map[ord])
                for m_ix, peaks in enumerate(all_peaks):
                    for g in peaks:
                        ax.plot([g.mean.value] * 2, xpos(g.mean.value, m_ix) +
                                np.array([-30, 30]), color='darkgrey', linestyle='-')
                for (wave, yval, xval, order, amp, fwhm) in lines_out:
                    xval = xpos(yval, int(order) - arm.m_min)
                    ax.plot([yval, yval],  xval + np.array([-30, 30]), 'r-')
                    ax.text(yval + 10, xval + 30, str(wave), color='blue', fontsize=10)
                plt.show()
                plt.ion()

            if plot1d:
                plot_extracted_spectra(ad, arm, all_peaks, lines_out, mask, nrows=4)

            if plotrms:
                m_final.plotit(y_values, orders, waves, mask,
                               filename=ad.filename.replace('.fits', '_2d.pdf'))

            # CJS: Append the WFIT as an extension. It will inherit the
            # header from the science plane (including irrelevant/wrong
            # keywords like DATASEC) but that's not really a big deal.
            ad[0].WFIT = m_final.parameters.reshape(wpars[0].data.shape)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def flatCorrect(self, adinputs=None, **params):
        """
        Flat-correct an extracted GHOST profile using a flat profile.

        This primitive works by extracting the
        profile from the relevant flat field using the object's extracted
        weights, and then performs simple division.

        .. warning::
            While the primitive is working, it has been found that the
            underlying algorithm is flawed. A new algorithm is being developed.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str/None
            Name of the (processed) standard flat to use for flat profile
            extraction. If None, the primitive will attempt to pull a flat
            from the calibrations database (or, if specified, the
            --user_cal processed_flat command-line option)
        slit: str/None
            Name of the (processed & stacked) slit image to use for extraction
            of the profile. If not provided/set to None, the primitive will
            attempt to pull a processed slit image from the calibrations
            database (or, if specified, the --user_cal processed_slit
            command-line option)
        slitflat: str/None
            Name of the (processed) slit flat image to use for extraction
            of the profile. If not provided, set to None, the RecipeSystem
            will attempt to pull a slit flat from the calibrations system (or,
            if specified, the --user_cal processed_slitflat command-line
            option)
        writeResult: bool
            Denotes whether or not to write out the result of profile
            extraction to disk. This is useful for both debugging, and data
            quality assurance.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        if params['skip']:
            log.stdinfo('Skipping the flat field correction step')
            return adinputs

        adinputs_orig = list(adinputs)
        adinputs = [_ for _ in adinputs if not _.phu.get(timestamp_key)]
        if len(adinputs) != len(adinputs_orig):
            log.stdinfo('flatCorrect is skipping the following files, '
                        'which are already flat corrected: '
                        '{}'.format(','.join([_ for _ in adinputs_orig
                                              if _ not in adinputs])))

        # CJS: See extractProfile() refactoring for explanation of changes
        slit_list = params["slit"]
        if slit_list is not None and isinstance(slit_list, list):
            slit_list = [slit_list[i] for i in range(len(slit_list))
                         if adinputs_orig[i] in adinputs]
        if slit_list is None:
            self.getProcessedSlit(adinputs)
            slit_list = [self._get_cal(ad, 'processed_slit')
                         for ad in adinputs]

        # CJS: I've renamed flat -> slitflat and obj_flat -> flat because
        # that's what the things are called! Sorry if I've overstepped.
        slitflat_list = params["slitflat"]
        if slitflat_list is not None and isinstance(slitflat_list, list):
            slitflat_list = [slitflat_list[i] for i in range(len(slitflat_list))
                         if adinputs_orig[i] in adinputs]
        if slitflat_list is None:
            self.getProcessedSlitFlat(adinputs)
            slitflat_list = [self._get_cal(ad, 'processed_slitflat')
                         for ad in adinputs]

        flat_list = params["flat"]
        if flat_list is not None and isinstance(flat_list, list):
            flat_list = [flat_list[i] for i in range(len(flat_list))
                         if adinputs_orig[i] in adinputs]
        if flat_list is None:
            self.getProcessedFlat(adinputs)
            flat_list = [self._get_cal(ad, 'processed_flat')
                         for ad in adinputs]

        # TODO: Have gt.make_lists handle multiple auxiliary lists?
        _, slit_list = gt.make_lists(adinputs, slit_list, force_ad=True)
        _, slitflat_list = gt.make_lists(adinputs, slitflat_list, force_ad=True)
        _, flat_list = gt.make_lists(adinputs, flat_list, force_ad=True)

        for ad, slit, slitflat, flat, in zip(adinputs, slit_list,
                                             slitflat_list, flat_list):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by flatCorrect".
                            format(ad.filename))
                continue

            # CJS: failure to find a suitable auxiliary file (either because
            # there's no calibration, or it's missing) places a None in the
            # list, allowing a graceful continuation.
            if slit is None or slitflat is None or flat is None:
                log.warning("Unable to find calibrations for {}; "
                            "skipping".format(ad.filename))
                continue

            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                log.warning("Cannot open required initial model files for {};"
                            " skipping".format(ad.filename))
                continue

            res_mode = ad.res_mode()
            arm = GhostArm(arm=ad.arm(), mode=res_mode, 
                           detector_x_bin= ad.detector_x_bin(),
                           detector_y_bin= ad.detector_y_bin()
                           )
            arm.spectral_format_with_matrix(flat[0].XMOD,
                                            wpars[0].data,
                                            spatpars[0].data,
                                            specpars[0].data,
                                            rotpars[0].data,
                                            )

            sview = SlitView(slit[0].data, slitflat[0].data,
                             slitvpars.TABLE[0], mode=res_mode,
                             microns_pix=4.54*180/50,
                             binning = slit.detector_x_bin())

            extractor = Extractor(arm, sview)

            #FIXME - Marc and were *going* to try:
            #adjusted_data = arm.bin_data(extractor.adjust_data(flat[0].data))
            extracted_flux, extracted_var = extractor.two_d_extract(
                arm.bin_data(flat[0].data), extraction_weights=ad[0].WGT)

            # Normalised extracted flat profile
            med = np.median(extracted_flux)
            extracted_flux /= med
            extracted_var /= med**2

            flatprof_ad = deepcopy(ad)
            flatprof_ad.update_filename(suffix='_extractedFlatProfile',
                                        strip=True)
            flatprof_ad[0].reset(extracted_flux, mask=None,
                                 variance=extracted_var)
            if params["write_result"]:
                flatprof_ad.write(overwrite=True)
                # Record this as the flat profile used
                ad.phu.set('FLATPROF', os.path.abspath(flatprof_ad.path),
                           self.keyword_comments['FLATPROF'])
            ad.phu.set('FLATIMG', os.path.abspath(flat.path),
                       keyword_comments.keyword_comments['FLATIMG'])
            ad.phu.set('SLITIMG', os.path.abspath(slit.path),
                       keyword_comments.keyword_comments['SLITIMG'])
            ad.phu.set('SLITFLAT', os.path.abspath(slitflat.path),
                       keyword_comments.keyword_comments['SLITFLAT'])

            # Divide the flat field through the science data
            # Arithmetic propagates VAR correctly
            ad /= flatprof_ad

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        # This nomenclature is misleading - this is the list of
        # intitially-passed AstroData objects, some of which may have been
        # skipped, and others which should have been modified by this
        # primitive
        return adinputs_orig

    def formatOutput(self, adinputs=None, **params):
        """
        Generate an output FITS file containing the data requested by the user.

        This primitive should not be called until *all* required
        processing steps have been performed on the data. THe resulting FITS
        file cannot be safely passed through to other primitives.

        .. note::
            All of the extra data packaged up by this primitive can also be
            obtained by using the ``write_result=True`` flag on selected
            other primitives. ``formatOutput`` goes and finds those output
            files, and then packages them into the main output file for
            convenience.

        Parameters
        ----------
        detail: str
            The level of detail the user would like in their final output file.

            Note that, in order to preserve the ordering of FITS file
            extensions, the options are sequential; each option will
            provide all the data of less-verbose options.

            Valid options are:

            ``default``
                Only returns the extracted, fully-processed object(s) and sky
                spectra. In effect, this causes ``formatOutput`` to do nothing.
                This includes computed variance data for each plane.

            ``processed_image``
                The option returns the data that have been bias and dark
                corrected, and has the flat BPM applied (i.e. the state the
                data are in immediately prior to profile extraction).

            ``flat_profile``
                This options includes the extracted flat profile used for
                flat-fielding the data.

            ``sensitivity_curve``
                This option includes the sensitivity calculated at the
                :meth:`responseCorrect <responseCorrect>` step of reduction.
        """

        # This should be the list of allowed detail descriptors in order of
        # increasing verbosity
        ALLOWED_DETAILS = ['default', 'processed_image', 'flat_profile',
                           'sensitivity_curve', ]

        log = self.log

        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params['suffix']

        if params['detail'] not in ALLOWED_DETAILS:
            raise ValueError('formatOutput: detail option {} not known. '
                             'Please use one of: {}'.format(
                params['detail'],
                ', '.join(ALLOWED_DETAILS),
            ))

        detail_index = ALLOWED_DETAILS.index(params['detail'])

        for ad in adinputs:
            # Move sequentially through the various levels of detail, adding
            # them as we go along
            # ad[0].hdr['DATADESC'] = ('Fully-reduced data',
            #                          self.keyword_comments['DATADESC'], )

            if ALLOWED_DETAILS.index('processed_image') <= detail_index:
                # Locate the processed image data
                fn = ad.phu.get('PROCIMG', None)
                if fn is None:
                    raise RuntimeError('The processed image file name for {} '
                                       'has not been '
                                       'recorded'.format(ad.filename))
                try:
                    proc_image = astrodata.open(fn)
                except astrodata.AstroDataError:
                    raise RuntimeError('You appear not to have written out '
                                       'the result of image processing to '
                                       'disk.')

                log.stdinfo('Opened processed image file {}'.format(fn))
                ad.append(proc_image[0])
                ad[-1].hdr['DATADESC'] = ('Processed image',
                                          self.keyword_comments['DATADESC'])

            if ALLOWED_DETAILS.index('flat_profile') <= detail_index:
                # Locate the flat profile data
                fn = ad.phu.get('FLATPROF', None)
                if fn is None:
                    raise RuntimeError('The flat profile file name for {} '
                                       'has not been '
                                       'recorded'.format(ad.filename))
                try:
                    proc_image = astrodata.open(fn)
                except astrodata.AstroDataError:
                    raise RuntimeError('You appear not to have written out '
                                       'the result of flat profiling to '
                                       'disk.')

                log.stdinfo('Opened flat profile file {}'.format(fn))
                # proc_image[0].WGT = None
                try:
                    del proc_image[0].WGT
                except AttributeError:
                    pass
                ad.append(proc_image[0])
                ad[-1].hdr['DATADESC'] = ('Flat profile',
                                          self.keyword_comments['DATADESC'])

            if ALLOWED_DETAILS.index('sensitivity_curve') <= detail_index:
                fn = ad.phu.get('SENSFUNC', None)
                if fn is None:
                    raise RuntimeError('The sensitivity curve file name for {} '
                                       'has not been '
                                       'recorded'.format(ad.filename))
                try:
                    proc_image = astrodata.open(fn)
                except astrodata.AstroDataError:
                    raise RuntimeError('You appear not to have written out '
                                       'the result of sensitivity calcs to '
                                       'disk.')

                log.stdinfo('Opened sensitivity curve file {}'.format(fn))
                # proc_image[0].WGT = None
                try:
                    del proc_image[0].WGT
                except AttributeError:
                    pass
                try:
                    del proc_image[0].WAVL
                except AttributeError:
                    pass
                ad.append(proc_image[0])
                ad[-1].hdr['DATADESC'] = ('Sensitivity curve (blaze func.)',
                                          self.keyword_comments['DATADESC'])

            # import pdb; pdb.set_trace();

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            ad.write(overwrite=True)

        return adinputs

    def measureBlaze(self, adinputs=None, **params):
        """

        Parameters
        ----------

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params['suffix']
        order = params.get('order')
        sigma = 3
        max_iters = 5

        slitflat_list = params["slitflat"]
        if slitflat_list is None:
            self.getProcessedSlitFlat(adinputs)
            slitflat_list = [self._get_cal(ad, 'processed_slitflat')
                             for ad in adinputs]

        for ad, slitflat in zip(*gt.make_lists(adinputs, slitflat_list, force_ad=True)):
            res_mode = ad.res_mode()
            arm = GhostArm(arm=ad.arm(), mode=res_mode,
                           detector_x_bin=ad.detector_x_bin(),
                           detector_y_bin=ad.detector_y_bin())

            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                raise RuntimeError("Cannot open required initial model files "
                                   f"for {ad.filename}; skipping")

            arm.spectral_format_with_matrix(ad[0].XMOD, wpars[0].data,
                        spatpars[0].data, specpars[0].data, rotpars[0].data)
            sview = SlitView(slitflat[0].data, slitflat[0].data,
                             slitvpars.TABLE[0], mode=res_mode,
                             microns_pix=4.54 * 180 / 50,
                             binning=slitflat.detector_x_bin())
            extractor = Extractor(arm, sview, badpixmask=ad[0].mask,
                                  vararray=ad[0].variance)
            extracted_flux, extracted_var, extracted_mask = extractor.quick_extract(ad[0].data)
            med_flux = np.median(extracted_flux[extracted_mask == 0])
            extracted_flux /= med_flux

            # Due to poorly-determined gains, we can't fit a function and use
            # that: there's a jump at the middle of each order. But I'll leave
            # this code here, in case that changes.
            if order is not None:
                for i, (flux, var, mask) in enumerate(
                        zip(extracted_flux, extracted_var, extracted_mask)):
                    ny = extracted_flux.shape[1]
                    m_init = models.Chebyshev1D(degree=order, domain=[0, ny-1])
                    fit_it = fitting.FittingWithOutlierRemoval(
                        fitting.LinearLSQFitter(), sigma_clip, sigma=sigma,
                        maxiters=max_iters)
                    m_final, _ = fit_it(m_init, np.arange(ny),
                                        np.ma.masked_where(mask, flux),
                                        weights=at.divide0(1., np.sqrt(var)))
                    print(f"Plotting {arm.m_min+i}")
                    fig, ax = plt.subplots()
                    ally = np.arange(ny)
                    ax.plot(ally[mask==0], flux[mask==0], 'bo')
                    ax.plot(ally[_], flux[_], 'ro')
                    ax.plot(ally, m_final(ally), 'k-')
                    plt.show()

            ad[0].BLAZE = extracted_flux
            ad[0].BLAZE[extracted_mask.astype(bool)] = 0
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def removeScatteredLight(self, adinputs=None, **params):
        """

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        skip: bool
            skip primitive entirely?
        debug_spline_smoothness: float
            scaling factor for spline smoothness
        debug_save_model: bool
            attach scattered light model to output
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        if params["skip"]:
            log.stdinfo("Not removing scattered light since skip=True")
            return adinputs

        smoothness = params["debug_spline_smoothness"]
        save_model = params["debug_save_model"]
        is_flat = ['FLAT' in ad.tags for ad in adinputs]
        self.getProcessedFlat([ad for ad in adinputs if not is_flat])
        flat_list = [None if this_is_flat else self._get_cal(ad, 'processed_flat')
                     for ad, this_is_flat in zip(adinputs, is_flat)]

        for ad, flat in zip(*gt.make_lists(adinputs, flat_list, force_ad=True)):
            if len(ad) > 1:
                log.warning(f"{ad.filename} has more than one extension - ignoring")
                continue
            arm = ad.arm()
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            # Finer sampling is required vertically to ensure there remains
            # data between the orders
            xsampling, ysampling = 16, max(ybin, 4)
            xrebin, yrebin = xsampling // xbin, ysampling // ybin
            if xrebin * yrebin > 1:
                ad_binned = self._rebin_ghost_ad(deepcopy(ad), xsampling, ysampling)
            else:
                ad_binned = ad
            if flat is None:
                illuminated_pixels = ad_binned[0].PIXELMODEL > 0  # will have been rebinned
            else:
                illuminated_pixels = flat[0].PIXELMODEL > 0
                illuminated_pixels = np.logical_or.reduce(np.logical_or.reduce(
                    illuminated_pixels.reshape(
                        illuminated_pixels.shape[0] // ysampling, ysampling,
                        illuminated_pixels.shape[1] // xsampling, xsampling),
                    axis=1), axis=2)

            # Keep bad pixels out of the fit
            if ad_binned.mask is not None:
                illuminated_pixels |= ad_binned.mask.astype(bool)

            # Mark all pixels outside the topmost/bottommost orders as
            # illuminated (i.e., do not use in the fit)
            for ix, iy in enumerate(illuminated_pixels.argmax(axis=0)):
                illuminated_pixels[:iy, ix] = True
            for ix, iy in enumerate(illuminated_pixels[::-1].argmax(axis=0)):
                if iy > 0:
                    illuminated_pixels[-iy:, ix] = True
                    if arm == "red":
                        illuminated_pixels[-iy-768//ysampling:, ix] = True

            # Reinstate some pixels to constrain the spline
            if arm == "red":
                illuminated_pixels[:256 // ysampling] = False

            xpts = (np.arange(illuminated_pixels.shape[1]) + 0.5) * xrebin - 0.5
            ypts = (np.arange(illuminated_pixels.shape[0]) + 0.5) * yrebin - 0.5
            y, x = np.meshgrid(ypts, xpts, indexing="ij")
            w = 1 / np.sqrt(ad_binned[0].variance[~illuminated_pixels])
            zpts = ad_binned[0].data[~illuminated_pixels]
            #if arm == "red":
            #    zpts[y[~illuminated_pixels] > 4096 // ysampling] = 0
            #    w[y[~illuminated_pixels] > 4096 // ysampling] = 2
            spl = interpolate.SmoothBivariateSpline(
                x[~illuminated_pixels], y[~illuminated_pixels],
                zpts, w=w, bbox=[0, ad[0].shape[1]-1, 0, ad[0].shape[0]-1],
                s=smoothness*np.sum(~illuminated_pixels))
            scattered_light = spl(np.arange(ad[0].shape[1]),
                                  np.arange(ad[0].shape[0]), grid=True).T / (xrebin * yrebin)
            scattered_light[scattered_light < 0] = 0
            if save_model:
                ad[0].INPUT = ad_binned[0].data.copy()
                ad[0].INPUT[illuminated_pixels] = 0
                scat_bin = spl(xpts, ypts, grid=True).T
                ad[0].SCATBIN = scat_bin
                ad[0].SCATTERED = scattered_light
                ad[0].ILLUM = illuminated_pixels.astype(int)
            ad[0].subtract(scattered_light)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def responseCorrect(self, adinputs=None, **params):
        """
        Use a standard star observation and reference spectrum to provide
        absolute flux calibration.

        This primitive follows the basic pattern for determining absolute flux
        from an observed standard with a relative flux scale (e.g. counts) and
        an absolute flux-calibrated reference spectrum:

        - Dividing the standard star observation (in counts or electrons per
          pixel) by
          the exposure time (in s), and then by the standard star reference
          spectrum (in some unit of flux, e.g. erg/cm:math:`^2`/s/A) gives a
          sensitivity curve in units of, in this example, counts / erg.
        - Dividing the object spectrum by the exposure time (i.e. converting 
          to counts per pixel per second) by the sensitivity curve
          (counts / flux unit) yields the object spectrum in the original flux
          units of the standard star reference spectrum.

        Parameters
        ----------
        skip: bool
            If True, this primitive will just return the adinputs immediately
        standard : str, giving a relative or absolute file path
            The name of the reduced standard star observation. Defaults to
            None, at which point a ValueError is thrown.
        specphot_file: str, giving a relative or absolute file path
            The name of the file where the standard star spectrum (the
            reference, not the observed one) is stored. Defaults to None,
            in which case a Gemini-provided file will be searched for based
            on the object() descriptor.
        units: str, the flux density units of the output file
        order: int, polynomial order for the fit to each echelle order
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        std_filename = params['standard']
        specphot_file = params['specphot_file']
        final_units = params["units"]
        poly_degree = params["order"]
        debug_plots = params["debug_plots"]

        if std_filename is None:
            log.warning("No standard star observation has been provided, so "
                        f"{self.myself()} cannot calibrate the input spectra.")
            return adinputs

        # Let the astrodata routine handle any issues with actually opening
        # the FITS file
        ad_std = astrodata.open(std_filename)

        # Code duplication from Spect.calculateSensitivity()
        if specphot_file is None:
            obj_filename = f"{ad_std.object().lower().replace(' ', '')}.dat"
            for module in (self.inst_lookups, 'geminidr.gemini.lookups',
                           'geminidr.core.lookups'):
                try:
                    path = import_module('.', module).__path__[0]
                except (ImportError, ModuleNotFoundError):
                    continue
                full_path = os.path.join(path, 'spectrophotometric_standards',
                                         obj_filename)
                try:
                    spec_table = Spect._get_spectrophotometry(
                        self, full_path, in_vacuo=True)
                except (FileNotFoundError, InconsistentTableError):
                    pass
                else:
                    log.stdinfo(f"Read spectrophotometry file {full_path}")
                    break
            else:
                raise RuntimeError("Cannot find/read spectrophotometric data "
                                   f"table {obj_filename}")
        else:
            # Let this crash
            spec_table = Spect._get_spectrophotometry(self, specphot_file,
                                                      in_vacuo=True)

        # Re-grid the standard reference spectrum onto the wavelength grid of
        # the observed standard
        regrid_std_ref = np.zeros(ad_std[0].data.shape[:-1])
        for od in range(ad_std[0].data.shape[0]):
            regrid_std_ref[od] = self._regrid_spect(
                spec_table['FLUX'].value,
                spec_table['WAVELENGTH_AIR'].to(u.AA).value,
                ad_std[0].WAVL[od, :],
                waveunits='angstrom'
            )

        regrid_std_ref = (regrid_std_ref * spec_table['FLUX'].unit).to(
            final_units, equivalencies=u.spectral_density(ad_std[0].WAVL * u.AA)
        ).value

        # Figure out which object is actually the standard observation
        # (i.e. of the dimensions [order, wavl, object], figure which of the
        # three objects is actually the spectrum (another will be sky, and
        # the third probably empty)
        #objn = targetn_dict.targetn_dict['object']
        #target = -1
        #if ad_std.phu['TARGET1'] == objn: target = 0
        #if ad_std.phu['TARGET2'] == objn: target = 1
        #if target < 0:
        #    raise ValueError(
        #        'Cannot determine which IFU contains standard star spectrum.'
        #    )
        target = 0  # according to new extractProfile() behaviour

        # Compute the sensitivity function
        scaling_fac = regrid_std_ref * ad_std.exposure_time()
        with warnings.catch_warnings():  # division by zero
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sens_func = ad_std[0].data[:, :, target] / scaling_fac
            sens_func_var = ad_std[0].variance[:, :, target] / scaling_fac ** 2

        # MCW 180501
        # The sensitivity function requires significant smoothing in order to
        # prevent noise from the standard being transmitted into the data
        # The easiest option is to perform a parabolic curve fit to each order
        # CJS: Remember that data have been "flatfielded" so overall response
        # of each order has been removed already!
        # import pdb; pdb.set_trace();
        sens_func_fits = []
        good = ~np.logical_or(regrid_std_ref == 0, ad_std[0].variance[:, :, target] == 0)
        # Try to mask out deep atmospheric absorption features and unreasonable
        # measurements from low counts at the extreme orders
        ratio = sens_func / np.percentile(sens_func, 60, axis=1)[:, np.newaxis]
        good &= np.logical_and(ratio >= 0.2, ratio <= 5)
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(), sigma_clip,
                                                   sigma_lower=2, maxiters=10)
        plt.ioff()
        for od in range(sens_func.shape[0]):
            good_order = good[od]
            wavelengths = ad_std[0].WAVL[od]
            min_wave, max_wave = wavelengths.min(), wavelengths.max()
            if good_order.sum() > poly_degree:
                m_init = models.Chebyshev1D(
                    degree=poly_degree, c0=sens_func[od, good_order].mean(),
                    domain=[min_wave, max_wave])
                m_final, mask = fit_it(m_init, wavelengths[good_order],
                                       sens_func[od, good_order],
                                       weights=1. / np.sqrt(sens_func_var[od, good_order]))
                if debug_plots:
                    fig, ax = plt.subplots()
                    ax.plot(ad_std[0].WAVL[od], sens_func[od], 'k-')
                    ax.plot(ad_std[0].WAVL[od, good_order][~mask],
                            sens_func[od, good_order][~mask], 'r-')
                    ax.plot(ad_std[0].WAVL[od], m_final(ad_std[0].WAVL[od]), 'b-')
                    plt.show()
                rms = np.std((m_final(wavelengths) - sens_func[od])[good_order][~mask])
                expected_rms = np.median(np.sqrt(sens_func_var[od, good_order]))
                #if rms > 2 * expected_rms:
                #    log.warning(f"Unexpectedly high rms for row {od} "
                #                f"({min_wave:.1f} - {max_wave:.1f} A)")
                sens_func_fits.append(m_final)
            else:
                log.warning(f"Cannot determine sensitivity for row {od} "
                            f"({min_wave:.1f} - {max_wave:.1f} A)")
                if debug_plots:
                    fig, ax = plt.subplots()
                    ax.plot(ad_std[0].WAVL[od], sens_func[od], 'k-')
                    plt.show()
                sens_func_fits.append(models.Const1D(np.inf))
        plt.ion()

        # import pdb; pdb.set_trace();

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by responseCorrect".
                            format(ad.filename))
                continue

            # Check that the ad matches the standard
            if ad.res_mode() != ad_std.res_mode():
                raise ValueError('Resolution modes do not match for '
                                 '{} and {}'.format(ad.filename, ad_std.filename))
            if ad.arm() != ad_std.arm():
                raise ValueError('Spectrograph arms do not match for '
                                 '{} and {}'.format(ad.filename, ad_std.filename))
            if ad.detector_y_bin() != ad_std.detector_y_bin() or \
                    ad.detector_x_bin() != ad_std.detector_x_bin():
                raise ValueError('Binning does not match for '
                                 '{} and {}'.format(ad.filename, ad_std.filename))

            # Easiest way to response correct is to stand up a new AstroData
            # instance containing the sensitivity function - this will
            # automatically handle, e.g., the VAR re-calculation
            sens_func_ad = deepcopy(ad)
            sens_func_ad.update_filename(suffix='_sensFunc', strip=True)

            # Do the response correction -- can't just ad.divide(sens_func_ad)
            # because overflows mess up the variance plane
            for i, ext in enumerate(ad):
                sens_func_regrid = np.empty_like(ext.data)
                for od, sensfunc in enumerate(sens_func_fits):
                    sens_func_regrid[od] = sensfunc(ext.WAVL[od])[:, np.newaxis]
                sens_func_ad[i].data = sens_func_regrid * ad.exposure_time()
                sens_func_ad[i].variance = None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ad[i].data /= sens_func_ad[i].data
                    # has overflows if we square sens_func_data first!
                    ad[i].variance /= sens_func_ad[i].data
                    ad[i].variance /= sens_func_ad[i].data

            # Make the relevant header update
            ad.hdr['BUNIT'] = final_units

            # Now that we've made the correction, remove the superfluous
            # extra dimension from sens_func_ad and write out, if requested
            if params['write_result']:
                for ext in sens_func_ad:
                    ext.data = ext.data[:, :, 0]
                    try:
                        del ext.WGT
                    except AttributeError:
                        pass
                sens_func_ad.write(overwrite=True)
                ad.phu.set("SENSFUNC", os.path.abspath(sens_func_ad.path),
                           self.keyword_comments['SENSFUNC'])
            # sens_func_ad.reset(sens_func_regrid,
            # variance=sens_func_regrid_var)

            # Timestamp & suffix updates
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def stackArcs(self, adinputs=None, **params):
        """
        This primitive stacks input arc frames by associating arcs taken in
        close temporal proximity, and stacking them together.

        This primitive works by 'clustering' the input arcs, and then calling
        the standard stackFrames primitive on each cluster in turn.

        Parameters
        ----------
        time_delta : float
            The time delta between arcs that will allow for association,
            expressed in minutes. Note that this time_delta is between
            two arcs in sequence; e.g., if time_delta is 20 minutes, and arcs
            are taken with the following gaps:
            A <- 19m -> B <- 10m -> C <- 30m -> D <- 19m -> E
            These arcs will be clustered as follows:
            [A, B, C] and [D, E]
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = "STCKARCS"  # FIXME - Needs to go into timestamp_keywords

        if params['skip']:
            log.stdinfo('Skipping the specialized arc stacking step')
            return adinputs

        time_delta = params['time_delta']
        stack_params = self._inherit_params(params, "stackFrames")

        # import pdb; pdb.set_trace()

        processed = [ad.filename for ad in adinputs if timestamp_key in ad.phu]
        if processed:
            raise RuntimeError("The following frames have already been "
                               "processed by stackArcs\n    " +
                               "\n    ".join(processed))

        clusters = []
        if time_delta is None:
            parent_names = [ad.phu['ORIGNAME'].split('.')[0][:-3]
                            for ad in adinputs]
            # Cluster by bundle (i.e. ORIGNAME')
            for bundle in set(parent_names):
                clusters.append([ad for ad, parent in zip(adinputs, parent_names)
                                 if parent == bundle])
        else:
            # Sort the input arc files by DATE-OBS/UTSTART
            adinputs.sort(key=lambda x: _construct_datetime(x.phu))

            # Cluster the inputs
            for ad in adinputs:
                try:
                    ref_time = _construct_datetime(clusters[-1][-1].phu)
                except IndexError:
                    # Start a new cluster
                    clusters.append([ad, ])
                    continue

                if np.abs((_construct_datetime(ad.phu) - ref_time).total_seconds()
                          < time_delta):
                    # Append to the last cluster
                    clusters[-1].append(ad)
                else:
                    # Start a new cluster
                    clusters.append([ad, ])

        # Stack each cluster
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                raise RuntimeError("I've ended up with an empty cluster...")

            # Don't need to touch a cluster with only 1 AD
            if len(cluster) > 1:
                clusters[i] = self.stackFrames(adinputs=cluster, **stack_params)

        # Flatten out the list
        clusters_flat = [item for sublist in clusters for item in sublist]

        # Book keeping
        for i, ad in enumerate(clusters_flat):
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

        return clusters_flat

    def standardizeStructure(self, adinputs=None, **params):
        """
        The Gemini-level version of this primitive
        will try to attach an MDF because a GHOST image is
        tagged as SPECT. Rather than set parameters for that primitive to
        stop it from doing so, just override with a no-op primitive.
        
        .. note::
            This could go in primitives_ghost.py if the SLITV version
            also no-ops.
        """
        return adinputs

    def standardizeSpectralFormat(self, adinputs=None, suffix=None):
        """
        Convert the input file into multiple apertures, each with its own
        gWCS object to describe the wavelength scale. This allows the
        spectrum (or each order) to be displayed by the "dgsplot" script

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        input_frame = cf.CoordinateFrame(naxes=1, axes_type=['SPATIAL'],
                                         axes_order=(0,), name="pixels",
                                         axes_names=['x'], unit=u.pix)
        output_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                        axes_names=["WAVE"],
                                        name="Wavelength in vacuo")

        wave_tol = 1e-4  # rms limit for a linear/loglinear wave model
        adoutputs = []
        for ad in adinputs:
            adout = astrodata.create(ad.phu)
            adout.update_filename(suffix=suffix, strip=True)
            log.stdinfo(f"Converting {ad.filename} to {adout.filename}")
            for i, ext in enumerate(ad, start=1):
                if not hasattr(ext, "WAVL"):
                    log.warning(f"    EXTVER {i} has no WAVL table. Ignoring.")
                    continue
                has_var = ext.variance is not None
                has_mask = ext.mask is not None
                npix, nobj = ext.shape[-2:]
                try:
                    orders = list(range(ext.shape[-3]))
                except:
                    orders = [None]
                wave_models = []
                for order in orders:
                    wave = 0.1 * ext.WAVL[order].ravel()  # because WAVL is always 2D
                    linear = np.diff(wave).std() < wave_tol
                    loglinear = (wave[1:] / wave[:-1]).std() < wave_tol
                    if linear or loglinear:
                        if linear:
                            log.debug(f"Linear wavelength solution found for order {order}")
                            fit_it = fitting.LinearLSQFitter()
                            m_init = models.Polynomial1D(
                                degree=1, c0=wave[0], c1=wave[1]-wave[0])
                        else:
                            log.debug(f"Loglinear wavelength solution found for order {order}")
                            fit_it = fitting.LevMarLSQFitter()
                            m_init = models.Exponential1D(
                                amplitude=wave[0], tau=1./np.log(wave[1] / wave[0]))
                        wave_model = fit_it(m_init, np.arange(npix), wave)
                        log.debug("    "+" ".join([f"{p}: {v}" for p, v in zip(
                            wave_model.param_names, wave_model.parameters)]))
                    else:
                        log.debug(f"Using tabular model for order {order}")
                        wave_model = Tabular1D(np.arange(npix), wave)
                    wave_model.name = "WAVE"
                    wave_models.append(wave_model)

                for spec in range(nobj):
                    for order, wave_model in zip(orders, wave_models):
                        ndd = ext.nddata.__class__(data=ext.data[order, :, spec].ravel(),
                                                   meta={'header': ext.hdr.copy()})
                        if has_mask:
                            ndd.mask = ext.mask[order, :, spec].ravel()
                        if has_var:
                            ndd.variance = ext.variance[order, :, spec].ravel()
                        adout.append(ndd)
                        adout[-1].hdr[ad._keyword_for('data_section')] = f"[1:{npix}]"
                        adout[-1].wcs = gWCS([(input_frame, wave_model),
                                              (output_frame, None)])

                log.stdinfo(f"    EXTVER {i} has been converted to EXTVERs "
                            f"{len(adout) - nobj * len(orders) + 1}-{len(adout)}")

            adoutputs.append(adout)

        return adoutputs

    def write1DSpectra(self, adinputs=None, **params):
        """
        Write 1D spectra to files listing the wavelength and data (and
        optionally variance and mask) in one of a range of possible formats.

        Because Spect.write1DSpectra requires APERTURE numbers, the GHOST
        version of this primitive adds them before calling the Spect version.

        Parameters
        ----------
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
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            for i, ext in enumerate(ad, start=1):
                ext.hdr['APERTURE'] = i
        Spect.write1DSpectra(self, adinputs, **params)
        return adinputs

    def createFITSWCS(self, adinputs=None, **params):
        """
        DRAGONS/AstroData v3.0.x does not write FITS keywords correctly for
        a log-linear wavelength scale. This primitive creates such files and
        writes them to disk, returning an empty adinputs list.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        iraf: bool
            write WCS in IRAF format?
        angstroms: bool
            write new wavelength keywords in Angstroms (rather than nm)?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        suffix = params["suffix"]
        iraf = params["iraf"]
        angstroms = params["angstroms"]

        for ad in adinputs:
            log.stdinfo(f"Processing {ad.filename}")
            hdulist = ad.to_hdulist()
            for ext, ver in zip(ad, ad.hdr['EXTVER']):
                if len(ext.shape) != 1:
                    log.stdinfo(f"    EXTVER {ver} has non-1D data... "
                                f"continuing")
                    continue
                m = ext.wcs.forward_transform
                if not isinstance(m, models.Exponential1D):
                    log.stdinfo(f"    EXTVER {ver}'s dispersion is not "
                                f"log-linear... continuing")
                    continue
                amp, tau = m.parameters
                if angstroms:
                    amp *= 10
                if iraf:
                    # IRAF/NOAO, according to Valdes (1993, ASP 52, 467)
                    dw = np.log10(m(1) / m(0))
                    new_kws = {"CTYPE1": "WAVE-LOG", "CRPIX1": 1,
                               "CRVAL1": np.log10(amp),
                               "CDELT1": dw, "CD1_1": dw, "DC-FLAG": 1}
                else:
                    # FITS standard, according to Eqn (5) of
                    # Greisen et al. (2006; A&A 446, 747)
                    dw = np.log(m(1) / m(0)) * amp
                    new_kws = {"CTYPE1": "WAVE-LOG", "CRPIX1": 1, "CRVAL1": amp,
                               "CDELT1": dw, "CD1_1": dw}
                if angstroms:
                    new_kws['CUNIT1'] = 'Angstrom'

                wcs_index = None
                for i, hdu in enumerate(hdulist):
                    if hdu.header.get('EXTVER') == ver:
                        if isinstance(hdu, fits.ImageHDU) and len(hdu.data.shape) == 1:
                            hdu.header.update(new_kws)
                            try:
                                del hdu.header['FITS-WCS']
                            except KeyError:
                                pass
                    elif hdu.header.get('EXTNAME') == "WCS":
                        wcs_index = i
                if wcs_index is not None:
                    del hdulist[wcs_index]

            ad.update_filename(suffix=suffix, strip=True)
            log.stdinfo(f"Writing {ad.filename}")
            hdulist.writeto(ad.filename, overwrite=True)

        return []


##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

    def _get_polyfit_filename(self, ad, caltype):
        """
        Gets the filename of the relevant initial polyfit file for this
        input GHOST science image

        This primitive uses the arm, resolution mode and observing epoch
        of the input AstroData object to determine the correct initial
        polyfit model to provide. The model provided matches the arm and
        resolution mode of the data, and is the most recent model generated
        before the observing epoch.

        Parameters
        ----------
        ad : :class:`astrodata.AstroData`
            AstroData object to return the relevant initial model filename for
        caltype : str
            The initial model type (e.g. ``'rotmod'``, ``'spatmod'``, etc.)
            requested. An :any:`AttributeError` will be raised if the requested
            model type does not exist.

        Returns
        -------
        str/None:
            Filename (including path) of the required polyfit file
        """
        return polyfit_lookup.get_polyfit_filename(self.log, ad.arm(),
                                                   ad.res_mode(), ad.ut_date(),
                                                   ad.filename, caltype)

    def _get_slitv_polyfit_filename(self, ad):
        return polyfit_lookup.get_polyfit_filename(self.log, 'slitv',
                                                   ad.res_mode(), ad.ut_date(),
                                                   ad.filename, 'slitvmod')

    def _request_bracket_arc(self, ad, before=None):
        """
        Request the 'before' or 'after' arc for the passed ad object.

        For maximum accuracy in wavelength calibration, GHOST data is calibrated
        the two arcs taken immediately before and after the exposure. However,
        the Gemini calibration system is not rigged to perform such logic (it
        can return multipled arcs, but cannot guarantee that they straddle
        the observation in time).

        This helper function works by doing the following:

        - Append a special header keyword, 'ARCBEFOR', to the PHU. This keyword
          will be True if a before arc is requested, or False if an after arc
          is wanted.
        - getProcessedArc is the invoked, followed by the _get_cal call. The
          arc calibration association rules will see the special descriptor
          related to the 'ARCBEFOR' header keyword, and fetch an arc
          accordingly.
        - The header keyword is then deleted from the ad object, returning it
          to its original state.

        Parameters
        ----------
        before : bool
            Denotes whether to ask for the most recent arc before (True) or
            after (False) the input AD was taken. Defaults to None, at which
            point :any:`ValueError` will be thrown.

        Returns
        -------
        arc_ad : astrodata.AstroData instance (or None)
            The requested arc. Will return None if no suitable arc is found.
        """
        if before is None:
            raise ValueError('_request_bracket_arc requires that the before '
                             'kwarg is either True or False. If you wish to '
                             'do a "standard" arc calibration fetch, simply '
                             'use getProcessedArc directly.')

        ad.phu['ARCBEFOR'] = before
        self.getProcessedArc([ad,],
                             howmany=None,
                             refresh=True)
        arc_ad = self._get_cal(ad, 'processed_arc', )
        del ad.phu['ARCBEFOR']
        return arc_ad

    @staticmethod
    def _interp_spect(orig_data, orig_wavl, new_wavl,
                      interp='linear'):
        """
        'Re-grid' a one-dimensional input spectrum by performing simple
        interpolation on the data.

        This function performs simple linear interpolation between points
        on the old wavelength grid, moving the data onto the new
        wavelength grid. It makes no attempt to be, e.g., flux-conserving.

        The interpolation is performed by
        :any:`scipy.interpolate.interp1d <scipy.interpolate.interp1d>`.

        Parameters
        ----------
        orig_data : 1D numpy array or list
            The original spectrum data
        orig_wavl : 1D numpy array or list
            The corresponding wavelength values for the original spectrum data
        new_wavl : 1D numpy array or list
            The new wavelength values to re-grid the spectrum data to
        interp : str
            The interpolation method to be used. Defaults to 'linear'. Will
            accept any valid value of the ``kind`` argument to
            :any:`scipy.interpolate.interp1d`.

        Returns
        -------
        regrid_data : 1D numpy array
            The spectrum re-gridded onto the new_wavl wavelength points.
            Will have the same shape as new_wavl.
        """
        # Input checking
        orig_data = np.asarray(orig_data, dtype=orig_data.dtype)
        orig_wavl = np.asarray(orig_wavl, dtype=orig_wavl.dtype)
        new_wavl = np.asarray(new_wavl, dtype=new_wavl.dtype)

        if orig_data.shape != orig_wavl.shape:
            raise ValueError('_interp_spect received data and wavelength '
                             'arrays of different shapes')

        interp_func = interpolate.interp1d(
            orig_wavl,
            orig_data,
            kind=interp,
            fill_value=np.nan,
            bounds_error=False,
        )
        regrid_data = interp_func(new_wavl)
        # regrid_data = np.interp(new_wavl, orig_wavl, orig_data, )
        return regrid_data

    @staticmethod
    def _regrid_spect(orig_data, orig_wavl, new_wavl,
                      waveunits='angstrom'):
        """
        Re-grid a one-dimensional input spectrum so as to conserve total flux.

        This is a more robust procedure than :meth:`_interp_spect`, and is
        designed for data with a wavelength dependence in the data units
        (e.g. erg/cm^2/s/A or similar).

        This function utilises the :any:`pysynphot` package.

        This function has been adapted from:
        http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/

        Parameters
        ----------
        orig_data : 1D numpy array or list
            The original spectrum data
        orig_wavl : 1D numpy array or list
            The corresponding wavelength values for the original spectrum data
        new_wavl : 1D numpy array or list
            The new wavelength values to re-grid the spectrum data to
        waveunits : str
            The units of the wavelength scale. Defaults to 'angstrom'.

        Returns
        -------
        egrid_data : 1D numpy array
            The spectrum re-gridded onto the new_wavl wavelength points.
            Will have the same shape as new_wavl.
        """

        spec = spectrum.ArraySourceSpectrum(wave=orig_wavl, flux=orig_data)
        f = np.ones(orig_wavl.shape)
        filt = spectrum.ArraySpectralElement(orig_wavl, f, waveunits=waveunits)
        obs = observation.Observation(spec, filt, binset=new_wavl,
                                      force='taper')
        return obs.binflux


def _construct_datetime(hdr):
    """
    Construct a datetime object from DATE-OBS and UTSTART.
    """
    return datetime.combine(
        datetime.strptime(hdr.get('DATE-OBS'), '%Y-%m-%d').date(),
        datetime.strptime(hdr.get('UTSTART'), '%H:%M:%S').time(),
    )


def plot_extracted_spectra(ad, arm, all_peaks, lines_out, mask=None, nrows=4):
    """
    Produce plot of all the extracted orders, located peaks, and matched
    arc lines. Abstracted here to make the primitive cleaner.
    """
    if mask is None:
        mask = np.zeros(len(lines_out), dtype=bool)
    pixels = np.arange(arm.szy)
    with PdfPages(ad.filename.replace('.fits', '.pdf')) as pdf:
        for m_ix, (flux, peaks) in enumerate(zip(ad[0].data[:, :, 0], all_peaks)):
            order = m_ix + arm.m_min
            if m_ix % nrows == 0:
                if m_ix > 0:
                    fig.subplots_adjust(hspace=0)
                    pdf.savefig(bbox_inches='tight')
                fig, axes = plt.subplots(nrows, 1, sharex=True)
            ax = axes[m_ix % nrows]
            ax.plot(pixels, flux, color='darkgrey', linestyle='-', linewidth=1)
            ymax = flux.max()
            for g in peaks:
                x = g.mean.value + np.arange(-3, 3.01, 0.1) * g.stddev.value
                ax.plot(x, g(x), 'k-', linewidth=1)
                ymax = max(ymax, g.amplitude.value)
            for (wave, yval, xval, ord, amp, fwhm), m in zip(lines_out, mask):
                if ord == order:
                    kwargs = {"fontsize": 5, "rotation": 90,
                              "color": "red" if m else "blue"}
                    if amp > 0.5 * ymax:
                        ax.text(yval, amp, str(wave), verticalalignment='top',
                                **kwargs)
                    else:
                        ax.text(yval, ymax*0.05, str(wave), **kwargs)
            ax.set_ylim(0, ymax * 1.05)
            ax.set_xlim(0, arm.szy - 1)
            ax.text(20, ymax * 0.8, f'order {order}')
            ax.set_yticks([])  # we don't care about y scale
        for i in range(m_ix % nrows + 1, 4):
            axes[i].axis('off')
        fig.subplots_adjust(hspace=0)
        pdf.savefig(bbox_inches='tight')


def model_scattered_light(data, mask=None, variance=None):
    """
    Model scattered light by fitting a surface to the 2D image. The fit is
    performed as the exponential of a 2D polynomial to ensure it is always
    non-negative.

    Parameters
    ----------
    data: `numpy.ndarray`
        data which requires a surface fit
    mask: `numpy.ndarray` (bool)/None
        mask indicating which pixels to use in the fit

    Returns
    -------
    model: `numpy.ndarray`
        a model fit to the data
    """
    from datetime import datetime
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    if variance is None:
        w = np.ones_like(data)
    else:
        w = np.sqrt(at.divide0(1., variance))
    start = datetime.now()
    tck = interpolate.bisplrep(x[~mask].ravel()[::32], y[~mask].ravel()[::32], data[~mask].ravel()[::32], w=w[~mask].ravel()[::32])
    print(datetime.now() - start)
    spl = interpolate.bisplev(np.arange(nx), np.arange(ny), tck)
    print(datetime.now() - start)
    return spl
