#
#                                                                  gemini_python
#
#                                               primitives_gmos_nodandshuffle.py
#
# NB This is a pure mixin and should not be instantiated as a primitives class!
# ------------------------------------------------------------------------------
import numpy as np
from copy import copy, deepcopy

from astropy import units as u
from astropy.modeling import models
from gwcs import coordinate_frames as cf

from astrodata.provenance import add_provenance
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am, astrotools as at
from gempy.library import transform

from geminidr.gemini.lookups import DQ_definitions as DQ
from .primitives_gmos import GMOS
from . import parameters_gmos_nodandshuffle

from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.utils.md5 import md5sum
# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GMOSNodAndShuffle(GMOS):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above.
    """
    tagset = set()

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gmos_nodandshuffle)

    def combineNodAndShuffleBeams(self, adinputs=None, **params):
        """
        This primitive takes spectral images which have gone through
        skyCorrectNodAndShuffle and, if these images contain both positive
        and negative beams (based on the nod distance), it makes a shifted
        copy of the image and subtracts it from the original so that the
        positive trace contains the signal from both beams.

        If the nod distance is larger than the shuffle distance (i.e., the
        B beam is purely sky), then no operation is performed (since the
        sky has already been subtracted).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        align_sources: bool
            attempt to find alignment between the beams using the profile
            along the slit?
        region: str / None
            pixel region for determining slit profile for cross-correlation
        tolerance: float
            Maximum distance from the header offset, for the correlation
            method (arcsec). If the correlation computed offset is too
            different from the header offset, then the latter is used.
        order: int
            order of polynomial for resampling
        subsample: int
            output pixel subsampling when resampling
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        align_sources = params["align_sources"]
        region = params["region"]
        tolerance = params["tolerance"]
        order = params["order"]
        subsample = params["subsample"]
        dq_threshold = params["dq_threshold"]

        for ad in adinputs:
            nod_arcsec = np.diff(ad.nod_offsets())[0]
            nod_pixels = nod_arcsec / ad.pixel_scale()

            # Determine whether the B beam is on-source or on-sky
            if abs(nod_pixels) < ad.shuffle_pixels() * ad.detector_y_bin():
                if self.timestamp_keys["skyCorrectNodAndShuffle"] not in ad.phu:
                    log.warning(f"{ad.filename} has not been processed by "
                                f"skyCorrectNodAndShuffle so {self.myself} "
                                "cannot be run.")
                    continue

                beamshift = models.Shift(0) & models.Shift(nod_pixels)

                if align_sources and len(ad) == 1:
                    ad2 = ad * -1
                    ad2[0].wcs.insert_transform(ad[0].wcs.input_frame,
                                                beamshift, after=True)

                    # This is a bit hacky. The sign of QOFFSET for GMOS-S is
                    # flipped if it's on the bottom port. Rather than copy
                    # that logic (which might change in future), we update
                    # the keyword with the most likely value and check that
                    # we get the correct result
                    ad2.phu['QOFFSET'] = ad.phu['QOFFSET'] - nod_arcsec

                    if abs((ad.detector_y_offset() - ad2.detector_y_offset()) - \
                           nod_pixels) > 0.000001:
                        ad2.phu['QOFFSET'] = ad.phu['QOFFSET'] + nod_arcsec

                    ad2 = self.adjustWCSToReference(
                        [ad, ad2], method="sources_offsets", fallback="offsets",
                        tolerance=tolerance, region=region)[1]
                    beamshift = (models.Shift(0) &
                                 (am.get_named_submodel(ad2[0].wcs.forward_transform, 'SKY')[0]))
                    del ad2
                elif align_sources:
                    log.warning(f"{ad.filename} has multiple extensions "
                                "and source alignment cannot be used.")
                else:
                    log.stdinfo(f"{ad.filename}: Applying the nod offset of "
                                f"{nod_pixels:.2f} pixels")

                # replace with adwcs.pixel_frame once enh/multi_amp_scorpio is merged
                aligned_frame = cf.CoordinateFrame(
                    naxes=2, axes_type=['SPATIAL'] * 2, axes_order=(0, 1),
                    name="nod_aligned", axes_names=("x", "y"), unit=[u.pix] * 2)

                # There could be multiple extensions (e.g., MOS) and we don't
                # want to mosaic them, so send them to transform sequentially
                for ext in ad:
                    orig_wcs = copy(ext.wcs)
                    ext.wcs.insert_frame(ext.wcs.input_frame, beamshift,
                                         aligned_frame)
                    ad_out = transform.resample_from_wcs(
                        ext, 'nod_aligned', order=order, subsample=subsample,
                        parallel=False, output_shape=ext.shape, origin=(0,0),
                        threshold=dq_threshold)
                    ext.subtract(ad_out[0])
                    ext.wcs = orig_wcs
            else:
                log.stdinfo(f"{ad.filename} has a nod distance greater than "
                            "its shuffle distance, so no beam combining is "
                            "required.")

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def flatCorrect(self, adinputs=None, suffix=None, flat=None, do_cal=True):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        If no flatfield is provided, the calibration database(s) will be
        queried.

        If the flatfield has had a QE correction applied, this information is
        copied into the science header to avoid the correction being applied
        twice.

        This version differs from the Spect version of the primitive in that
        the flatfield is turned into a "double flat" by duplicating the
        exposed region and shifting it down by the required number of pixels
        so both the A and B regions of the N&S spectral image can be
        flatfielded before combining.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str/AstroData/None
            flatfield to be used (None => refer to calibration database)
        do_cal: str ("force", "skip", "procmode")
            determine whether to perform flatfielding, or whether it is a
            required step
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        qecorr_key = self.timestamp_keys['QECorrect']

        if do_cal == 'skip':
            log.warning("Flat correction has been turned off.")
            return adinputs

        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        last_shuffle = None
        # Provide a flat AD object for every science frame, and an origin
        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "flatCorrect. Continuing.")
                continue

            if flat is None:
                if 'sq' in self.mode or do_cal == 'force':
                   raise OSError("No processed flat listed for "
                                 f"{ad.filename}")
                else:
                   log.warning(f"No changes will be made to {ad.filename}, "
                               "since no flatfield has been specified")
                   continue

            # Check the inputs have matching filters, binning, and shapes
            try:
                gt.check_inputs_match(ad, flat)
            except ValueError:
                # Else try to clip the flat frame to the size of the science
                # data (e.g., for GMOS, this allows a full frame flat to
                # be used for a CCD2-only science frame.
                flat = gt.clip_auxiliary_data(adinput=ad,
                                    aux=flat, aux_type="cal")
                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, flat)

            # Do the division
            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: dividing by the flat "
                         f"{flat.filename}{origin_str}")

            shuffle_pixels = ad.shuffle_pixels() // ad.detector_y_bin()
            # No need to remake the double flat if it's the same file and
            # the same shuffle distance as the last one
            if shuffle_pixels != last_shuffle or flat.filename != double_flat.filename:
                log.stdinfo(f"Making a double flat from {flat.filename} "
                            f"by shifting {shuffle_pixels} pixels")
                double_flat = deepcopy(flat)
                for ext in double_flat:
                    ext.variance[ext.mask & DQ.unilluminated] = 0
                    ext.data[:-shuffle_pixels] *= ext.data[shuffle_pixels:]
                    ext.variance[:-shuffle_pixels] += ext.variance[shuffle_pixels:]
                    ext.mask[:-shuffle_pixels] = (
                        (ext.mask[:-shuffle_pixels] & ext.mask[shuffle_pixels:] & DQ.unilluminated) |
                        ((ext.mask[:-shuffle_pixels] | ext.mask[shuffle_pixels:]) & (DQ.max ^ DQ.unilluminated)))
                last_shuffle = shuffle_pixels
            else:
                log.stdinfo("Using previously-created double flat.")
            ad.divide(double_flat)
            double_flat.write("double_flat.fits", overwrite=True)

            # Update the header and filename, copying QECORR keyword from flat
            ad.phu.set("FLATIM", flat.filename, self.keyword_comments["FLATIM"])
            try:
                qecorr_value = flat.phu[qecorr_key]
            except KeyError:
                pass
            else:
                log.fullinfo("Copying {} keyword from flatfield".format(qecorr_key))
                ad.phu.set(qecorr_key, qecorr_value, flat.phu.comments[qecorr_key])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
            if flat.path:
                add_provenance(ad, flat.filename, md5sum(flat.path) or "", self.myself())

        return adinputs

    def skyCorrectNodAndShuffle(self, adinputs=None, suffix=None):
        """
        Perform sky correction on GMOS N&S images by taking each image and
        subtracting from it a shifted version of the same image.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            # Check whether the myScienceStep primitive has been run previously
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by skyCorrectNodShuffle".
                            format(ad.filename))
                continue

            # Determine N&S offset in (binned) pixels
            shuffle = ad.shuffle_pixels() // ad.detector_y_bin()
            a_nod_count, b_nod_count = ad.nod_count()

            ad_nodded = deepcopy(ad)

            # Shuffle B position data up for all extensions (SCI, DQ, VAR)
            for ext, ext_nodded in zip(ad, ad_nodded):
                #TODO: Add DQ=16 to top and bottom?
                # Set image initially to zero
                ext_nodded.multiply(0)
                # Then replace with the upward-shifted data
                for attr in ('data', 'mask', 'variance'):
                    getattr(ext_nodded, attr)[shuffle:] = getattr(ext,
                                                        attr)[:-shuffle]
                    ext_nodded.data[:shuffle] = 0
                    ext_nodded.mask[:shuffle] |= DQ.no_data
                    ext_nodded.variance[:shuffle] = 0

            # Normalize if the A and B nod counts differ
            if a_nod_count != b_nod_count:
                log.stdinfo("{} A and B nod counts differ...normalizing".
                            format(ad.filename))
                ad.multiply(0.5 * (a_nod_count + b_nod_count) / a_nod_count)
                ad_nodded.multiply(0.5 * (a_nod_count + b_nod_count) / b_nod_count)

            # Subtract nodded image from image to complete the process
            ad.subtract(ad_nodded)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
