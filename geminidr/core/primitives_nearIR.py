#
#                                                                  gemini_python
#
#                                                           primitives_nearIR.py
# ------------------------------------------------------------------------------
import datetime
from functools import partial
from itertools import product as cart_product
import warnings

from astropy.stats import sigma_clip
import numpy as np

from gempy.gemini import gemini_tools as gt
from gempy.library.nddops import NDStacker

from geminidr import PrimitivesBASE
from geminidr.gemini.lookups import DQ_definitions as DQ
from . import parameters_nearIR, Bookkeeping

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------

@parameter_override
@capture_provenance
class NearIR(Bookkeeping):
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_nearIR)

    def addLatencyToDQ(self, adinputs=None, **params):
        """
        Flags pixels in the DQ plane of an image based on whether the same
        pixel has been flagged as saturated in a previous image.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        non_linear : bool
            flag non-linear pixels (as well as saturated ones)?
        time: float
            time (in seconds) for which latency is an issue
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        flags = DQ.saturated | (DQ.non_linear if params["non_linear"] else 0)
        # Create a timedelta object using the value of the "time" parameter
        seconds = datetime.timedelta(seconds=params["time"])

        # Avoids n^2 calls to the descriptor
        times = [ad.ut_datetime() for ad in adinputs]
        for i, ad in enumerate(adinputs):
            # Find which frames have their bright pixels propagated
            propagated = [x for x in zip(adinputs, times) if (x[1]<times[i] and times[i]-x[1]<seconds)]
            if propagated:
                log.stdinfo('{} affected by {}'.format(ad.filename,
                                    ','.join([x[0].filename for x in propagated])))

                for ad_latent in list(zip(*propagated))[0]:
                    # AD extensions might not be in the same order
                    # Set aux_type to 'bpm' which means hot pixels in a subarray
                    # can still be propagated to a subsequent full-array image
                    ad_latent = gt.clip_auxiliary_data(ad, aux=ad_latent,
                                                       aux_type='bpm')
                    for ext, ext_latent in zip(ad, ad_latent):
                        if ext_latent.mask is not None:
                            latency = np.where(ext_latent.mask & flags, DQ.cosmic_ray,
                                            0).astype(DQ.datatype)
                            ext.mask = latency if ext.mask is None \
                                else ext.mask | latency
            else:
                log.stdinfo('{} is not affected by latency'.format(ad.filename))

            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def makeBPM(self, adinputs=None, **params):
        """
        To be run from recipe makeProcessedBPM.

        The main input is a flat field image that has been constructed by
        stacking the differences of lamp on / off exposures in a given filter
        and normalizing the resulting image to unit average.

        A 'darks' stream must also be provided, containing a single image
        constructed by stacking short darks.

        Parameters
        ----------
        dark_lo_thresh, dark_hi_thresh: float, optional
            Range of data values (always in ADUs) outside which pixels in the
            input dark are considered bad (eg. -20 and 100, but these defaults
            vary by instrument). A limit of None is not applied and all pixels
            are considered good at that end of the range.
        flat_lo_thresh, flat_hi_thresh: float, optional
            Range of unit-normalized data values outside which pixels in the
            input flat are considered bad (eg. 0.8 and 1.25, but these defaults
            vary by instrument). A limit of None is not applied and all pixels
            are considered good at that end of the range.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # This has been adapted to do almost the same as niflat in IRAF; it
        # could most likely be improved upon, but produces reasonable results,
        # whereas the original version wasn't deriving good thresholds.

        dark_lo = params['dark_lo_thresh']
        dark_hi = params['dark_hi_thresh']
        flat_lo = params['flat_lo_thresh']
        flat_hi = params['flat_hi_thresh']

        if dark_lo is None:
            dark_lo = float('-Inf')
        if dark_hi is None:
            dark_hi = float('Inf')
        if flat_lo is None:
            flat_lo = float('-Inf')
        if flat_hi is None:
            flat_hi = float('Inf')

        # This could probably be improved by using an input DQ mask (which
        # currently isn't produced by the recipe)?

        # Get the stacked flat and dark; these are single-element lists
        try:
            flat = adinputs[0]
        except IndexError:
            raise OSError("A SET OF FLATS IS REQUIRED INPUT")
        try:
            dark = self.streams['darks'][0]
        except (KeyError, TypeError, IndexError):
            raise OSError("A SET OF DARKS IS REQUIRED INPUT")

        for dark_ext, flat_ext in zip(dark, flat):
            msg = "BPM Flat Mask Lower < > Upper Limit: {} < > {}"
            log.stdinfo(msg.format(flat_lo, flat_hi))
            flat_mask = np.ma.masked_outside(flat_ext.data, flat_lo, flat_hi)

            msg = "BPM Dark Mask Lower < > Upper Limit: {} < > {} ADU\n" \
                  "                                    ({} < > {})"
            bunit = dark_ext.hdr.get('BUNIT', 'ADU').upper()
            if bunit in ('ELECTRON', 'ELECTRONS'):
                conv = dark_ext.gain()
            elif bunit == 'ADU':
                conv = 1
            else:
                raise ValueError("Input units for dark should be ADU or "
                                 "ELECTRON, not {}".format(bunit))
            log.stdinfo(msg.format(dark_lo, dark_hi,
                                   conv*dark_lo, conv*dark_hi))
            # create the mask -- darks (hot pixels)
            dark_mask = np.ma.masked_outside(dark_ext.data,
                                             conv*dark_lo, conv*dark_hi)

            # combine masks and write to bpm file
            data_mask = np.ma.mask_or(np.ma.getmaskarray(dark_mask),
                                      np.ma.getmaskarray(flat_mask),
                                      shrink=False)
            flat_ext.reset(data_mask.astype(np.int16), mask=None, variance=None)

        flat.update_filename(suffix="_bpm", strip=True)
        flat.phu.set('OBJECT', 'BPM')
        gt.mark_history(flat, primname=self.myself(), keyword=timestamp_key)
        return [flat]

    def makeLampFlat(self, adinputs=None, **params):
        """
        This separates the lamp-on and lamp-off flats, stacks them, subtracts
        one from the other, and returns that single frame. Since they are lamp
        flats, no scaling is performed during the stacking.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        suffix = params["suffix"]

        lamp_on_list = self.selectFromInputs(adinputs, tags='LAMPON')
        lamp_off_list = self.selectFromInputs(adinputs, tags='LAMPOFF')
        self.showInputs(lamp_on_list, purpose='lampOn')
        self.showInputs(lamp_off_list, purpose='lampOff')

        stack_params = self._inherit_params(params, "stackFrames")
        stack_params.update({'zero': False, 'scale': False})
        ad_on = self.stackFrames(lamp_on_list, **stack_params)
        ad_off = self.stackFrames(lamp_off_list, **stack_params)

        if ad_on and ad_off:
            if ad_on[0].exposure_time() != ad_off[0].exposure_time():
                log.warning("Lamp-on and lamp-off flats do not have the same exposure time.")
            log.fullinfo("Subtracting lampOff stack from lampOn stack")
            flat = ad_on[0]
            flat.subtract(ad_off[0])
            flat.update_filename(suffix=suffix, strip=True)
            return [flat]
        else:
            log.warning("Cannot subtract lampOff from lampOn flats as do not "
                        "have some of each")
            if ad_on:
                log.warning("Returning stacked lamp on flats")
                return ad_on
            else:
                return []

    def removeFirstFrame(self, adinputs=None, **params):
        """
        This removes the first frame (according to timestamp) from the input
        list. It is intended for use with NIRI.

        Parameters
        ----------
        remove_first: bool
            remove the first frame? If False, this primitive no-ops (but logs
            a warning)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        remove_first = params["remove_first"]
        remove_files = params["remove_files"]

        if remove_first:
            if adinputs:
                remove_ad = self.sortInputs(adinputs, descriptor="ut_datetime")[0]
                log.stdinfo(f"Removing {remove_ad.filename} as a first frame")
                adinputs = [ad for ad in adinputs if ad != remove_ad]
            else:
                log.stdinfo("No frames, nothing to remove")

        for f in (remove_files or []):
            f_root = f.replace('.fits', '')
            length = len(adinputs)
            adinputs = [ad for ad in adinputs if f_root not in ad.filename]
            if len(adinputs) == length:
                log.warning(f"Filename {f} cannot be removed as it is not in input list")
            else:
                log.stdinfo(f"Removing {f} as requested")

        if not (remove_first or remove_files):
            log.warning("No frames are being removed: data quality may suffer")
        return adinputs

    def removePatternNoise(self, adinputs=None, **params):
        """
        This attempts to remove the pattern noise in NIRI/GNIRS data. In each
        quadrant, boxes of a specified size are extracted and, for each pixel
        location in the box, the median across all the boxes is determined.
        The resultant median is then tiled to the size of the quadrant and
        subtracted. Optionally, the median of each box can be subtracted
        before performing the operation.

        Based on Andy Stephens's "cleanir"

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        force: bool
            perform operation even if standard deviation in quadrant increases?
        hsigma/lsigma: float
            sigma-clipping limits
        pattern_x_size: int
            size of pattern "box" in x direction
        pattern_y_size: int
            size of pattern "box" in y direction
        subtract_background: bool
            remove median of each "box" before calculating pattern noise?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        hsigma, lsigma = params["hsigma"], params["lsigma"]
        pxsize, pysize = params["pattern_x_size"], params["pattern_y_size"]
        bgsub = params["subtract_background"]
        force = params["force"]
        stack_function = NDStacker(combine='median', reject='sigclip',
                                   hsigma=hsigma, lsigma=lsigma)
        sigclip = partial(sigma_clip, sigma_lower=lsigma, sigma_upper=hsigma)
        zeros = None  # will remain unchanged if not subtract_background

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by removePatternNoise".
                            format(ad.filename))
                continue

            log.stdinfo(f"Applying pattern noise removal to {ad.filename}.")

            for ext in ad:

                if ad.instrument() == 'GNIRS':
                    # Number of rows must be 4-divisible, or it crashes.
                    log.debug("Padding GNIRS SCI, VAR, DQ y-axis by 2 rows to "
                              "be divisible by 4.")
                    log.debug("Original image shape:\n"
                              f"  SCI: {ext.data.shape}\n"
                              f"  VAR: {ext.variance.shape}\n"
                              f"   DQ: {ext.mask.shape}")
                    ext.data = np.append(
                        ext.data,
                        np.zeros((2, ext.shape[1])),
                        axis=0)
                    ext.variance = np.append(
                        ext.variance,
                        np.zeros((2, ext.shape[1])),
                        axis=0)
                    ext.mask = np.append(
                        ext.mask,
                        np.full((2, ext.shape[1]), DQ.no_data),  # Mask rows
                        axis=0)
                    log.debug("New image shape:\n"
                              f"  SCI: {ext.data.shape}\n"
                              f"  VAR: {ext.variance.shape}\n"
                              f"   DQ: {ext.mask.shape}")

                qysize, qxsize = [size // 2 for size in ext.data.shape]
                yticks = [(y, y + pysize) for y in range(0, qysize, pysize)]
                xticks = [(x, x + pxsize) for x in range(0, qxsize, pxsize)]
                for ystart in (0, qysize):
                    for xstart in (0, qxsize):
                        quad = ext.nddata[ystart:ystart + qysize,
                                          xstart:xstart + qxsize]
                        sigma_in = sigclip(np.ma.masked_array(quad.data,
                                                              quad.mask)).std()
                        blocks = [quad[tuple(slice(start, end)
                                             for (start, end) in coords)]
                                  for coords in cart_product(yticks, xticks)]
                        if bgsub:
                            # If all pixels are masked in a box, we'll get no
                            # result from the mean. Suppress warning.
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore",
                                                      category=UserWarning)
                                zeros = np.nan_to_num([-np.ma.masked_array(
                                                            block.data,
                                                            block.mask).mean()
                                                       for block in blocks])
                        out = stack_function(blocks, zero=zeros).data
                        out_quad = (quad.data + np.mean(out) -
                                    np.tile(out, (len(yticks), len(xticks))))
                        sigma_out = sigclip(np.ma.masked_array(out_quad,
                                                               quad.mask)).std()
                        if sigma_out > sigma_in:
                            qstr = (f"{ad.filename} extension {ext.id} "
                                    f"quadrant ({xstart},{ystart})")
                            if force:
                                log.stdinfo("Forcing cleaning on " + qstr)
                            else:
                                log.stdinfo("No improvement for "+qstr)
                                continue
                        ext.data[ystart:ystart + qysize,
                                 xstart:xstart + qxsize] = out_quad

                if ad.instrument() == 'GNIRS':
                    # Remove padding before returning
                    log.debug('Removing padding from GNIRS SCI, VAR, DQ y-axis.')
                    # Delete last 2 rows
                    ext.data = np.delete(ext.data, [-2, -1], axis=0)
                    ext.variance = np.delete(ext.variance, [-2, -1], axis=0)
                    ext.mask = np.delete(ext.mask, [-2, -1], axis=0)
                    log.debug(f"  SCI: {ext.data.shape}")
                    log.debug(f"  VAR: {ext.variance.shape}")
                    log.debug(f"   DQ: {ext.mask.shape}")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def separateFlatsDarks(self, adinputs=None, **params):
        """
        This primitive produces two streams, one containing flats, and one
        containing darks. Other files remain in the main stream
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Initialize lists of AstroData objects to be added to the streams
        dark_list = []
        flat_list = []
        adoutputs = []
        for ad in adinputs:
            tags = ad.tags
            if "DARK" in tags:
                dark_list.append(ad)
                log.fullinfo("Dark: {}, {}".format(ad.data_label(), ad.filename))
            elif "FLAT" in tags:
                flat_list.append(ad)
                log.fullinfo("Flat: {}, {}".format(ad.data_label(), ad.filename))
            else:
                adoutputs.append(ad)
                log.warning("Not Dark/Flat: {} {}".format(ad.data_label(),
                                                          ad.filename))
        if not dark_list:
            log.warning("No Darks in input list")
        if not flat_list:
            log.warning("No Flats in input list")

        self.streams.update({"flats" : flat_list})
        self.streams.update({"darks" : dark_list})
        return adoutputs
