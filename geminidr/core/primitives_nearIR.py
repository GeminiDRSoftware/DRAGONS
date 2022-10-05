#
#                                                                  gemini_python
#
#                                                           primitives_nearIR.py
# ------------------------------------------------------------------------------
from copy import deepcopy
import datetime
from functools import partial
from itertools import product as cart_product
import warnings

from astropy.stats import sigma_clip, sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt

from gempy.gemini import gemini_tools as gt
from gempy.library.nddops import NDStacker

from geminidr import PrimitivesBASE
from geminidr.gemini.lookups import DQ_definitions as DQ
from . import parameters_nearIR, Bookkeeping

from recipe_system.utils.decorators import parameter_override, capture_provenance

from scipy.optimize import minimize
from tqdm import tqdm

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
        suffix: str, Default: "_patternNoiseRemoved"
            Suffix to be added to output files.
        must_reduce_rms: bool, Default: True
            If True, will not apply pattern subtraction to each quadrant of the
            image if doing so would increase the RMS.
        hsigma/lsigma: float, Defaults: 3 for both
            High and low sigma-clipping limits.
        pattern_x_size: int, Default: 16
            Size of pattern "box" in x direction. Must be a multiple of 4.
        pattern_y_size: int, Default: 4
            size of pattern "box" in y direction. Must be a multiple of 4.
        subtract_background: bool, Default: True
            Remove median of each "box" before calculating pattern noise?
        region: str
            A user-specified region (or regions) to perform the cleaning in.
            Pattern noise will either cover the entire array, or will appear in
            one or more horizontal stripes, so these regions should be given as
            a string of y1:y2 values (ints), with multiple regions optionally
            separated by commas, e.g. '1:184,840:1024'. The default is to use
            the entire image. GNIRS data will be padded with 2 rows to make
            1024 in order to be divisible by 4.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        hsigma, lsigma = params["hsigma"], params["lsigma"]
        pxsize, pysize = params["pattern_x_size"], params["pattern_y_size"]
        bgsub = params["subtract_background"]
        must_reduce_rms = params["must_reduce_rms"]
        region = params["region"]
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
                    log.stdinfo("Padding GNIRS SCI, VAR, DQ y-axis by 2 rows to "
                                "be divisible by 4.")
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

                # Generate mask for applying cleaning, whole frame by default.
                region_mask = np.zeros_like(ext.data)

                # Slightly hacky way to store references to the four qudrants
                # for later leveling of their backgrounds.
                quads = {}
                quad_means = []
                quad_num = 0

                for roi in region.strip('[]').split(','):
                    r = roi.split(':')
                    if len(r) != 2:
                        raise ValueError('Regions must be in the form '
                                         f'y1:y2, passed: "{r}"')
                    y1 = int(r[0])
                    y2 = int(r[1])
                    log.stdinfo(f'Removing pattern noise from y={y1} to y={y2}')

                    region_mask[y1:y2, :] = 1

                qysize, qxsize = [size // 2 for size in ext.data.shape]
                yticks = [(y, y + pysize) for y in range(0, qysize, pysize)]
                xticks = [(x, x + pxsize) for x in range(0, qxsize, pxsize)]
                for ystart in (0, qysize):
                    for xstart in (0, qxsize):
                        quad = ext.nddata[ystart:ystart + qysize,
                                          xstart:xstart + qxsize]
                        quads[quad_num] = quad
                        quad_num += 1
                        quad_mask = region_mask[ystart:ystart + qysize,
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
                        pattern = np.tile(out, (len(yticks), len(xticks)))
                        out_quad = (quad.data + np.mean(out) -
                                    pattern * quad_mask)
                        sigma_clipped_out = sigclip(np.ma.masked_array(out_quad,
                                                                       quad.mask))
                        sigma_out = sigma_clipped_out.std()
                        quad_means.append(sigma_clipped_out.mean())
                        if sigma_out > sigma_in:
                            qstr = (f"{ad.filename} extension {ext.id} "
                                    f"quadrant ({xstart},{ystart})")
                            if not must_reduce_rms:
                                log.stdinfo("Forcing cleaning on " + qstr)
                            else:
                                log.stdinfo("No improvement for " + qstr +
                                            ", not applying pattern removal.")
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

            log.debug(f"The background level of the found quadrants is {quad_means}")
            arrays_median = np.median(quad_means)

            # Offset each quad by the difference between it and the overall
            # background.
            for quad_num, quad in quads.items():
                quad.data += arrays_median - quad_means[quad_num]

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def removePatternNoise_2(self, adinputs=None, **params):
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
        suffix: str, Default: "_patternNoiseRemoved"
            Suffix to be added to output files.
        must_reduce_rms: bool, Default: True
            If True, will not apply pattern subtraction to each quadrant of the
            image if doing so would increase the RMS.
        hsigma/lsigma: float, Defaults: 3 for both
            High and low sigma-clipping limits.
        pattern_x_size: int, Default: 16
            Size of pattern "box" in x direction. Must be a multiple of 4.
        pattern_y_size: int, Default: 4
            size of pattern "box" in y direction. Must be a multiple of 4.
        subtract_background: bool, Default: True
            Remove median of each "box" before calculating pattern noise?
        edge_threshold: float, Default: 10
            sigma threshold for automatically identifying edges of pattern coverage
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        hsigma, lsigma = params["hsigma"], params["lsigma"]
        pxsize, pysize = params["pattern_x_size"], params["pattern_y_size"]
        bgsub = params["subtract_background"]
        must_reduce_rms = params["must_reduce_rms"]
        stack_function = NDStacker(combine='median', reject='sigclip',
                                   hsigma=hsigma, lsigma=lsigma)
        sigclip = partial(sigma_clip, sigma_lower=lsigma, sigma_upper=hsigma)
        zeros = None  # will remain unchanged if not subtract_background
        edge_threshold = params["edge_threshold"]

        def pattern_func(x, block, pattern):
            squared_err = 0
            squared_err += np.sum(((np.ma.masked_array(block.data, block.mask)) - x[0] * np.ma.masked_array(pattern, block.mask))**2.0)
            return squared_err


        ## MS: to get a progress bar
        with tqdm(total=np.sum([4.0*len(ad) for ad in adinputs])) as pbar:
            for ad in adinputs:
                if ad.phu.get(timestamp_key):
                    log.warning("No changes will be made to {}, since it has "
                                "already been processed by removePatternNoise".
                                format(ad.filename))
                    pbar.update(4)
                    continue

                for ext in ad:
                    # padding for GNIRS
                    if ad.instrument() == 'GNIRS':
                        # Number of rows must be 4-divisible, or it crashes.
                        log.stdinfo("Padding GNIRS SCI, VAR, DQ y-axis by 2 rows to "
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
                            np.ones((2, ext.shape[1]), dtype=ext.mask.dtype),
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
                            quad = ext.nddata[ystart:ystart + qysize, xstart:xstart + qxsize]
                            sigma_in = sigclip(np.ma.masked_array(quad.data, quad.mask)).std()
                            blocks = [quad[tuple(slice(start, end)
                                                 for (start, end) in coords)]
                                      for coords in cart_product(yticks, xticks)]
                            slice_hist = [tuple(slice(start, end) for (start, end) in coords)
                                      for coords in cart_product(yticks, xticks)]

                            if bgsub:
                                # If all pixels are masked in a box, we'll get no
                                # result from the mean. Suppress warning.
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=UserWarning)
                                    zeros = np.nan_to_num([-np.ma.masked_array(block.data, block.mask).mean()
                                                           for block in blocks])
                            out = stack_function(blocks, zero=zeros).data
                            out_quad = (quad.data + np.mean(out) -
                                        np.tile(out, (len(yticks), len(xticks))))
                            sigma_out = sigclip(np.ma.masked_array(out_quad, quad.mask)).std()
                            if sigma_out > sigma_in:
                                qstr = (f"{ad.filename} extension {ext.id} "
                                        f"quadrant ({xstart},{ystart})")
                                if not must_reduce_rms:
                                    log.stdinfo("Forcing cleaning on " + qstr)
                                else:
                                    log.stdinfo("No improvement for " + qstr +
                                                ", not applying pattern removal.")
                                    continue

                            ## MS: now finding the applicable roi for pattern subtraction. Note slopes={'0':end,'1':+slope,'-1':-slope}
                            scaling_factors = np.array([])
                            for block in blocks:
                                res = minimize(pattern_func, [0.1], args=(block, out), method='BFGS')
                                scaling_factors = np.append(scaling_factors, res.x[0])

                            scaling_factors_quad = np.ones(quad.data.shape)
                            for i in range(len(blocks)):
                                scaling_factors_quad[slice_hist[i]] = scaling_factors[i]
                            _, YY, __ = sigma_clipped_stats(scaling_factors_quad, axis=1, sigma=2.0)
                            D_YY = np.diff(YY)
                            idxs = np.array([0])
                            slopes = np.array([0])
                            for ff, kk in zip([edge_threshold, -1.0*edge_threshold], [1, -1]):
                                t_idxs = np.argwhere(D_YY>(D_YY.mean()+ff*D_YY.std())).flatten()
                                if kk == -1:
                                    t_idxs = np.argwhere(D_YY<(D_YY.mean()+ff*D_YY.std())).flatten()
                                idxs = np.concatenate((idxs, t_idxs+1))
                                slopes = np.concatenate((slopes, np.array([kk]*len(t_idxs))))
                            idxs = np.append(idxs, quad.shape[0])
                            slopes = np.append(slopes, 0)
                            args_sorted = np.argsort(idxs)
                            idxs = idxs[args_sorted]
                            slopes = (slopes[args_sorted]).astype(int)
                            new_out_quad = out_quad.copy()

                            ## MS: final cleaned quad
                            for i in range(1, len(idxs)):
                                if i == len(idxs)-1:
                                    if slopes[i-1] == -1:
                                        new_out_quad[idxs[i-1]:idxs[i],:] = quad.data[idxs[i-1]:idxs[i],:]
                                elif slopes[i] == 1:
                                    new_out_quad[idxs[i-1]:idxs[i],:] = quad.data[idxs[i-1]:idxs[i],:]

                            ext.data[ystart:ystart + qysize, xstart:xstart + qxsize] = new_out_quad
                            pbar.update(1)

                    if ad.instrument() == 'GNIRS':
                        # Remove padding before returning
                        log.debug('Removing padding from GNIRS SCI, VAR, DQ y-axis.')
                        # Delete last 2 rows
                        ext.data = np.delete(ext.data, [-2, -1], axis=0)
                        ext.variance = np.delete(ext.variance, [-2, -1], axis=0)
                        ext.mask = np.delete(ext.mask, [-2, -1], axis=0)

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
