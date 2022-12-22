#
#                                                                  gemini_python
#
#                                                           primitives_nearIR.py
# ------------------------------------------------------------------------------
import datetime
from functools import partial
from itertools import product as cart_product
import warnings

from astropy.stats import sigma_clip, sigma_clipped_stats
import numpy as np

from gempy.gemini import gemini_tools as gt
from gempy.library.nddops import NDStacker

from geminidr.gemini.lookups import DQ_definitions as DQ
from . import parameters_nearIR, Bookkeeping

from recipe_system.utils.decorators import parameter_override, capture_provenance

from scipy.signal import savgol_filter

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

    def _levelQuad(self, ext, sig=2.0, smoothing_extent=5, edges=None):
        """
        Parameters
        ----------
        ext: AstroData
            This is the pattern-corrected AstroData.
        sig: float, Default: 2.0
            Sigma-clipping threshold used for both lower and upper limits.
        smoothing_extent: int
            Width (in pixels) of the region at a given quad interface to be smoothed over
            on each side of the interface.
            Note that for intra-quad leveling, this width is broadened by a factor 10.
        edges: dict
            information about the edges of a bias offset at the sub-quad level
            each dict item is keyed by the (y, x) coords of the bottom-left
            quad corner and its value is a sequence of rows indicating where
            the pattern switches from "off" to "on" or vice versa
        """
        log = self.log
        log.debug(f"Leveling quads for {ext.filename}:{ext.id}")
        qysize, qxsize = [size // 2 for size in ext.data.shape]

        clippedstats_lr = partial(sigma_clipped_stats, axis=1, sigma=sig)
        clippedstats_tb = partial(sigma_clipped_stats, axis=0, sigma=sig)

        def find_offset(masked_data, arr1_slicers, arr2_slicers, clipper):
            """Determine offset between two regions of an image. Should
            this use gt.measure_bg_from_image()?"""
            arr1 = masked_data[arr1_slicers]
            arr2 = masked_data[arr2_slicers]
            if arr1.mask.all() or arr2.mask.all():
                return 0  # all pixels are masked

            meds_1 = clipper(arr1)[1]  # 2nd tuple item is median
            meds_2 = clipper(arr2)[1]  # 2nd tuple item is median
            offsets = meds_1 - meds_2
            return np.median(offsets[np.isfinite(offsets)])  # to be added to arr2

        # Create a masked_array object now for speed
        masked_data = np.ma.masked_array(ext.data, mask=ext.mask)
        if hasattr(ext, 'OBJMASK'):
            masked_data.mask |= ext.OBJMASK.astype(bool)

        # Go through the quads aligning the image levels at all the intra-quad
        # edges. Work from the top/bottom towards the centre
        intraquad_smooth = smoothing_extent * 10
        for (ystart, xstart), quad_edges in edges.items():
            xslice = slice(xstart, xstart+qxsize)
            if ystart:
                quad_edges = quad_edges[::-1]
            for edge in quad_edges:
                # "edge" is the row number in the quad (0-indexed) *above* the edge
                this_smooth = min(intraquad_smooth, edge, qysize-edge)
                edge += ystart  # the real row number
                arr1_slicers = (slice(edge, edge+this_smooth), xslice)
                arr2_slicers = (slice(edge-this_smooth, edge), xslice)
                offset = find_offset(masked_data, arr1_slicers, arr2_slicers, clippedstats_tb)
                if ystart:
                    log.debug(f"Adding {-offset} to row {edge} and above "
                              f"in {xslice.start} quad")
                    ext.data[slice(edge, None), xslice] -= offset
                else:
                    log.debug(f"Adding {offset} below row {edge} "
                              f"in {xslice.start} quad")
                    ext.data[slice(0, edge), xslice] += offset

        # match top and bottom halves of left and right separately
        for xslice in (slice(0, qxsize), slice(qxsize, None)):
            arr1_slicers = (slice(qysize,qysize + smoothing_extent), xslice)
            arr2_slicers = (slice(qysize - smoothing_extent,qysize), xslice)
            offset = find_offset(masked_data, arr1_slicers, arr2_slicers,
                                 clippedstats_tb)
            log.debug(f"Adding {offset} to bottom of {xslice.start} quad")
            ext.data[slice(0, qysize), xslice] += offset

        # match left and right halves
        arr1_slicers = (slice(None), slice(qxsize - smoothing_extent,qxsize))
        arr2_slicers = (slice(None), slice(qxsize,qxsize + smoothing_extent))
        offset = find_offset(masked_data, arr1_slicers, arr2_slicers,
                             clippedstats_lr)
        # xslice still set to right half from previous loop
        log.debug(f"Adding {offset} to right half")
        ext.data[slice(None), xslice] += offset

    def cleanReadout(self, adinputs=None, **params):
        """
        This attempts to remove the pattern noise in NIRI/GNIRS data
        after automatically determining the coverage of the pattern in
        a given quadrant. The latter is done by fitting for scaling factors 
        of the pattern in each quadrant. Note, however, that the scaling factors 
        are not used in subtracting off the pattern. 

        Parameters
        ----------
        suffix: str, Default: "_readoutCleaned"
            Suffix to be added to output files.
        hsigma/lsigma: float, Defaults: 3.0 for both
            High and low sigma-clipping limits.
        pattern_x_size: int, Default: 16
            Size of pattern "box" in x direction. Must be a multiple of 4.
        pattern_y_size: int, Default: 4
            Size of pattern "box" in y direction. Must be a multiple of 4.
        subtract_background: bool, Default: True
            Remove median of each "box" before calculating pattern noise?
        level_bias_offset: bool, Default: True
            Level the offset in bias level across (sub-)quads that typically accompany
            pattern noise.
        smoothing_extent: int, Default: 5
            Used only when `level_bias_offset` is set to True.
            Width (in pixels) of the region at a given quad interface to be smoothed over
            on each side of the interface.
            Note that for intra-quad leveling, this width is broadened by a factor 10.
        sg_win_size: int, Default: 25
            Smoothing window size for the Savitzky-Golay filter applied during automated 
            detection of pattern coverage. 
        simple_thres: float, Default: 0.6
            Threshold used in automated detection of pattern coverage. 
            Favorable range [0.3, 0.8]. If the result (at the intra-quad level) is not satisfactory, 
            play with this parameter. 
        pat_strength_thres: float, Default: 15.0
            Threshold used to characterise the strength of the pattern noise. If greater than 
            this value, run the whole machinery otherwise leave the frame untouched.  
        clean: str, Default: "skip"
            Must be one of "skip", "default", or "force".
            skip: Skip this routine entirely when called from a recipe.
            default: Apply the pattern subtraction to each quadrant of the image if doing
                     so decreases the RMS.
            force: Force the pattern subtraction in each quadrant.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        hsigma, lsigma = params["hsigma"], params["lsigma"]
        pxsize, pysize = params["pattern_x_size"], params["pattern_y_size"]
        bgsub = params["subtract_background"]
        clean = params["clean"]
        level_bias_offset = params["level_bias_offset"]
        smoothing_extent = params["smoothing_extent"]

        sg_win_size = params["sg_win_size"]
        # MS: increased from 0.4 to minimize over-subtraction of pattern at edges in intra-quad cases,
        # which can be sometimes deleterious
        simple_thres = params["simple_thres"]
        pat_strength_thres = params["pat_strength_thres"]

        if clean == "skip":
            log.stdinfo("Skipping cleanReadout since 'clean' is set to 'skip'")
            return adinputs

        stack_function = NDStacker(combine='median', reject='sigclip',
                                   hsigma=hsigma, lsigma=lsigma)

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, since"
                            " it has already been processed by cleanReadout")
                continue

            for ext in ad:
                edges = {}

                # padding (for GNIRS), data must be a multiple number of
                # pattern boxes
                ny, nx = ext.shape
                padding = ny % pysize
                if padding:
                    padding = pysize - padding
                    log.stdinfo(f"Padding data by {padding} rows to be "
                                f"divisible by {pysize}.")
                    ext.data = np.append(
                        ext.data, np.zeros((padding, nx),
                                           dtype=ext.data.dtype), axis=0)
                    ext.variance = np.append(
                        ext.variance, np.zeros((padding, nx),
                                               dtype=ext.variance.dtype), axis=0)
                    ext.mask = np.append(
                        ext.mask, np.full((padding, nx), DQ.no_data), axis=0)

                qysize, qxsize = [size // 2 for size in ext.data.shape]
                nypat, nxpat = qysize // pysize, qxsize // pxsize
                nblocks = nxpat * nypat
                pattern_size = pxsize * pysize
                cleaned_quads = 0

                def reblock(data):
                    """Reshape data into a stack of pattern-box-sized arrays"""
                    return data.reshape(qysize // pysize, pysize, -1, pxsize).swapaxes(
                        1, 2).reshape(-1, pysize, pxsize)

                for ystart, ydesc in zip((0, qysize), ('bottom', 'top')):
                    for xstart, xdesc in zip((0, qxsize), ('left', 'right')):

                        # Reshape each quad into a stack of pattern-box-sized arrays
                        quad = ext.nddata[ystart:ystart + qysize, xstart:xstart + qxsize]
                        data_block = reblock(quad.data)
                        var_block = reblock(quad.variance) if quad.variance is not None else None
                        mask_block = reblock(quad.mask) if quad.mask is not None else None
                        blocks = ext.nddata.__class__(data=data_block, mask=mask_block, variance=var_block)

                        # If all pixels are masked in a box, we'll get no
                        # result from the mean. Suppress warning.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UserWarning)
                            zeros = np.nan_to_num(np.ma.masked_array(
                                blocks.data.reshape(nblocks, pattern_size),
                                blocks.mask.reshape(nblocks, pattern_size)).mean(axis=1))

                        # compute average pattern from all the pattern boxes
                        pattern = stack_function(blocks, zero=-zeros if bgsub else None).data
                        pattern -= pattern.mean()
                        out_quad = quad.data - np.tile(pattern, (nypat, nxpat))

                        # MS: do not touch the quad if pattern strength is weak
                        if pattern.std() >= pat_strength_thres:
                            # MS: now finding the applicable roi for pattern subtraction.
                            # Calculate the scaling factors for the pattern in
                            # all pattern boxes and investigate as a fn of row
                            pattern2 = np.ma.masked_array(
                                np.tile(pattern.data, (nblocks, 1, 1)), mask=blocks.mask)
                            sum1 = ((blocks - zeros[:, np.newaxis, np.newaxis]) * pattern).reshape(
                                nblocks, pattern_size).sum(axis=1)
                            sum2 = (pattern2 ** 2).reshape(nblocks, pattern_size).sum(axis=1)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                scaling_factors = sum1 / sum2

                            # Compute the mean of each row of pattern boxes and
                            # then replicate to height of the quad
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pattern_strength = sigma_clipped_stats(
                                    scaling_factors.reshape(nypat, nxpat),
                                    axis=1, sigma=2.0)[1].repeat(pysize)

                            # MS: Important to smooth with a Savitzky-Golay
                            # filter to remove noise while using simple thresholding
                            pattern_strength = savgol_filter(
                                pattern_strength, sg_win_size, 1, deriv=0)

                            # Using a simple threshold and finding its crossings
                            # slopes: +1 means pattern turning on, -1 means
                            # turning off (as we move up the quad)
                            unaffected_rows = pattern_strength < simple_thres
                            switch_locations = np.where(np.diff(unaffected_rows))[0]

                            # If it looks like pattern noise only affected part
                            # of the quad, reinstate the unaffected part
                            out_quad[unaffected_rows] = quad.data[unaffected_rows]

                            # store locations of intra-quad edges for _levelQuad()
                            # if a switch_location is 100, for example, it means
                            # the pattern switches between rows 100 and 101
                            # (0-indexed), so store 101 in order to be able to
                            # create slices more easily
                            edges[(ystart, xstart)] = switch_locations + 1

                        else:
                            qstr = (f"{ad.filename} extension {ext.id} "
                                    f"{ydesc}-{xdesc} quadrant")
                            if clean == "force":
                                log.stdinfo(f"Forcing cleaning on {qstr}")
                            else:
                                log.stdinfo(f"Weak pattern for {qstr}, "
                                            "not applying pattern removal.")
                                continue
                        cleaned_quads += 1
                        ext.data[ystart:ystart+qysize, xstart:xstart+qxsize] = out_quad

                if level_bias_offset and cleaned_quads > 0:
                    log.stdinfo("Leveling quads now.....")
                    self._levelQuad(ext, smoothing_extent=smoothing_extent, edges=edges)

                if padding:
                    # Remove padding before returning
                    log.debug('Removing padding from data')
                    ext.data = ext.data[:-padding]
                    ext.variance = ext.variance[:-padding]
                    ext.mask = ext.mask[:-padding]

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
