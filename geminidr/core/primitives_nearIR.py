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

    def _levelQuad(self, masked_data, sig=2.0, smoothing_extent=5, edges=None):
        """
        Parameters
        ----------
        masked_data: np.ma.masked_array
            pattern-corrected image
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
        qysize, qxsize = [size // 2 for size in masked_data.shape]

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

        # Go through the quads aligning the image levels at all the intra-quad
        # edges. Work from the top/bottom towards the centre. Be careful to
        # directly modify the 'data' attribute or else the masked pixels will
        # be unaffected
        intraquad_smooth = smoothing_extent * 10
        for (ystart, xstart), quad_edges in edges.items():
            xslice = slice(xstart, xstart+qxsize)
            for i, edge in enumerate(quad_edges):
                # "edge" is the row number in the quad (0-indexed) *above* the edge
                try:  # ensure we don't go over a subsequent edge
                    this_smooth = min(intraquad_smooth, edge, qysize-edge,
                                      abs(edge - quad_edges[i+1]))
                except IndexError:
                    this_smooth = min(intraquad_smooth, edge, qysize-edge)
                edge += ystart  # the real row number
                arr1_slicers = (slice(edge, edge+this_smooth), xslice)
                arr2_slicers = (slice(edge-this_smooth, edge), xslice)
                offset = find_offset(masked_data, arr1_slicers, arr2_slicers, clippedstats_tb)
                if ystart:
                    log.debug(f"Adding {-offset} to row {edge} and above "
                              f"in (X{xstart}, Y{ystart}) quad")
                    masked_data.data[slice(edge, None), xslice] -= offset
                else:
                    log.debug(f"Adding {offset} below row {edge} "
                              f"in (X{xstart}, Y{ystart}) quad")
                    masked_data.data[slice(0, edge), xslice] += offset

        # match top and bottom halves of left and right separately
        for xslice in (slice(0, qxsize), slice(qxsize, None)):
            arr1_slicers = (slice(qysize,qysize + smoothing_extent), xslice)
            arr2_slicers = (slice(qysize - smoothing_extent,qysize), xslice)
            offset = find_offset(masked_data, arr1_slicers, arr2_slicers,
                                 clippedstats_tb)
            log.debug(f"Adding {offset} to bottom of X{xslice.start} quad")
            masked_data.data[slice(0, qysize), xslice] += offset

        # match left and right halves
        arr1_slicers = (slice(None), slice(qxsize - smoothing_extent,qxsize))
        arr2_slicers = (slice(None), slice(qxsize,qxsize + smoothing_extent))
        offset = find_offset(masked_data, arr1_slicers, arr2_slicers,
                             clippedstats_lr)
        # xslice still set to right half from previous loop
        log.debug(f"Adding {offset} to right half")
        masked_data.data[slice(None), xslice] += offset

    def cleanReadout(self, adinputs=None, **params):
        """
        This attempts to remove the pattern noise in NIRI/GNIRS data
        after automatically determining the coverage of the pattern in
        a given quadrant. It operates in several steps.

        1. For each quadrant, determine the structure of the pattern noise by
           making an average of all the pattern-box-sized regions. The strength
           of the pattern is determined from its standard deviation.
        2. If the strength of the average pattern exceeds a critical value
           (pattern_strength_thres), determine its amplitude as a function of
           row by scaling it to fit the data in each pattern box.
        3. After investigating all 4 quads, combine the amplitude functions to
           improve S/N when looking for the pattern-affected regions (taking
           advantage of the four-fold symmetry) and look for edges where the
           pattern amplitude changes using a 1D version of the Canny method.
        4. Subtract the average pattern in affected regions of affected quads
        5. Normalize signal levels across the edges within each quad
        6. Normalize signal levels across the top and bottom halves of each side
        7. Normalize signal levels across the left and right halves

        Parameters
        ----------
        suffix: str, Default: "_readoutCleaned"
            Suffix to be added to output files.
        hsigma/lsigma: float
            High and low sigma-clipping limits.
        pattern_x_size: int
            Size of pattern "box" in x direction. Must be a multiple of 4.
        pattern_y_size: int
            Size of pattern "box" in y direction. Must be a multiple of 4.
        subtract_background: bool (debug-level)
            Remove median of each "box" before calculating pattern noise?
        level_bias_offset: bool
            Level the offset in bias level across (sub-)quads that typically accompany
            pattern noise.
        smoothing_extent: int
            Used only when `level_bias_offset` is set to True.
            Width (in pixels) of the region at a given quad interface to be smoothed over
            on each side of the interface.
            Note that for intra-quad leveling, this width is broadened by a factor 10.
        sg_win_size: int
            Smoothing window size for the Savitzky-Golay filter applied during automated
            detection of pattern coverage.
        simple_thres: float
            Threshold used in automated detection of pattern coverage.
            Favorable range [0.3, 0.8]. If the result (at the intra-quad level) is not satisfactory,
            play with this parameter.
        pat_strength_thres: float
            Threshold used to characterise the strength of the pattern noise. If greater than
            this value, run the whole machinery otherwise leave the frame untouched.
        clean: str, Default: "skip"
            Must be one of "skip", "default", or "force".
            skip: Skip this routine entirely when called from a recipe.
            default: Apply the pattern subtraction to each quadrant of the image if doing
                     so decreases the RMS.
            force: Force the pattern subtraction in each quadrant.
        canny_sigma: float (debug-level)
            standard deviation of Gaussian smoothing kernel used in Canny edge detection
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        hsigma, lsigma = params["hsigma"], params["lsigma"]
        pxsize, pysize = params["pattern_x_size"], params["pattern_y_size"]
        bgsub = params["debug_subtract_background"]
        clean = params["clean"]
        level_bias_offset = params["level_bias_offset"]
        smoothing_extent = params["smoothing_extent"]

        simple_thres = params["simple_thres"]
        pat_strength_thres = params["pat_strength_thres"]
        canny_sigma = params["debug_canny_sigma"]

        if clean == "skip":
            log.stdinfo("Skipping cleanReadout since 'clean' is set to 'skip'")
            return adinputs

        stack_function = NDStacker(combine='median', reject='sigclip',
                                   hsigma=hsigma, lsigma=lsigma)

        def pad(arr, padding, fill_value=0):
            if arr is None:
                return arr
            elif padding:
                pad_data = np.full((padding, arr.shape[1]), fill_value, dtype=arr.dtype)
                return np.append(arr, pad_data, axis=0)
            return arr.copy()

        def flip(arr, padding=0, flipit=True):
            """
            GNIRS is weird, the top quad is really only 510 rows but
            these are read out synchronously with the bottom 510 rows
            therefore the "reflection" needs a bit of tweaking, you
            flip the bottom 510 rows and then add 2 dummy rows at the top.
            This function does that if needed. If there's no padding, then
            it just reverses the array. It can be called with flipit=False,
            it which case it does nothing.
            """
            if flipit and padding:
                return np.r_[arr[-(padding + 1)::-1], np.zeros((padding,), dtype=arr.dtype)]
            elif flipit:
                return arr[::-1]
            return arr

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, since"
                            " it has already been processed by cleanReadout")
                continue

            for ext in ad:
                edges = {}

                # data must be a multiple number of pattern boxes
                ny, nx = ext.shape
                padding = (pysize - ny % pysize) % pysize
                data = pad(ext.data, padding)
                mask = pad(ext.mask, padding, DQ.no_data)
                padded_array = np.ma.masked_array(data, mask=mask)
                qysize, qxsize = [size // 2 for size in data.shape]
                nypat, nxpat = qysize // pysize, qxsize // pxsize
                nblocks = nxpat * nypat
                pattern_size = pxsize * pysize
                if hasattr(ext, 'OBJMASK'):
                    padded_array.mask |= pad(ext.OBJMASK.astype(bool), padding)

                cleaned_quads = 0
                def reblock(data):
                    """Reshape data into a stack of pattern-box-sized arrays"""
                    return data.reshape(qysize // pysize, pysize, -1, pxsize).swapaxes(
                        1, 2).reshape(-1, pysize, pxsize)

                pattern_strengths = []
                for ystart, ydesc in zip((0, qysize), ('bottom', 'top')):
                    for xstart, xdesc in zip((0, qxsize), ('left', 'right')):
                        quad_slice = (slice(ystart, ystart+qysize), slice(xstart, xstart+qxsize))

                        # Reshape each quad into a stack of pattern-box-sized arrays
                        data_block = reblock(data[quad_slice])
                        mask_block = reblock(mask[quad_slice]) if mask is not None else None
                        blocks = np.ma.masked_array(data=data_block, mask=mask_block)

                        # If all pixels are masked in a box, we'll get no
                        # result from the mean. Suppress warning.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UserWarning)
                            zeros = np.nan_to_num(
                                blocks.reshape(nblocks, pattern_size).mean(axis=1))

                        # compute average pattern from all the pattern boxes
                        pattern = stack_function(ext.nddata.__class__(data=blocks.data, mask=blocks.mask),
                                                 zero=-zeros if bgsub else None).data
                        pattern -= pattern.mean()
                        edges[ystart, xstart] = {"pattern": pattern,
                                                 "clean": False}

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
                                pattern_strength = sigma_clipped_stats(
                                    scaling_factors.reshape(nypat, nxpat),
                                    axis=1, sigma=2.0)[1].repeat(pysize)
                            pattern_strengths.append(flip(pattern_strength, padding, ystart > 0))
                        else:
                            qstr = (f"{ad.filename} extension {ext.id} "
                                    f"{ydesc}-{xdesc} quadrant")
                            if clean == "force":
                                log.stdinfo(f"Forcing cleaning on {qstr}")
                            else:
                                log.stdinfo(f"Weak pattern for {qstr}, "
                                            "not applying pattern removal.")
                                continue
                        edges[ystart, xstart]["clean"] = True
                        cleaned_quads += 1

                if cleaned_quads > 0:
                    combined_pattern_strength = np.mean(pattern_strengths, axis=0)
                    new_edges, affected_rows = find_edges(
                        combined_pattern_strength, sigma=canny_sigma,
                        min_range=simple_thres, pysize=pysize)
                    top_affected_rows = flip(affected_rows, padding)
                for (ystart, xstart), _dict in edges.items():
                    if _dict["clean"]:
                        quad_slice = (slice(ystart, ystart + qysize),
                                      slice(xstart, xstart + qxsize))
                        out_quad = data[quad_slice] - np.tile(_dict["pattern"], (nypat, nxpat))
                        if ystart:
                            padded_array.data[quad_slice][top_affected_rows] = out_quad[top_affected_rows]
                        else:
                            padded_array.data[quad_slice][affected_rows] = out_quad[affected_rows]
                        # store locations of intra-quad edges for _levelQuad()
                        # if a switch_location is 100, for example, it means
                        # the pattern switches between rows 100 and 101
                        # (0-indexed), so store 101 in order to be able to
                        # create slices more easily
                        if ystart:  # again, cope with GNIRS
                            edges[ystart, xstart] = qysize - padding - new_edges + 1
                        else:
                            edges[ystart, xstart] = new_edges + 1
                    else:
                        edges[ystart, xstart] = []  # no levelling here
                if level_bias_offset and cleaned_quads > 0:
                    log.stdinfo(f"Leveling quads for {ext.filename}:{ext.id}...")
                    self._levelQuad(padded_array, smoothing_extent=smoothing_extent, edges=edges)

                if padding:  # Remove padding before returning
                    ext.data = padded_array.data[:-padding]
                else:
                    ext.data = padded_array.data

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


def find_edges(data, sigma=3, min_range=1, pysize=4):
    """
    Find edges in a 1D array of noisy data, using the a 1D implementation
    of the Canny edge detector

    Parameters
    ----------
    data: sequence
        1D data array
    sigma: float
        standard deviation of Gaussian for convolution
    min_range: float
        minimum assumed range of pattern on/off strengths
    pysize: int
        pattern size in y-direction (any edges within this distance of
        the top or bottom are ignored)

    Returns
    -------
    array: locations of edges
    """
    twosigsq = 2 * sigma * sigma
    halfsize = int(sigma * 2 + 0.8)
    x = np.arange(-halfsize, halfsize+1)
    norm = 1 / np.sqrt(np.pi * twosigsq)
    gauss_kernel = norm * np.exp(-x*x / twosigsq)
    # mitigate edge effects
    new_data = np.r_[[data[0]] * halfsize, data, [data[-1]] * halfsize]
    conv_data = np.convolve(new_data, gauss_kernel, mode='same')
    grad = np.convolve(conv_data, [1, 0, -1], mode='same')[halfsize:-halfsize]
    range = max(0.5 * np.diff(np.nanpercentile(data, [10, 90]))[0], min_range)
    threshold = 0.8 * range / (1.2 * sigma + 0.28)  # empirical
    ypixels = np.arange(data.size)[pysize:-pysize]
    diffs = np.array([np.diff(grad[pysize-1:-pysize]),
                      -np.diff(grad[pysize:-pysize+1])])
    extrema = np.logical_and(np.multiply.reduce(diffs, axis=0) >= 0,
                             abs(grad[pysize:-pysize]) > threshold)
    extrema_locations = ypixels[extrema]
    extrema_types = grad[extrema_locations]
    # We know there's a pattern here or this code wouldn't be run, so
    # ensure all rows are flagged if there are no edges
    affected_rows = np.ones(data.shape, dtype=bool)
    last = 0
    for loc, minmax in zip(extrema_locations, extrema_types):
        if minmax > 0:  # pattern switching on
            affected_rows[last:loc] = False
            affected_rows[loc:] = True
        else:  # pattern switching off
            affected_rows[last:loc] = True
            affected_rows[loc:] = False
        last = loc
    return extrema_locations, affected_rows
