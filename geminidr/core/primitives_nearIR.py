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

    @staticmethod
    def levelQuad(ext, sig=2.0, smoothing_extent=5, subquad=None):
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
        subquad: dict or None, Default: None
            If ext also has a bias offset at the sub-quad level, this dictionary contains
            the following information.
            subquad = {'border':y, 'stitch_direction':x}, where
            y: list of int
                Indices of rows across which bias offsets are to be determined.
                If ext was padded, make sure the indices passed were determined from it and not the
                original ext.
            x: list of int
                Indices of the starting column of the corresponding quad.
        """
        qysize, qxsize = [size // 2 for size in ext.data.shape]

        clippedstats_lr = partial(sigma_clipped_stats, axis=0, sigma=sig)
        clippedstats_tb = partial(sigma_clipped_stats, axis=1, sigma=sig)

        def find_offset(ext, arr1_slicers, arr2_slicers, clipper):
            quad_1 = ext.nddata[arr1_slicers[0], arr1_slicers[1]]
            quad_2 = ext.nddata[arr2_slicers[0], arr2_slicers[1]]
            if hasattr(ext, 'OBJMASK'):
                objmask_1 = ext.OBJMASK[arr1_slicers[0], arr1_slicers[1]]
                objmask_2 = ext.OBJMASK[arr2_slicers[0], arr2_slicers[1]]
                mask_1 = quad_1.mask | objmask_1
                mask_2 = quad_2.mask | objmask_2
            else:
                mask_1 = quad_1.mask
                mask_2 = quad_2.mask
            arr1 = np.ma.masked_array(quad_1.data, mask_1)
            arr2 = np.ma.masked_array(quad_2.data, mask_2)
            if arr1.mask.all() or arr2.mask.all():
                #print ("All masked")
                return 0.0

            meds_1 = clipper(arr1)[1] # 2nd tuple item is median
            meds_2 = clipper(arr2)[1] # 2nd tuple item is median
            offsets = meds_1 - meds_2
            return np.median(offsets[np.isfinite(offsets)]) # to be added to arr2's quad

        if subquad is not None:
            # some assignment gymnastics to tackle intra-quad edges in top and bottom quads
            # -99 is a flag for quad interfaces
            Y = np.array(subquad['border'])
            X = np.array(subquad['stitch_direction'])
            todel = np.argwhere((Y==0) | (Y==qysize) | (Y==2*qysize))
            Y = np.delete(Y, todel)
            X = np.delete(X, todel)
            idx = np.argsort(Y)
            X, Y = X[idx], Y[idx]
            Y = np.insert(Y, 0, 0)
            X = np.insert(X, 0, -99)
            Y = np.append(Y, ext.data.shape[0])
            X = np.append(X, -99)

            X_bottom = X[Y<qysize]
            Y_bottom = Y[Y<qysize]
            Y_bottom = np.append(Y_bottom, qysize)
            X_bottom = np.append(X_bottom, -99)

            X_top = X[Y>=qysize]
            Y_top = Y[Y>=qysize]
            Y_top = np.insert(Y_top, 0, qysize)
            X_top = np.insert(X_top, 0, -99)

            for i in range(len(X_bottom)):
                ylow = 0
                if X_bottom[i] == -99:
                    continue

                if X_bottom[i] < qxsize:
                    idx = np.argwhere((X_bottom<qxsize) | (X_bottom==-99))
                else:
                    idx = np.argwhere((X_bottom>=qxsize) | (X_bottom==-99))
                yup = Y_bottom[idx[idx>i][0]]

                sq_smoothe = min(smoothing_extent*10, Y_bottom[i]-ylow, yup-Y_bottom[i])
                arr1_slicers = [slice(Y_bottom[i],Y_bottom[i] + sq_smoothe), slice(X_bottom[i],X_bottom[i] + qxsize)]
                arr2_slicers = [slice(Y_bottom[i] - sq_smoothe,Y_bottom[i]), slice(X_bottom[i],X_bottom[i] + qxsize)]
                offset = find_offset(ext, arr1_slicers, arr2_slicers, clippedstats_lr)
                ext.data[ylow:Y_bottom[i], X_bottom[i]:X_bottom[i] + qxsize] += offset

            X_top = X_top[::-1]
            Y_top = Y_top[::-1]
            for i in range(len(X_top)):
                yup = Y_top[0]
                if X_top[i] == -99:
                    continue

                if X_top[i] < qxsize:
                    idx = np.argwhere((X_top<qxsize) | (X_top==-99))
                else:
                    idx = np.argwhere((X_top>=qxsize) | (X_top==-99))
                ylow = Y_top[idx[idx>i][0]]

                sq_smoothe = min(smoothing_extent*10, yup-Y_top[i], Y_top[i]-ylow)
                arr1_slicers = [slice(Y_top[i] - sq_smoothe,Y_top[i]), slice(X_top[i],X_top[i] + qxsize)]
                arr2_slicers = [slice(Y_top[i],Y_top[i] + sq_smoothe), slice(X_top[i],X_top[i] + qxsize)]
                offset = find_offset(ext, arr1_slicers, arr2_slicers, clippedstats_lr)
                ext.data[Y_top[i]:yup, X_top[i]:X_top[i] + qxsize] += offset



        ## stitch along qysize
        for xstart in [0, qxsize]:
            arr1_slicers = [slice(qysize,qysize + smoothing_extent), slice(xstart,xstart + qxsize)]
            arr2_slicers = [slice(qysize - smoothing_extent,qysize), slice(xstart,xstart + qxsize)]
            offset = find_offset(ext, arr1_slicers, arr2_slicers,
                                 clippedstats_lr
                                )
            ext.data[0:qysize, xstart:xstart + qxsize] += offset


        ## stitch in full along 2 x qxsize
        arr1_slicers = [slice(0,ext.data.shape[0]), slice(qxsize - smoothing_extent,qxsize)]
        arr2_slicers = [slice(0,ext.data.shape[0]), slice(qxsize,qxsize + smoothing_extent)]
        offset = find_offset(ext, arr1_slicers, arr2_slicers,
                             clippedstats_tb
                            )
        ext.data[:, qxsize:2 * qxsize] += offset

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

        if clean=="skip":
            log.stdinfo("Skipping cleanReadout since 'clean' is set to 'skip'")
            return adinputs

        stack_function = NDStacker(combine='median', reject='sigclip',
                                   hsigma=hsigma, lsigma=lsigma)
        sigclip = partial(sigma_clip, sigma_lower=lsigma, sigma_upper=hsigma)
        zeros = None  # will remain unchanged if not subtract_background

        def exact_sol(block, pattern):
            mn = np.ma.masked_array(block.data, block.mask).mean()
            pp = np.ma.masked_array(pattern, block.mask).mean()
            sum1 = np.sum(np.ma.masked_array(block.data - mn, block.mask) * np.ma.masked_array(pattern - pp, block.mask))
            sum2 = np.sum(np.ma.masked_array(pattern - pp, block.mask)**2.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sum1/sum2 
            return res
            

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by cleanReadout".
                            format(ad.filename))
                continue

            for ext in ad:
                subquad = {'border':[], 'stitch_direction':[]} # preparing for bias leveling

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
                        np.zeros((2, ext.shape[1]),
                                 dtype=ext.data.dtype),
                        axis=0)
                    ext.variance = np.append(
                        ext.variance,
                        np.zeros((2, ext.shape[1]),
                                 dtype=ext.variance.dtype),
                        axis=0)
                    ext.mask = np.append(
                        ext.mask,
                        np.ones((2, ext.shape[1]), dtype=ext.mask.dtype)*16,
                        axis=0) # 16 is DQ.no_data
                    log.debug("New image shape:\n"
                              f"  SCI: {ext.data.shape}\n"
                              f"  VAR: {ext.variance.shape}\n"
                              f"   DQ: {ext.mask.shape}")

                qysize, qxsize = [size // 2 for size in ext.data.shape]
                yticks = [(y, y + pysize) for y in range(0, qysize, pysize)]
                xticks = [(x, x + pxsize) for x in range(0, qxsize, pxsize)]
                quads_info = {}
                cleaned_quads = 0

                for ystart in (0, qysize):
                    quads_info[ystart] = {}
                    for xstart in (0, qxsize):
                        quads_info[ystart][xstart] = {}

                        quad = ext.nddata[ystart:ystart + qysize, xstart:xstart + qxsize]
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

                        ## MS: now finding the applicable roi for pattern subtraction.
                        ## Note slopes={'0':end,'1':+slope,'-1':-slope}
                        scaling_factors = np.array([])
                        for block in blocks:
                            res = exact_sol(block, out)
                            scaling_factors = np.append(scaling_factors, res)

                        scaling_factors_quad = np.ones(quad.data.shape)
                        for i in range(len(blocks)):
                            scaling_factors_quad[slice_hist[i]] = scaling_factors[i]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            YY = sigma_clipped_stats(scaling_factors_quad, axis=1, sigma=2.0)[1]
                        ## MS: Important to smoothe out YY with a Savitzky Golay filter to remove noise while using simple thresholding
                        YY = savgol_filter(YY, sg_win_size, 1, deriv=0) # May consider putting the window as a primitive's parameter

                        ## Using a simple threshold of 0.4 and finding the crossings
                        idxs = np.argwhere(np.diff(YY < simple_thres))
                        slopes = np.ones_like(idxs, dtype=int)
                        mm = YY[idxs] < simple_thres
                        slopes[~mm] = -1
                        idxs = np.append(idxs, [0, quad.shape[0]])
                        slopes = np.append(slopes, [0, 0])
                        args_sorted = np.argsort(idxs)
                        idxs = idxs[args_sorted]
                        slopes = slopes[args_sorted]
                        new_out_quad = out_quad.copy()

                        quads_info[ystart][xstart] = {'ystart':int(ystart),
                                                      'ystop':int(ystart + qysize),
                                                      'xstart':int(xstart),
                                                      'xstop':int(xstart + qxsize),
                                                      'idxs':idxs,
                                                      'slopes':slopes,
                                                      'new_out_quad':new_out_quad,
                        }


                for ys in [0, qysize]:
                    for xs, Q in quads_info[ys].items():
                        idxs = Q['idxs']
                        quad = ext.nddata[Q['ystart']:Q['ystop'], Q['xstart']:Q['xstop']]
                        new_out_quad = Q['new_out_quad']
                        slopes = Q['slopes']
                        sigma_in = sigclip(np.ma.masked_array(quad.data, quad.mask)).std()

                        ## MS: final cleaned quad
                        for i in range(1, len(idxs)):
                            if i == len(idxs)-1:
                                if slopes[i-1] == -1:
                                    new_out_quad[idxs[i-1]:idxs[i],:] = quad.data[idxs[i-1]:idxs[i],:]
                                    subquad['border'] += [idxs[i-1]+Q['ystart'], idxs[i]+Q['ystart']]
                                    subquad['stitch_direction'] += [Q['xstart'], Q['xstart']]
                            elif slopes[i] == 1:
                                new_out_quad[idxs[i-1]:idxs[i],:] = quad.data[idxs[i-1]:idxs[i],:]
                                subquad['border'] += [idxs[i-1]+Q['ystart'], idxs[i]+Q['ystart']]
                                subquad['stitch_direction'] += [Q['xstart'], Q['xstart']]

                        sigma_out = sigclip(np.ma.masked_array(new_out_quad, quad.mask)).std()
                        if sigma_out > sigma_in:
                            qstr = (f"{ad.filename} extension {ext.id} "
                                    f"quadrant ({Q['xstart']},{Q['ystart']})")
                            if clean=="force":
                                log.stdinfo("Forcing cleaning on " + qstr)
                            else: # clean is default
                                log.stdinfo("No improvement for " + qstr +
                                            ", not applying pattern removal.")
                                continue
                        cleaned_quads += 1
                        ext.data[Q['ystart']:Q['ystop'], Q['xstart']:Q['xstop']] = new_out_quad

                if level_bias_offset and cleaned_quads>0:
                    log.stdinfo("Leveling quads now.....")
                    if len(subquad['border']) > 0:
                        NearIR.levelQuad(ext, smoothing_extent=smoothing_extent, subquad=subquad)
                    else:
                        NearIR.levelQuad(ext, smoothing_extent=smoothing_extent)

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
