#
#                                                                  gemini_python
#
#                                                            primitives_stack.py
# ------------------------------------------------------------------------------
import astrodata
from astrodata.fits import windowedOp

import numpy as np
from astropy import table
from copy import deepcopy

from gempy.gemini import gemini_tools as gt
from gempy.library.nddops import NDStacker

from geminidr import PrimitivesBASE
from . import parameters_stack

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Stack(PrimitivesBASE):
    """
    This is the class containing all of the primitives for stacking.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_stack)

    def stackBiases(self, adinputs=None, **params):
        """
        This primitive stacks the inputs without any scaling or offsetting,
        suitable for biases.
        """
        log = self.log
        log.debug(gt.log_message("primitve", self.myself(), "starting"))

        if not all('BIAS' in bias.tags for bias in adinputs):
            raise ValueError("Not all inputs have BIAS tag")

        stack_params = self._inherit_params(params, "stackFrames")
        stack_params.update({'zero': False, 'scale': False})
        adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs

    def stackFlats(self, adinputs=None, **params):
        """This primitive stacks the inputs without offsetting, suitable
        for flats."""
        log = self.log
        log.debug(gt.log_message("primitve", self.myself(), "starting"))

        stack_params = self._inherit_params(params, "stackFrames")
        stack_params.update({'zero': False})
        adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs

    def stackDarks(self, adinputs=None, **params):
        """
        This primitive checks the inputs have the same exposure time and
        stacks them without any scaling or offsetting, suitable for darks.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        if not all('DARK' in dark.tags for dark in adinputs):
            raise ValueError("Not all inputs have DARK tag")

        # some GMOS 2008 darks were found to have exposure times varied by a
        # tiny little amount.  So we use a delta to check for equality.
        if not all(abs(dark.exposure_time() - adinputs[0].exposure_time()) < 0.01
                   for dark in adinputs[1:]):
                raise ValueError("Darks are not of equal exposure time")
        if any('NODANDSHUFFLE' in dark.tags for dark in adinputs):
            if not all('NODANDSHUFFLE' in dark.tags for dark in adinputs):
                raise ValueError("Some darks are nod-and-shuffle, some are not.")
            if not all(dark.shuffle_pixels() == adinputs[0].shuffle_pixels()
                       for dark in adinputs[1:]):
                raise ValueError("Darks are not of equal shuffle size.")

        stack_params = self._inherit_params(params, "stackFrames")
        stack_params.update({'zero': False, 'scale': False})
        adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs

    def stackFrames(self, adinputs=None, **params):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Any set of 2D.

        suffix : str
            Suffix to be added to output files.

        apply_dq : bool
            Apply DQ mask to data before combining?

        nlow, nhigh : int
            Number of low and high pixels to reject, for the 'minmax' method.
            The way it works is inherited from IRAF: the fraction is specified
            as the number of  high  and low  pixels,  the  nhigh and nlow
            parameters, when data from all the input images are used.  If
            pixels  have  been  rejected  by offseting,  masking, or
            thresholding then a matching fraction of the remaining pixels,
            truncated to an integer, are used.  Thus::

                nl = n * nlow/nimages + 0.001
                nh = n * nhigh/nimages + 0.001

            where n is the number of pixels  surviving  offseting,  masking,
            and  thresholding,  nimages  is the number of input images, nlow
            and nhigh are task parameters  and  nl  and  nh  are  the  final
            number  of  low  and high pixels rejected by the algorithm.  The
            factor of 0.001 is to adjust for rounding of the ratio.

        operation : str
            Combine method.

        reject_method : str
            Pixel rejection method (none, minmax, sigclip, varclip).

        zero : bool
            Apply zero-level offset to match background levels?

        scale : bool
            Scale images to the same intensity?

        memory : float or None
            Available memory (in GB) for stacking calculations.

        statsec : str
            Section for statistics.

        separate_ext : bool
            Handle extensions separately?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Sky stacked image. This list contains only one element. The list
            format is maintained so this primitive is consistent with all the
            others.

        Raises
        ------
        IOError
            If the number of extensions in any of the `AstroData` objects is
            different.

        IOError
            If the shape of any extension in any `AstroData` object is different.

        AssertError
            If any of the `.gain()` descriptors is None.

        AssertError
            If any of the `.read_noise()` descriptors is None.
        """
        def flatten(*args):
            return (el for item in args for el in (
                flatten(*item) if isinstance(item, (list, tuple)) else (item,)))

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["stackFrames"]
        sfx = params["suffix"]
        memory = params["memory"]
        if memory is not None:
            memory = int(memory * 1000000000)

        zero = params.get("zero", False)
        scale = params.get("scale", False)
        apply_dq = params["apply_dq"]
        separate_ext = params.get("separate_ext", False)
        statsec = params.get("statsec", None)
        reject_method = params["reject_method"]
        save_rejection_map = params["save_rejection_map"]

        if statsec:
            statsec = tuple([slice(int(start)-1, int(end))
                             for x in reversed(statsec.strip('[]').split(','))
                             for start, end in [x.split(':')]])

        if len(adinputs) <= 1:
            log.stdinfo("No stacking will be performed, since at least two "
                        "input AstroData objects are required for stackFrames")
            return adinputs

        if (reject_method == "minmax" and self.mode == "qa" and
                params["nlow"] + params["nhigh"] >= len(adinputs)):
            log.warning("Trying to reject too many images. Setting nlow=nhigh=0.")
            params["nlow"] = 0
            params["nhigh"] = 0

        if len({len(ad) for ad in adinputs}) > 1:
            raise OSError("Not all inputs have the same number of extensions")
        if len({ext.nddata.shape for ad in adinputs for ext in ad}) > 1:
            raise OSError("Not all inputs images have the same shape")

        # We will determine the average gain from the input AstroData
        # objects and add in quadrature the read noise
        gain_list = [ad.gain() for ad in adinputs]
        rn_list = [ad.read_noise() for ad in adinputs]

        # Determine whether we can construct these averages
        process_gain = not None in flatten(gain_list)
        process_rn = not None in flatten(rn_list)

        # Compute gain and read noise of final stacked images
        num_img = len(adinputs)
        num_ext = len(adinputs[0])
        zero_offsets = np.zeros((num_ext, num_img), dtype=np.float32)
        scale_factors = np.ones_like(zero_offsets)

        # Try to determine how much memory we're going to need to stack and
        # whether it's necessary to flush pixel data to disk first
        # Also determine kernel size from offered memory and bytes per pixel
        bytes_per_ext = []
        for ext in adinputs[0]:
            bytes = 0
            # Count _data twice to handle temporary arrays
            bytes += 2 * ext.data.dtype.itemsize
            if ext.variance is not None:
                bytes += ext.variance.dtype.itemsize

            bytes += 2  # mask always created
            bytes_per_ext.append(bytes * np.prod(ext.shape))

        if memory is not None and (num_img * max(bytes_per_ext) > memory):
            adinputs = self.flushPixels(adinputs)

        # Compute the scale and offset values by accessing the memmapped data
        # so we can pass those to the stacking function
        # TODO: Should probably be done better to consider only the overlap
        # regions between frames
        if scale or zero:
            levels = np.empty((num_img, num_ext), dtype=np.float32)
            for i, ad in enumerate(adinputs):
                for index in range(num_ext):
                    nddata = (ad[index].nddata.window[:] if statsec is None
                              else ad[index].nddata.window[statsec])
                    #levels[i, index] = np.median(nddata.data)
                    levels[i, index] = gt.measure_bg_from_image(nddata, value_only=True)
            if scale and zero:
                log.warning("Both scale and zero are set. Setting scale=False.")
                scale = False
            if separate_ext:
                # Target value is corresponding extension of first image
                if scale:
                    scale_factors = (levels[0] / levels).T
                else:  # zero=True
                    zero_offsets = (levels[0] - levels).T
            else:
                # Target value is mean of all extensions of first image
                target = np.mean(levels[0])
                if scale:
                    scale_factors = np.tile(target / np.mean(levels, axis=1),
                                            num_ext).reshape(num_ext, num_img)
                else:  # zero=True
                    zero_offsets = np.tile(target - np.mean(levels, axis=1),
                                           num_ext).reshape(num_ext, num_img)
            if scale and np.min(scale_factors) < 0:
                log.warning("Some scale factors are negative. Not scaling.")
                scale_factors = np.ones_like(scale_factors)
                scale = False
            if scale and np.any(np.isinf(scale_factors)):
                log.warning("Some scale factors are infinite. Not scaling.")
                scale_factors = np.ones_like(scale_factors)
                scale = False
            if scale and np.any(np.isnan(scale_factors)):
                log.warning("Some scale factors are undefined. Not scaling.")
                scale_factors = np.ones_like(scale_factors)
                scale = False

        if reject_method == "varclip" and any(ext.variance is None
                                              for ad in adinputs for ext in ad):
            log.warning("Rejection method 'varclip' has been chosen but some"
                        " extensions have no variance. 'sigclip' will be used"
                        " instead.")
            reject_method = "sigclip"

        log.stdinfo("Combining {} inputs with {} and {} rejection"
                    .format(num_img, params["operation"], reject_method))

        stack_function = NDStacker(combine=params["operation"],
                                   reject=reject_method,
                                   log=self.log, **params)

        # NDStacker uses DQ if it exists; if we don't want that, delete the DQs!
        if not apply_dq:
            [setattr(ext, 'mask', None) for ad in adinputs for ext in ad]

        ad_out = astrodata.create(adinputs[0].phu)
        for index, (ext, sfactors, zfactors) in enumerate(
                zip(adinputs[0], scale_factors, zero_offsets)):
            status = (f"Combining extension {ext.id}." if num_ext > 1 else
                      "Combining images.")
            if scale:
                status += " Applying scale factors."
                numbers = sfactors
            elif zero:
                status += " Applying offsets."
                numbers = zfactors
            log.stdinfo(status)
            if (scale or zero) and (index == 0 or separate_ext):
                for ad, value in zip(adinputs, numbers):
                    log.stdinfo(f"{ad.filename:40s}{value:10.3f}")

            shape = adinputs[0][index].nddata.shape
            if memory is None:
                kernel = shape
            else:
                # Chop the image horizontally into equal-sized chunks to process
                # This uses the minimum number of steps and uses minimum memory
                # per step.
                oversubscription = (bytes_per_ext[index] * num_img) // memory + 1
                kernel = ((shape[0] + oversubscription - 1) // oversubscription,) + shape[1:]

            with_uncertainty = True  # Since all stacking methods return variance
            with_mask = apply_dq and not any(ad[index].nddata.window[:].mask is None
                                             for ad in adinputs)
            result = windowedOp(stack_function,
                                [ad[index].nddata for ad in adinputs],
                                scale=sfactors,
                                zero=zfactors,
                                kernel=kernel,
                                dtype=np.float32,
                                with_uncertainty=with_uncertainty,
                                with_mask=with_mask,
                                save_rejection_map=save_rejection_map)
            ad_out.append(result)

            if process_gain:
                gains = [g[index] for g in gain_list]
                # If all inputs have the same gain, the output will also have
                # this gain, and since the header has been copied, we don't
                # need to do anything! (Can't use set() if gains are lists.)
                if not all(g == gains[0] for g in gains):
                    log.warning("Not all inputs have the same gain.")
                    try:
                        output_gain = num_img / np.sum([1/g for g in gains])
                    except TypeError:
                        pass
                    else:
                        ad_out[-1].hdr[ad_out._keyword_for("gain")] = output_gain

            if process_rn:
                # Output gets the rms value of the inputs
                rns = [rn[index] for rn in rn_list]
                output_rn = np.sqrt(np.sum([np.square(np.asarray(rn).mean())
                                             for rn in rns]) / num_img)
                ad_out[-1].hdr[ad_out._keyword_for("read_noise")] = output_rn

            log.stdinfo("")

        # Propagate REFCAT as the union of all input REFCATs
        refcats = [ad.REFCAT for ad in adinputs if hasattr(ad, 'REFCAT')]
        if refcats:
            try:
                out_refcat = table.unique(table.vstack(refcats, metadata_conflicts='silent'),
                                          keys=('RAJ2000', 'DEJ2000'))
            except KeyError:
                pass
            else:
                out_refcat['Id'] = list(range(1, len(out_refcat)+1))
                ad_out.REFCAT = out_refcat

        # Propagate MDF from first input (no checking that they're all the same)
        if hasattr(adinputs[0], 'MDF'):
            ad_out.MDF = deepcopy(adinputs[0].MDF)

        # Set AIRMASS to be the mean of the input values
        try:
            airmass_kw = ad_out._keyword_for('airmass')
            mean_airmass = np.mean([ad.airmass() for ad in adinputs])
        except Exception:  # generic implementation failure (probably non-Gemini)
            pass
        else:
            ad_out.phu.set(airmass_kw, mean_airmass, "Mean airmass for the exposure")

        # Add suffix to datalabel to distinguish from the reference frame
        if sfx[0] == '_':
            extension = sfx.replace('_', '-', 1).upper()
        else:
            extension = '-' + sfx.upper()
        ad_out.phu.set('DATALAB', "{}{}".format(ad_out.data_label(), extension),
                       self.keyword_comments['DATALAB'])

        # Add other keywords to the PHU about the stacking inputs
        ad_out.orig_filename = ad_out.phu.get('ORIGNAME')
        ad_out.phu.set('NCOMBINE', len(adinputs), self.keyword_comments['NCOMBINE'])
        for i, ad in enumerate(adinputs, start=1):
            ad_out.phu.set('IMCMB{:03d}'.format(i), ad.phu.get('ORIGNAME', ad.filename))

        # Timestamp and update filename and prepare to return single output
        gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
        ad_out.update_filename(suffix=sfx, strip=True)

        return [ad_out]

    def stackSkyFrames(self, adinputs=None, **params):
        """
        Adds the `OBJMASK` object mask to the `DQ` data quality plane for every
        image and its extensions. Then, stacks all the images into a single
        frame.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science images with objects properly detected and added to the
            `OBJMASK` plane.

        suffix : str
            Suffix to be added to output files.

        apply_dq : bool
            Apply DQ mask to data before combining?

        dilation : int
            Dilation radius for expanding object mask.

        mask_objects : bool
            Mask objects from the input frames?

        nhigh : int
            Number of high pixels to reject.

        nlow : int
            Number of low pixels to reject.

        operation : str
            Combine method.

        reject_method : str
            Type of pixel rejection (passed to gemcombine).

        memory : float or None
            Available memory (in GB) for stacking calculations.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Sky stacked image. This list contains only one element. The list
            format is maintained so this primitive is consistent with all the
            others.

        See Also
        --------
        :meth:`~geminidr.core.primitives_stack.Stack.stackFrames`
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys["stackSkyFrames"]

        # Not what stackFrames does when both are set
        stack_params = self._inherit_params(params, 'stackFrames',
                                            pass_suffix=True)
        if stack_params["scale"] and stack_params["zero"]:
            log.warning("Both the scale and zero parameters are set. "
                        "Setting zero=False.")
            stack_params["zero"] = False

        # Need to deepcopy here to avoid changing DQ of inputs
        dilation=params["dilation"]
        if params["mask_objects"]:
            # Purely cosmetic to avoid log reporting unnecessary calls to
            # dilateObjectMask
            if dilation > 0:
                adinputs = self.dilateObjectMask(adinputs, dilation=dilation)
            adinputs = self.addObjectMaskToDQ([deepcopy(ad) for ad in adinputs])

        #if scale or zero:
        #    ref_bg = gt.measure_bg_from_image(adinputs[0], value_only=True)
        #    for ad in adinputs[1:]:
        #        this_bg = gt.measure_bg_from_image(ad, value_only=True)
        #        for ext, this, ref in zip(ad, this_bg, ref_bg):
        #            if scale:
        #                ext *= ref / this
        #            elif zero:
        #                ext += ref - this
        adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs
