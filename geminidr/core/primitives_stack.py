#
#                                                                  gemini_python
#
#                                                            primitives_stack.py
# ------------------------------------------------------------------------------
import astrodata
from astrodata.fits import windowedOp

import numpy as np
from astropy import table
from functools import partial

from gempy.gemini import gemini_tools as gt
from gempy.gemini.eti import gemcombineeti

from geminidr import PrimitivesBASE
from .parameters_stack import ParametersStack

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Stack(PrimitivesBASE):
    """
    This is the class containing all of the primitives for stacking.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Stack, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersStack

    def alignAndStack(self, adinputs=None, **params):
        """
        This primitive calls a set of primitives to perform the steps
        needed for alignment of frames to a reference image and stacking.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Return entire list if only one object (which would presumably be the
        # adinputs, or return the input list if we can't stack
        if len(adinputs) <= 1:
            log.stdinfo("No alignment or correction will be performed, since "
                        "at least two input AstroData objects are required "
                        "for alignAndStack")
            return adinputs
        else:
            adinputs = self.matchWCSToReference(adinputs, **params)
            adinputs = self.resampleToCommonFrame(adinputs, **params)
            adinputs = self.stackFrames(adinputs, **params)
        return adinputs

    def stackFlats(self, adinputs=None, **params):
        """Default behaviour is just to stack images as normal"""
        return self.stackFrames(adinputs, **params)

    def stackFramesOld(self, adinputs=None, **params):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        apply_dq: bool
            apply DQ mask to data before combining?
        nhigh: int
            number of high pixels to reject
        nlow: int
            number of low pixels to reject
        operation: str
            combine method
        reject_method: str
            type of pixel rejection (passed to gemcombine)
        zero: bool
            apply zero-level offset to match background levels?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["stackFrames"]
        sfx = params["suffix"]

        if len(adinputs) <= 1:
            log.stdinfo("No stacking will be performed, since at least two "
                        "input AstroData objects are required for stackFrames")
            return adinputs

        # Ensure that each input AstroData object has been prepared
        for ad in adinputs:
            if not "PREPARED" in ad.tags:
                raise IOError("{} must be prepared" .format(ad.filename))

        # Determine the average gain from the input AstroData objects and
        # add in quadrature the read noise
        gains = [ad.gain() for ad in adinputs]
        read_noises = [ad.read_noise() for ad in adinputs]

        assert all(gain is not None for gain in gains), "Gain problem"
        assert all(rn is not None for rn in read_noises), "RN problem"

        # Compute gain and read noise of final stacked images
        nexts = len(gains[0])
        gain_list = [np.mean([gain[i] for gain in gains])
                     for i in range(nexts)]
        read_noise_list = [np.sqrt(np.sum([rn[i]*rn[i] for rn in read_noises]))
                                     for i in range(nexts)]

        # Match the background levels
        if params["zero"]:
            adinputs = self.correctBackgroundToReference(adinputs,
                                remove_background=params["remove_background"])

        # Instantiate ETI and then run the task
        gemcombine_task = gemcombineeti.GemcombineETI(adinputs, params)
        ad_out = gemcombine_task.run()

        # Propagate REFCAT as the union of all input REFCATs
        refcats = [ad.REFCAT for ad in adinputs if hasattr(ad, 'REFCAT')]
        if refcats:
            out_refcat = table.unique(table.vstack(refcats,
                                metadata_conflicts='silent'), keys='Cat_Id')
            out_refcat['Cat_Id'] = list(range(1, len(out_refcat)+1))
            ad_out.REFCAT = out_refcat

        # Gemcombine sets the GAIN keyword to the sum of the gains;
        # reset it to the average instead. Set the RDNOISE to the
        # sum in quadrature of the input read noise. Set the keywords in
        # the variance and data quality extensions to be the same as the
        # science extensions.
        for ext, gain, rn in zip(ad_out, gain_list, read_noise_list):
            ext.hdr.set('GAIN', gain, self.keyword_comments['GAIN'])
            ext.hdr.set('RDNOISE', rn, self.keyword_comments['RDNOISE'])
        # Stick the first extension's values in the PHU
        ad_out.phu.set('GAIN', gain_list[0], self.keyword_comments['GAIN'])
        ad_out.phu.set('RDNOISE', read_noise_list[0], self.keyword_comments['RDNOISE'])

        # Add suffix to datalabel to distinguish from the reference frame
        ad_out.phu.set('DATALAB', "{}{}".format(ad_out.data_label(), sfx),
                   self.keyword_comments['DATALAB'])

        # Timestamp and update filename and prepare to return single output
        gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
        ad_out.update_filename(suffix=sfx, strip=True)

        return [ad_out]

    def stackFrames(self, adinputs=None, **params):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        apply_dq: bool
            apply DQ mask to data before combining?
        nhigh: int
            number of high pixels to reject
        nlow: int
            number of low pixels to reject
        operation: str
            combine method
        reject_method: str
            type of pixel rejection (passed to gemcombine)
        zero: bool
            apply zero-level offset to match background levels?
        """
        from gempy.library.nddops import NDStacker

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["stackFrames"]
        sfx = params["suffix"]

        zero = params["zero"]
        scale = params["scale"]
        apply_dq = params["apply_dq"]
        separate_ext = params["separate_ext"]
        statsec = params["statsec"]
        if statsec:
            try:
                statsec = tuple([slice(int(start)-1, int(end))
                             for x in reversed(statsec.strip('[]').split(','))
                             for start, end in [x.split(':')]])
            except ValueError:
                log.warning("Cannot parse statistics section {}. Using full "
                            "frame.".format(statsec))
                statsec = None

        if len(adinputs) <= 1:
            log.stdinfo("No stacking will be performed, since at least two "
                        "input AstroData objects are required for stackFrames")
            return adinputs

        # Ensure that each input AstroData object has been prepared
        for ad in adinputs:
            if not "PREPARED" in ad.tags:
                raise IOError("{} must be prepared" .format(ad.filename))

        # Determine the average gain from the input AstroData objects and
        # add in quadrature the read noise
        gains = [ad.gain() for ad in adinputs]
        read_noises = [ad.read_noise() for ad in adinputs]

        assert all(gain is not None for gain in gains), "Gain problem"
        assert all(rn is not None for rn in read_noises), "RN problem"

        # Compute gain and read noise of final stacked images
        nexts = len(gains[0])
        gain_list = [np.mean([gain[i] for gain in gains])
                     for i in range(nexts)]
        read_noise_list = [np.sqrt(np.sum([rn[i]*rn[i] for rn in read_noises]))
                                     for i in range(nexts)]


        # Compute the scale and offset values by accessing the memmapped data
        # so we can pass those to the stacking function
        num_img = len(adinputs)
        num_ext = len(adinputs[0])
        zero_offsets = np.zeros((num_ext, num_img), dtype=np.float32)
        scale_factors = np.ones_like(zero_offsets)
        if scale or zero:
            levels = np.empty((num_img, num_ext), dtype=np.float32)
            for i, ad in enumerate(adinputs):
                for index in range(num_ext):
                    nddata = (ad[index].nddata.window[:] if statsec is None
                              else ad[i][index].nddata.window[statsec])
                    # TODO: measure_bg_from_image?
                    levels[i, index] = np.median(nddata.data)
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
            if scale and np.isinf(np.max(scale_factors)):
                log.warning("Some scale factors are infinite. Not scaling.")
                scale_factors = np.ones_like(scale_factors)

        stack_function = NDStacker(combine=params["operation"], reject=params["reject_method"],
                                   log=self.log, **params)

        # NDStacker uses DQ if it exists; if we don't want that, delete the DQs!
        if not apply_dq:
            [setattr(ext, 'mask', None) for ad in adinputs for ext in ad]

        # Let's be a bit lazy here. Let's compute the stack outputs and stuff
        # the NDData objects into the first (reference) image
        ad_out = astrodata.create(adinputs[0].phu)
        for index, (scale, zero) in enumerate(zip(scale_factors, zero_offsets)):
            with_uncertainty = True  # Since all stacking methods return variance
            with_mask = apply_dq and not any(ad[index].nddata.window[:].mask is None
                                             for ad in adinputs)
            result = windowedOp(partial(stack_function, scale=scale, zero=zero),
                                [ad[index].nddata for ad in adinputs],
                                kernel=(2048,2048), dtype=np.float32,
                                with_uncertainty=with_uncertainty, with_mask=with_mask)
            ad_out.append(result)

        # Propagate REFCAT as the union of all input REFCATs
        refcats = [ad.REFCAT for ad in adinputs if hasattr(ad, 'REFCAT')]
        if refcats:
            out_refcat = table.unique(table.vstack(refcats,
                                metadata_conflicts='silent'), keys='Cat_Id')
            out_refcat['Cat_Id'] = list(range(1, len(out_refcat)+1))
            ad_out.REFCAT = out_refcat

        # Set GAIN to the average of input gains. Set the RDNOISE to the
        # sum in quadrature of the input read noises.
        for ext, gain, rn in zip(ad_out, gain_list, read_noise_list):
            ext.hdr.set('GAIN', gain, self.keyword_comments['GAIN'])
            ext.hdr.set('RDNOISE', rn, self.keyword_comments['RDNOISE'])
        # Stick the first extension's values in the PHU
        ad_out.phu.set('GAIN', gain_list[0], self.keyword_comments['GAIN'])
        ad_out.phu.set('RDNOISE', read_noise_list[0], self.keyword_comments['RDNOISE'])

        # Add suffix to datalabel to distinguish from the reference frame
        ad_out.phu.set('DATALAB', "{}{}".format(ad_out.data_label(), sfx),
                   self.keyword_comments['DATALAB'])

        # Timestamp and update filename and prepare to return single output
        gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
        ad_out.update_filename(suffix=sfx, strip=True)

        return [ad_out]

    def stackSkyFrames(self, adinputs=None, **params):
        """
        This primitive stacks the AD frames sent to it with object masking.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        apply_dq: bool
            apply DQ mask to data before combining?
        dilation: int
            dilation radius for expanding object mask
        mask_objects: bool
            mask objects from the input frames?
        nhigh: int
            number of high pixels to reject
        nlow: int
            number of low pixels to reject
        operation: str
            combine method
        reject_method: str
            type of pixel rejection (passed to gemcombine)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys["stackSkyFrames"]

        scale = params["scale"]
        zero = params["zero"]
        if scale and zero:
            log.warning("Both the scale and zero parameters are set. "
                        "Setting zero=False.")
            zero = False

        # Parameters to be passed to stackFrames
        stack_params = {k: v for k,v in params.items() if
                        k in self.parameters.stackFrames and k != "suffix"}
        # We're taking care of the varying sky levels here so stop
        # stackFrames from getting involved
        stack_params.update({'zero': False,
                             'remove_background': False})

        # Run detectSources() on any frames without any OBJMASKs
        if params["mask_objects"]:
            adinputs = [ad if any(hasattr(ext, 'OBJMASK') for ext in ad) else
                        self.detectSources([ad])[0] for ad in adinputs]
            adinputs = self.dilateObjectMask(adinputs,
                                             dilation=params["dilation"])
            adinputs = self.addObjectMaskToDQ(adinputs)

        if scale or zero:
            ref_bg = gt.measure_bg_from_image(adinputs[0], value_only=True)
            for ad in adinputs[1:]:
                this_bg = gt.measure_bg_from_image(ad, value_only=True)
                for ext, this, ref in zip(ad, this_bg, ref_bg):
                    if scale:
                        ext *= ref / this
                    elif zero:
                        ext += ref - this
        adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs
