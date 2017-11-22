#
#                                                                  gemini_python
#
#                                                            primitives_stack.py
# ------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy

from astropy import table

from gempy.gemini import gemini_tools as gt
from gempy.gemini.eti import gemcombineeti
from gempy.utils import logutils

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

        # Sum the values
        nexts = len(gains[0])
        gain_list = [np.mean([gain[i] for gain in gains])
                     for i in range(nexts)]
        read_noise_list = [np.sqrt(np.sum([rn[i]*rn[i] for rn in read_noises]))
                                     for i in range(nexts)]

        if params["zero"]:
            adinputs = self.correctBackgroundToReference(adinputs)

        # Instantiate ETI and then run the task
        gemcombine_task = gemcombineeti.GemcombineETI(adinputs, params)
        ad_out = gemcombine_task.run()

        # Propagate REFCAT as the union of all input REFCATs
        refcats = [ad.REFCAT for ad in adinputs if hasattr(ad, 'REFCAT')]
        if refcats:
            out_refcat = table.unique(table.vstack(refcats,
                                metadata_conflicts='silent'), keys='Cat_Id')
            out_refcat['Cat_Id'] = range(1, len(out_refcat)+1)
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
