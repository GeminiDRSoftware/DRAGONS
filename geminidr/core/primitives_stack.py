#
#                                                                  gemini_python
#
#                                                            primitives_stack.py
# ------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy

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

        # Add the input frames to the forStack list and
        # get other available frames from the same list
        self.addToList(adinputs, purpose='forStack')
        stack_inputs = self.getList(adinputs, purpose='forStack')

        # Return entire list if only one object (which would presumably be the
        # adinputs, or return the input list if we can't stack
        if len(stack_inputs) <= 1:
            log.stdinfo("No alignment or correction will be performed, since "
                        "at least two input AstroData objects are required "
                        "for alignAndStack")
            return stack_inputs
        else:
            if (self.parameters.alignAndStack['check_if_stack'] and
                    not _can_stack(stack_inputs)):
                log.warning("No stacking will be performed, since a frame varies "
                            "from the reference image by more than 1 degree")
                return adinputs
            else:
                adinputs = stack_inputs
                #TODO: Must be an easier way than this to determine whether
                # an AD object has no OBJCATs
                if any(all(getattr(ext, 'OBJCAT', None) is None for ext in ad)
                       for ad in adinputs):
                    adinputs = self.detectSources(adinputs, **params)
                adinputs = self.correctWCSToReferenceFrame(adinputs, **params)
                adinputs = self.alignToReferenceFrame(adinputs, **params)
                adinputs = self.correctBackgroundToReferenceImage(adinputs, **params)
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
        mask: bool
            apply mask to data before combining?
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

        # Preserve the input dtype for the data quality extension
        #dq_dtypes_list = []
        #for ad in ad_input_list:
        #    if ad[DQ]:
        #        for ext in ad[DQ]:
        #            dq_dtypes_list.append(ext.data.dtype)

        #if dq_dtypes_list:
        #    unique_dq_dtypes = set(dq_dtypes_list)
        #    unique_dq_dtypes_list = [dtype for dtype in unique_dq_dtypes]
        #    if len(unique_dq_dtypes_list) == 1:
        #        # The input data quality extensions have the same dtype
        #        dq_dtype = unique_dq_dtypes_list[0]
        #    elif len(unique_dq_dtypes_list) == 2:
        #        dq_dtype = np.promote_types(unique_dq_dtypes_list[0],
        #                                    unique_dq_dtypes_list[1])
        #    else:
        #        # The input data quality extensions have more than two
        #        # different dtypes. Since np.promote_types only accepts two
        #        # dtypes as input, for now, just use uint16 in this case
        #        # (when gemcombine is replaced with a python function, the
        #        # combining of the DQ extension can be handled correctly by
        #        # numpy).
        #        dq_dtype = np.dtype(np.uint16)

        # Instantiate ETI and then run the task
        gemcombine_task = gemcombineeti.GemcombineETI(adinputs, params)
        ad = gemcombine_task.run()

        # Gemcombine sets the GAIN keyword to the sum of the gains;
        # reset it to the average instead. Set the RDNOISE to the
        # sum in quadrature of the input read noise. Set the keywords in
        # the variance and data quality extensions to be the same as the
        # science extensions.
        for ext, gain, rn in zip(ad, gain_list, read_noise_list):
            ext.hdr.set('GAIN', gain, self.keyword_comments['GAIN'])
            ext.hdr.set('RDNOISE', rn, self.keyword_comments['RDNOISE'])
        # Stick the first extension's values in the PHU
        ad.phu.set('GAIN', gain_list[0], self.keyword_comments['GAIN'])
        ad.phu.set('RDNOISE', read_noise_list[0], self.keyword_comments['RDNOISE'])

        # Add suffix to datalabel to distinguish from the reference frame
        ad.phu.set('DATALAB', "{}{}".format(ad.phu.get('DATALAB', ''), sfx),
                   self.keyword_comments['DATALAB'])

        # Timestamp and update filename and prepare to return single output
        gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        adinputs = [ad]

        return adinputs

    def stackSkyFrames(self, adinputs=None, **params):
        """
        This primitive stacks the sky frames for each science frame (as
        determined from the self.sky_dict attribute previously set) by
        calling stackFrames and then attaches AD objects of the stacked sky
        frames to each science frame via the self.stacked_sky_dict attribute

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mask: bool
            apply mask to data before combining?
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
        timestamp_key = self.timestamp_keys["stackSkyFrames"]

        # Initialize the list of output science and stacked sky AstroData
        # objects
        ad_sky_for_correction_output_list = []

        # Initialize the dictionary that will contain the association between
        # the science AstroData objects and the stacked sky AstroData objects
        stacked_sky_dict = {}

        # The associateSky primitive creates an attribute with the information
        # associating the sky frames to the science frames
        try:
            sky_dict = self.sky_dict
        except AttributeError:
            sky_dict = {}

        for ad_sci in adinputs:
            # Retrieve the list of sky AstroData objects associated with the
            # input science AstroData object
            origname = ad_sci.phu.get('ORIGNAME')
            if origname in sky_dict:
                ad_sky_list = sky_dict[origname]

                if not ad_sky_list:
                    log.warning("No sky frames available for {}".format(origname))
                    continue

                # Generate a unique suffix for the stacked sky AstroData object
                sky_suffix = "_for{}".format(origname.replace('.fits', ''))

                if len(ad_sky_list) == 1:
                    # There is only one associated sky AstroData object for
                    # this science AstroData object, so there is no need to
                    # call stackFrames. Update the dictionary with the single
                    # sky AstroDataRecord object associated with this science
                    # AstroData object
                    ad_sky = deepcopy(ad_sky_list[0])

                    # Update the filename
                    ad_sky.filename = gt.filename_updater(
                      adinput=ad_sky, suffix=sky_suffix, strip=True)

                    # Create the AstroDataRecord for this new AstroData Object
                    log.fullinfo("Only one sky frame available for {}: {}".
                                format(origname, ad_sky.filename))

                    # Update the dictionary with the stacked sky
                    # AstroDataRecord object associated with this science
                    # AstroData object
                    stacked_sky_dict.update({origname: ad_sky})

                    # Update the output stacked sky AstroData list to contain
                    # the sky for correction
                    ad_sky_for_correction_output_list.append(ad_sky)
                else:
                    ad_sky_to_stack_list = []

                    # Combine the list of sky AstroData objects
                    log.stdinfo("Combining the following sky frames for {}".
                                 format(origname))
                    for ad_sky in ad_sky_list:
                        log.stdinfo("  {}".format(ad_sky.filename))
                        ad_sky_to_stack_list.append(ad_sky)

                    ad_stacked_sky_list = self.stackFrames(ad_sky_to_stack_list,
                                                           **params)

                    # Add the sky to be used to correct this science AstroData
                    # object to the list of output sky AstroData objects
                    if len(ad_stacked_sky_list) == 1:
                        ad_stacked_sky = ad_stacked_sky_list[0]
                        # Change filename so the _skyStacked frame associated
                        # with a particular science frame has the same root
                        ad_stacked_sky.filename = gt.filename_updater(
                            infilename=origname, suffix=params['suffix'])

                        # Add the appropriate time stamp to the PHU
                        gt.mark_history(adinput=ad_stacked_sky,
                                        primname=self.myself(),
                                        keyword=timestamp_key)

                        #ad_sky_for_correction_output_list.append(
                        #  ad_stacked_sky)

                        # Update the dictionary with the stacked sky AD object
                        stacked_sky_dict.update({origname: ad_stacked_sky})
                    else:
                        log.warning("Problem with stacking")

        # Add the appropriate time stamp to the PHU and update the filename of
        # the science AstroData objects
        #adinputs = gt.finalise_adinput(adinputs, timestamp_key=timestamp_key,
        #                                suffix=params["suffix"])

        # Add the association dictionary to the reduction context
        self.stacked_sky_dict = stacked_sky_dict

        #TODO: This list doesn't seem to be picked up anywhere else
        # Report the list of output stacked sky AstroData objects to the
        # forSkyCorrection stream in the reduction context
        # rc.report_output(
        #  ad_sky_for_correction_output_list, stream="forSkyCorrection")
        # rc.run("showInputs(stream='forSkyCorrection')")
        return adinputs

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################
def _can_stack(adinputs):
    """
    This function checks for a set of AstroData input frames whether there is
    more than 1 degree of rotation between the first frame and successive
    frames. If so, stacking will not be performed.

    :param adinput: List of AstroData instances
    :type adinput: List of AstroData instances
    """
    ref_pa = adinputs[0].phu['PA']
    return all(abs(ad.phu['PA'] - ref_pa) < 1.0 for ad in adinputs)