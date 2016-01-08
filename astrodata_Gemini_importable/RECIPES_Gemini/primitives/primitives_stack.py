import sys
import math
import numpy as np
from copy import deepcopy
from astrodata.utils import Errors
from recipe_system.reduction import reductionContextRecords as RCR
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.utils.gemconstants import SCI, VAR, DQ
from gempy.gemini import gemini_tools as gt
from gempy.gemini import eti
from primitives_GENERAL import GENERALPrimitives
import time

class StackPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def alignAndStack(self, rc):
        """
        This primitive calls a set of primitives to perform the steps
        needed for alignment of frames to a reference image and stacking.
        
        :param check_if_stack: Parameter to call a check as to whether 
                               stacking should be performed. If not, this
                               part of the recipe is skipped and the single
                               input file returned.
        :type check_if_stack: bool
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "alignAndStack", "starting"))
         
        # Add the input frame to the forStack list and 
        # get other available frames from the same list
        single_ad = rc.get_inputs_as_astrodata()
        rc.run("addToList(purpose=forStack)")
        rc.run("getList(purpose=forStack)")

        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.stdinfo("No alignment or correction will be performed, since "
                        "at least two input AstroData objects are required "
                        "for alignAndStack")
            rc.report_output(adinput)
        else:
            # If required, perform a check as to whether stacking should
            # be performed, and if it is not, return only the single 
            # original AstroData input
            check_if_stack = rc["check_if_stack"]
            if (check_if_stack and not _is_stack(adinput)):
                rc.report_output(single_ad)

            else:
                recipe_list = []

                # Check to see if detectSources needs to be run
                run_ds = False
                for ad in adinput:
                    objcat = ad["OBJCAT"]
                    if objcat is None:
                        run_ds = True
                        break
                if run_ds:
                    recipe_list.append("detectSources")
            
                # Register all images to the first one
                recipe_list.append("correctWCSToReferenceFrame")
            
                # Align all images to the first one
                recipe_list.append("alignToReferenceFrame")
            
                # Correct background level in all images to the first one
                recipe_list.append("correctBackgroundToReferenceImage")

                # Stack all frames
                recipe_list.append("stackFrames") 
            
                # Run all the needed primitives
                rc.run("\n".join(recipe_list))
        
        yield rc
    
    def stackFrames(self, rc):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.
        
        :param operation: type of combining operation to use.
        :type operation: string, options: 'average', 'median'.
        
        :param reject_method: type of rejection algorithm
        :type reject_method: string, options: 'avsigclip', 'minmax', None
        
        :param mask: Use the data quality extension to mask bad pixels?
        :type mask: bool
        
        :param nlow: number of low pixels to reject (used with
                     reject_method=minmax)
        :type nlow: int
        
        :param nhigh: number of high pixels to reject (used with
                      reject_method=minmax)
        :type nhigh: int
        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackFrames", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["stackFrames"]
        
        # Initialize the list of output AstroData objects
        ad_output_list = []
        
        # Get the input AstroData objects
        ad_input_list = rc.get_inputs_as_astrodata()
        
        # Ensure that each input AstroData object has been prepared
        for ad in ad_input_list:
            if not "PREPARED" in ad.types:
                raise Errors.InputError("%s must be prepared" % ad.filename)
        
        if len(ad_input_list) <= 1:
            log.stdinfo("No stacking will be performed, since at least two "
                        "input AstroData objects are required for stackFrames")
            
            # Set the list of input AstroData objects to the list of output
            # AstroData objects without further processing 
            ad_output_list = ad_input_list
        
        else:
            
            # Get the gain and read noise from the first AstroData object in
            # the input list using the appropriate descriptors
            
            # Determine the average gain from the input AstroData objects and
            # add in quadrature the read noise
            gain_dict = {}
            read_noise_dict = {}

            gain_dvs = [ad.gain() for ad in ad_input_list]
            read_noise_dvs = [ad.read_noise() for ad in ad_input_list]

            # Check for Nones:
            if True in [gain_dv.is_none() for gain_dv in gain_dvs]:
                 raise Errors.InputError("One or more gain DVs are None")

            if True in [read_noise_dv.is_none() for read_noise_dv in
                        read_noise_dvs]:
                 raise Errors.InputError("One or more read noise DVs are None")

            # Sum the values
            for extver in gain_dvs[0].ext_vers():
                for gain_dv in gain_dvs:
                    if extver not in gain_dict:
                        gain_dict.update({extver: 0})
                    gain_dict[extver] += gain_dv.get_value(extver)

                for read_noise_dv in read_noise_dvs:
                    if extver not in read_noise_dict:
                        read_noise_dict.update({extver: 0})
                    read_noise_dict[extver] += read_noise_dv.get_value(
                                                   extver)**2

            for key in gain_dict.keys():
                gain_dict[key] /= len(ad_input_list)
                read_noise_dict[key] = math.sqrt(read_noise_dict[key])
            
            # Preserve the input dtype for the data quality extension
            dq_dtypes_list = []
            for ad in ad_input_list:
                if ad[DQ]:
                    for ext in ad[DQ]:
                        dq_dtypes_list.append(ext.data.dtype)

            if dq_dtypes_list:
                unique_dq_dtypes = set(dq_dtypes_list)
                unique_dq_dtypes_list = [dtype for dtype in unique_dq_dtypes]
                if len(unique_dq_dtypes_list) == 1:
                    # The input data quality extensions have the same dtype
                    dq_dtype = unique_dq_dtypes_list[0]
                elif len(unique_dq_dtypes_list) == 2:
                    dq_dtype = np.promote_types(unique_dq_dtypes_list[0],
                                                unique_dq_dtypes_list[1])
                else:
                    # The input data quality extensions have more than two
                    # different dtypes. Since np.promote_types only accepts two
                    # dtypes as input, for now, just use uint16 in this case
                    # (when gemcombine is replaced with a python function, the
                    # combining of the DQ extension can be handled correctly by
                    # numpy).
                    dq_dtype = np.dtype(np.uint16)
            
            # Instantiate ETI and then run the task 
            gemcombine_task = eti.gemcombineeti.GemcombineETI(rc)
            ad_output = gemcombine_task.run()
            
            # Revert the BUNIT for the variance extension (gemcombine sets it
            # to the same value as the science extension)
            bunit = ad_output[SCI,1].get_key_value("BUNIT")
            if ad_output[VAR]:
                for ext in ad_output[VAR]:
                    if bunit is not None:
                        gt.update_key(adinput=ext, keyword="BUNIT",
                                      value="%s*%s" % (bunit, bunit),
                                      comment=None, extname=VAR)
            
            # Revert the dtype and BUNIT for the data quality extension
            # (gemcombine sets them to int32 and the same value as the science
            # extension, respectively)
            if ad_output[DQ]:
                for ext in ad_output[DQ]:
                    ext.data = ext.data.astype(dq_dtype)
                    
                    if bunit is not None:
                        gt.update_key(adinput=ext, keyword="BUNIT",
                                      value="bit", comment=None, extname=DQ)
            
            # Gemcombine sets the GAIN keyword to the sum of the gains; 
            # reset it to the average instead. Set the RDNOISE to the
            # sum in quadrature of the input read noise. Set the keywords in
            # the variance and data quality extensions to be the same as the
            # science extensions.
            for ext in ad_output:
                extver = ext.extver()
                gain = gain_dict[extver]
                read_noise = read_noise_dict[extver]
                
                gt.update_key(adinput=ext, keyword="GAIN", value=gain,
                              comment=None, extname="pixel_exts")
                gt.update_key(adinput=ext, keyword="RDNOISE", value=read_noise,
                              comment=None, extname="pixel_exts")
            
            gain = gain_dict[1]
            read_noise = read_noise_dict[1]
            
            gt.update_key(adinput=ad_output, keyword="GAIN", value=gain,
                          comment=None, extname="PHU")
            gt.update_key(adinput=ad_output, keyword="RDNOISE",
                          value=read_noise, comment=None, extname="PHU")
            
            suffix = rc["suffix"]
            
            # The ORIGNAME keyword should not be updated in this way, since it
            # defeats the point of having the ORIGNAME keyword.
            
            # Add suffix to the ORIGNAME to prevent future stripping 
            #ad_output.phu_set_key_value("ORIGNAME", 
            #    gt.filename_updater(adinput=adinput[0],
            #                        suffix=suffix,strip=True),
            #    comment=self.keyword_comments["ORIGNAME"])
            
            # Add suffix to the datalabel to distinguish from the reference
            # frame 
            orig_datalab = ad_output.phu_get_key_value("DATALAB")
            new_datalab = "%s%s" % (orig_datalab, suffix)
            gt.update_key(adinput=ad_output, keyword="DATALAB",
                          value=new_datalab, comment=None, extname="PHU")
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad_output, keyword=timestamp_key)
            
            # Change the filename
            ad_output.filename = gt.filename_updater(adinput=ad_output,
                                                     suffix=suffix, strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            ad_output_list.append(ad_output)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(ad_output_list)
        
        yield rc
    
    def stackSkyFrames(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackSkyFrames", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["stackSkyFrames"]
        
        # Initialize the list of output science and stacked sky AstroData
        # objects
        ad_science_output_list = []
        ad_sky_for_correction_output_list = []
        
        # Initialize the dictionary that will contain the association between
        # the science AstroData objects and the stacked sky AstroData objects
        stacked_sky_dict = {}
        
        # The associateSky primitive puts the dictionary containing the
        # information associating the sky frames to the science frames in
        # the reduction context
        sky_dict = rc["sky_dict"]
        
        # Loop over each science AstroData object in the science list
        ad_science_list = rc.get_inputs_as_astrodata()
        for ad_science in ad_science_list:
            
            # Retrieve the list of sky AstroData objects associated with the
            # input science AstroData object
            origname = ad_science.phu_get_key_value("ORIGNAME")
            if sky_dict and (origname in sky_dict):
                adr_sky_list = sky_dict[origname]
                
                if not adr_sky_list:
                    # There are no associated sky AstroData objects for this
                    # science AstroData object
                    log.warning("No sky frames available for %s" % origname)
                    continue
                
                # Generate a unique suffix for the stacked sky AstroData object
                if origname.endswith(".fits"):
                    sky_suffix = "_for%s" % origname[:-5]
                else:
                    sky_suffix = "_for%s" % origname
                
                if len(adr_sky_list) == 1:
                    # There is only one associated sky AstroData object for
                    # this science AstroData object, so there is no need to
                    # call stackFrames. Update the dictionary with the single
                    # sky AstroDataRecord object associated with this science
                    # AstroData object
                    ad_sky = deepcopy(adr_sky_list[0].ad)
                    
                    # Update the filename
                    ad_sky.filename = gt.filename_updater(
                      adinput=ad_sky, suffix=sky_suffix, strip=True)
                    
                    # Create the AstroDataRecord for this new AstroData Object
                    adr_sky = RCR.AstroDataRecord(ad_sky)
                    log.fullinfo("Only one sky frame available for %s: %s" % (
                      origname, adr_sky.ad.filename))
                    
                    # Update the dictionary with the stacked sky
                    # AstroDataRecord object associated with this science 
                    # AstroData object
                    stacked_sky_dict.update({origname: adr_sky})
                    
                    # Update the output stacked sky AstroData list to contain
                    # the sky for correction
                    ad_sky_for_correction_output_list.append(adr_sky.ad)
                
                else:
                    # Initialize the list of sky AstroData objects to be
                    # stacked
                    ad_sky_to_stack_list = []
                    
                    # Combine the list of sky AstroData objects
                    log.stdinfo("Combining the following sky frames for %s"
                                 % origname)
                    
                    for adr_sky in adr_sky_list:
                        log.stdinfo("  %s" % adr_sky.ad.filename)
                        ad_sky_to_stack_list.append(adr_sky.ad)
                    
                    # Add the sky AstroData objects to the forStack stream
                    rc.report_output(ad_sky_to_stack_list, stream="forStack")
                    
                    # Call stackFrames using the sky AstroData objects in the
                    # forStack stream. The stacked sky AstroData objects will
                    # be added back into the forStack stream.
                    rc.run("showInputs(stream='forStack')")
                    rc.run("stackFrames(stream='forStack', suffix='%s',"
                           "operation='%s')" % (sky_suffix, rc["operation"]))
                    rc.run("showInputs(stream='forStack')")
                    
                    # Get the stacked sky AstroData object from the forStack
                    # stream and empty the forStack stream, in preparation for
                    # creating the next stacked sky AstroData object
                    adr_stacked_sky_list = rc.get_stream(
                      stream="forStack", empty=True)
                    
                    # Add the sky to be used to correct this science AstroData
                    # object to the list of output sky AstroData objects
                    if len(adr_stacked_sky_list) == 1:
                        adr_stacked_sky = adr_stacked_sky_list[0]
                        
                        # Add the appropriate time stamps to the PHU
                        gt.mark_history(adinput=adr_stacked_sky.ad,
                                        keyword=timestamp_key)
                        
                        ad_sky_for_correction_output_list.append(
                          adr_stacked_sky.ad)
                        
                        # Update the dictionary with the stacked sky
                        # AstroDataRecord object associated with this science 
                        # AstroData object
                        stacked_sky_dict.update({origname: adr_stacked_sky})
                    else:
                        log.warning("Problem with stacking")
        
        # Add the appropriate time stamp to the PHU and update the filename of
        # the science AstroData objects
        ad_science_output_list = gt.finalise_adinput(
          adinput=ad_science_list, timestamp_key=timestamp_key,
          suffix=rc["suffix"])
        
        # Add the association dictionary to the reduction context
        rc["stacked_sky_dict"] = stacked_sky_dict
        
        # Report the list of output stacked sky AstroData objects to the
        # forSkyCorrection stream in the reduction context 
        rc.report_output(
          ad_sky_for_correction_output_list, stream="forSkyCorrection")
        rc.run("showInputs(stream='forSkyCorrection')")
        
        # Report the list of output science AstroData objects to the reduction
        # context 
        rc.report_output(ad_science_output_list)
        
        yield rc
        
##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################
def _is_stack(adinput):
    """
    This function checks for a set of AstroData input frames whether there is
    more than 1 degree of rotation between the first frame and successive 
    frames. If so, stacking will not be performed.
    
    :param adinput: List of AstroData instances
    :type adinput: List of AstroData instances
    """    

    # Instantiate the log
    log = logutils.get_logger(__name__)
    
    ref_pa = adinput[0].phu_get_key_value("PA")
    for i in range(len(adinput)-1):
        pa = adinput[i+1].phu_get_key_value("PA")
        if abs(pa - ref_pa) >= 1.0:
            log.warning("No stacking will be performed, since a frame varies "
                        "from the reference image by more than 1 degree")
            return False
    
    return True

