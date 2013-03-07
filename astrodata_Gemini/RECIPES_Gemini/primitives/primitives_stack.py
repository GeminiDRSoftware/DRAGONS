import sys
import math
import numpy as np
from copy import deepcopy
from astrodata import Errors
from astrodata import ReductionContextRecords as RCR
from astrodata.adutils import logutils
from astrodata.adutils.gemutil import pyrafLoader
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
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "alignAndStack", "starting"))
         
        # Add the input frame to the forStack list and 
        # get other available frames from the same list
        rc.run("addToList(purpose=forStack)")
        rc.run("getList(purpose=forStack)")

        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.stdinfo("No alignment or correction will be performed, " \
                        "since at least two input AstroData objects are " \
                        "required for alignAndStack")
            rc.report_output(adinput)
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

        :param mask: Use DQ plane to mask bad pixels?
        :type mask: bool
        
        :param nlow: number of low pixels to reject (used with
                     reject_method=minmax)
        :type nlow: int

        :param nhigh: number of high pixels to reject (used with
                      reject_method=minmax)
        :type nhigh: int
        """
        t1 = time.time()
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", "stackFrames", "starting"))
        
        adinput = rc.get_inputs_as_astrodata()
        adoutput_list = []
        timestamp_key = self.timestamp_keys["stackFrames"]

        # Check if inputs prepared
        for ad in adinput:
            if (ad.phu_get_key_value('GPREPARE')==None) and \
               (ad.phu_get_key_value('PREPARE')==None):
               raise Errors.InputError("%s must be prepared" % ad.filename)
        
        if len(adinput) <= 1:
            log.stdinfo("No stacking will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "stackFrames")
            adoutput_list = adinput
        else:
            
            # Get average of current GAIN parameters from input files
            # and add in quadrature the read-out noise
            gain = adinput[0].gain().as_dict()
            ron = adinput[0].read_noise().as_dict()
            for ad in adinput[1:]:
                for ext in ad["SCI"]:
                    gain[("SCI",ext.extver())] += ext.gain()
                    ron[("SCI",ext.extver())] += ext.read_noise()**2
            for key in gain.keys():
                gain[key] /= len(adinput)
                ron[key] = math.sqrt(ron[key])
        
            # Instantiate ETI and then run the task 
            gemcombine_task = eti.gemcombineeti.GemcombineETI(rc)
            adout = gemcombine_task.run()
            
            # Change type of DQ plane back to int16 (gemcombine sets
            # it to int32)
            if adout["DQ"] is not None:
                for dqext in adout["DQ"]:
                    dqext.data = dqext.data.astype(np.int16)

                    # Also delete the BUNIT keyword (gemcombine
                    # sets it to same value as SCI)
                    if dqext.get_key_value("BUNIT") is not None:
                        del dqext.header['BUNIT']

            # Fix BUNIT in VAR plane as well
            # (gemcombine sets it to same value as SCI)
            bunit = adout["SCI",1].get_key_value("BUNIT")
            if adout["VAR"] is not None and bunit is not None:
                for ext in adout["VAR"]:
                    ext.set_key_value(
                        "BUNIT","%s*%s" % (bunit,bunit),
                        comment=self.keyword_comments["BUNIT"])

            # Gemcombine sets the GAIN keyword to the sum of the gains; 
            # reset it to the average instead.  Set the RDNOISE to the
            #  sum in quadrature of the input read noise. Set VAR/DQ
            # keywords to the same as the science.
            for ext in adout:
                ext.set_key_value("GAIN", gain[("SCI",ext.extver())],
                                  comment=self.keyword_comments["GAIN"])
                ext.set_key_value("RDNOISE", ron[("SCI",ext.extver())],
                                  comment=self.keyword_comments["RDNOISE"])
            
            if adout.phu_get_key_value("GAIN") is not None:
                adout.phu_set_key_value(
                    "GAIN",gain[("SCI",1)],
                    comment=self.keyword_comments["GAIN"])
            if adout.phu_get_key_value("RDNOISE") is not None:
                adout.phu_set_key_value(
                    "RDNOISE",ron[("SCI",1)],
                    comment=self.keyword_comments["RDNOISE"])

            suffix = rc["suffix"]
            
            # The ORIGNAME keyword should not be updated in this way, since it
            # defeats the point of having the ORIGNAME keyword.
            
            # Add suffix to the ORIGNAME to prevent future stripping 
            #adout.phu_set_key_value("ORIGNAME", 
            #    gt.filename_updater(adinput=adinput[0],
            #                        suffix=suffix,strip=True),
            #    comment=self.keyword_comments["ORIGNAME"])

            # Add suffix to the datalabel to distinguish from the reference
            # frame 
            orig_dl = adout.phu_get_key_value("DATALAB")
            adout.phu_set_key_value(
                "DATALAB", orig_dl+suffix,
                comment=self.keyword_comments["DATALAB"])

            gt.mark_history(adinput=adout, keyword=timestamp_key)
            adoutput_list.append(adout)

        # Report the output list to the reduction context
        rc.report_output(adoutput_list)
        #print("ETI TIME: %s sec" % str(time.time()-t1))
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
            if origname in sky_dict:
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
                    rc.run("stackFrames(stream='forStack', suffix='%s')"
                           % sky_suffix)
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
