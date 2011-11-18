import sys
import math
import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from primitives_GENERAL import GENERALPrimitives

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
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
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
            recipe_list.append("correctWCSToReferenceImage")
            
            # Align all images to the first one
            recipe_list.append("alignToReferenceImage")
            
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
        :type reject_method: string, options: 'avsigclip', 'minmax'

        :param mask_type: type of masking to use
        :type mask_type: string, options: 'goodvalue', 'none'
        
        :param nlow: number of low pixels to reject (used with
                     reject_method=minmax)
        :type nlow: int

        :param nhigh: number of high pixels to reject (used with
                      reject_method=minmax)
        :type nhigh: int
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackFrames", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["stackFrames"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.stdinfo("No stacking will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "stackFrames")
            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput
        else:

            # Get parameters from the RC
            suffix = rc["suffix"]
            operation = rc["operation"]
            reject_method = rc["reject_method"]
            mask_type = rc["mask_type"]
            nlow = rc["nlow"]
            nhigh = rc["nhigh"]

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

            # Load PyRAF
            pyraf, gemini, yes, no = pyrafLoader()

            # Use the CL manager to get the input parameters
            clm = mgr.CLManager(imageIns=adinput, funcName="combine",
                                suffix=suffix, combinedImages=True, log=log)
            if not clm.status:
                raise Errors.InputError("Inputs must be prepared")
        
            # Get the input parameters for IRAF as specified by
            # the stackFrames primitive 
            clPrimParams = {
                # Retrieving the inputs as a list from the CLManager
                "input"   : clm.imageInsFiles(type="listFile"),
                "output"  : clm.imageOutsFiles(type="string"),
                # This returns a unique/temp log file for IRAF
                "logfile" : clm.templog.name,
                }
        
            # Get the input parameters for IRAF as specified by the user
            fl_vardq = no
            fl_dqprop = no
            for ad in adinput:
                if ad["DQ"]:
                    fl_dqprop = yes
                    if ad["VAR"]:
                        fl_vardq = yes
            clSoftcodedParams = {
                "fl_vardq"  : fl_vardq,
                "fl_dqprop" : fl_dqprop,
                "combine"   : operation,
                "reject"    : reject_method,
                "nlow"      : nlow,
                "nhigh"     : nhigh,
                "masktype"  : mask_type,
                }
        
            # Get the default parameters for IRAF and update them
            # using the above dictionaries
            clParamsDict = CLDefaultParamsDict("gemcombine")
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)

            # Log the parameters
            mgr.logDictParams(clParamsDict)

            # Call gemcombine
            gemini.gemcombine(**clParamsDict)
            if gemini.gemcombine.status:
                raise Errors.OutputError("The IRAF task gemcombine failed")
            else:
                log.fullinfo("The IRAF task gemcombine completed " +
                             "sucessfully")
        
            # Create the output AstroData object by loading the
            # output file from gemcombine into AstroData, remove
            # intermediate temporary files from disk 
            adstack, junk, junk = clm.finishCL()
            adout = adstack[0]
        
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

            # Add suffix to the ORIGNAME header so future filenames
            # can't strip it out
            adout.phu_set_key_value(
                "ORIGNAME", 
                gt.fileNameUpdater(adinput[0],suffix=suffix,strip=True),
                comment=self.keyword_comments["ORIGNAME"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adstack, keyword=timestamp_key)

            adoutput_list = adstack

        # Report the output list to the reduction context
        rc.report_output(adoutput_list)
                
        yield rc

