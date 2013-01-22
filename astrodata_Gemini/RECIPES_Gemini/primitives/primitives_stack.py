import sys
import math
import numpy as np
from astrodata import Errors
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
