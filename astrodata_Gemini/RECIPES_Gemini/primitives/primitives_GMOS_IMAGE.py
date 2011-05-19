# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu

import sys
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import calibrate as cal
from gempy.science import standardization as sdz
from primitives_GMOS import GMOSPrimitives

class GMOS_IMAGEPrimitives(GMOSPrimitives):
    """
    This is the class containing all of the primitives for the GMOS_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    astrotype = "GMOS_IMAGE"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc
    
    def normalizeFlat(self, rc):
        """
        This primitive will combine the input flats and then normalize them
        using the CL script giflat.
        
        Warning: giflat calculates its own DQ frames and thus replaces the
        previously produced ones in calculateDQ. This may be fixed in the
        future by replacing giflat with a Python equivilent with more
        appropriate options for the recipe system. 
        
        :param overscan: Subtract the overscan level from the frames?
        :type overscan: Python boolean (True/False)
        
        :param trim: Trim the overscan region from the frames?
        :type trim: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalizeFlat", "starting"))

        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('GIFLAT'):
                log.warning('%s has already been processed by normalizeFlat' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue
            
            ad = cal.normalize_flat_image_gmos(adinput=ad,
                                               trim=rc["trim"],
                                               overscan=rc["overscan"])
            adoutput_list.append(ad[0])

        rc.report_output(output)
        yield rc
    
    def standardizeStructure(self,rc):
        """
        This primitive will not add a MDF, for GMOS images. The
        user level function standardize_structure_gmos in
        gempy.science.standardization is utilized to do the work for this
        primitive.  Currently this function just passes input back without
        modification, except TLM stamp.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        log.debug(gt.log_message("primitive", "standardizeStructure", "starting"))

        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('STDSTRUC'):
                log.warning('%s has already been processed by standardizeStructure' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue            

            ad = sdz.standardize_structure_gmos(adinput=ad)
            adoutput_list.append(ad[0])

        rc.report_output(output)
        yield rc
