# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu

import sys
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
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
        
        :param suffix: Value to be post pended onto each input name(s) to
                       create the output name(s).
        :type suffix: string
        
        :param fl_over: Subtract the overscan level from the frames?
        :type fl_over: Python boolean (True/False)
        
        :param fl_trim: Trim the overscan region from the frames?
        :type fl_trim: Python boolean (True/False)
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "normalizeFlat", "starting"))
        try:
            # Load the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()
            # Call the normalize_flat_image_gmos user level function
            output = cal.normalize_flat_image_gmos(
                input=rc.getInputs(style="AD"),
                output_names=rc["output_names"],
                suffix=rc["suffix"],
                fl_trim=rc["fl_trim"],
                fl_over=rc["fl_over"],
                fl_vardq="AUTO")
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc 
    
    def standardizeStructure(self,rc):
        """
        This primitive will to add an MDF to the inputs if they are of type
        SPECT, those of type IMAGE will be handled by the standardizeStructure
        in the primitives_GMOS_IMAGE set where no MDF will be added. The
        user level function standardize_structure_gmos in
        gempy.science.standardization is utilized to do the work for this
        primitive.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param addMDF: A flag to turn on/off appending the appropriate MDF 
                       file to the inputs.
        :type addMDF: Python boolean (True/False)
                      default: True
                      
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "standardizeStructure",
                                "starting"))
        try:
            # Call the standardize_structure_gmos user level function
            output = sdz.standardize_structure_gmos(
                input=rc.getInputs(style="AD"),
                output_names=rc["output_names"],
                suffix=rc["suffix"],
                addMDF=rc["addMDF"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc 
