import sys

from astrodata import AstroData
from astrodata.adutils import gemLog
from astrodata.ConfigSpace import lookupPath
from gempy import geminiTools as gt
from gempy.science import geminiScience as gs
from gempy.science import standardization as sdz
from primitives_GEMINI import GEMINIPrimitives

class F2Primitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the FLAMINGOS-2
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, "GEMINIPrimitives".
    """
    astrotype = "F2"
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
    
    def addBPM(self, rc):
        """
        This primitive is called by addDQ (which is located in
        primitives_GEMINI.py) to add the appropriate Bad Pixel Mask (BPM) to
        the inputs. This function will add the BPM as frames matching that of
        the SCI frames and ensure the BPM's data array is the same size as
        that of the SCI data array. If the SCI array is larger (say SCI's were
        overscan trimmed, but BPMs were not), the BPMs will have their arrays
        padded with zero's to match the sizes and use the data_section
        descriptor on the SCI data arrays to ensure the match is a correct fit.
        
        Using this approach, rather than appending the BPM in the addDQ allows
        for specialized BPM processing to be done in the instrument specific
        primitive sets where it belongs.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type logLevel: integer or string
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addBPM", "starting"))
        try:
            # Load the BPM file into AstroData
            bpm = AstroData(lookupPath("Gemini/F2/BPM/F2_bpm.fits"))
            # Call the add_bpm user level function
            output = gs.add_bpm(
                input=rc.getInputs(style="AD"),
                output_names=rc["output_names"],
                suffix=rc["suffix"],
                bpm=bpm)
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc 
    
    def standardizeHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of FLAMINGOS-2 data, specifically.
        
        :param loglevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type loglevel: integer or string
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeHeaders",
                                 "starting"))
        try:
            # Call the standardize_headers_f2 user level function
            output = sdz.standardize_headers_f2(
                adinput=rc.getInputs(style="AD"),
                output_names=rc["output_names"],
                suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc 
    
    def standardizeStructure(self, rc):
        """
        This primitive is used to standardize the structure of FLAMINGOS-2
        data, specifically.
        
        :param loglevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type loglevel: integer or string
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeStructure",
                                 "starting"))
        try:
            # Call the standardize_structure_f2 user level function
            output = sdz.standardize_structure_f2(
                adinput=rc.getInputs(style="AD"),
                output_names=rc["output_names"],
                suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def validateData(self, rc):
        """
        This primitive is used to validate FLAMINGOS-2 data, specifically.
        
        :param loglevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type loglevel: integer or string
        """
        # Instantiate the log. This needs to be done outside of the try
        # block, since the log object is used in the except block
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        try:
            # Log the standard "starting primitive" debug message
            log.debug(gt.log_message("primitive", "validateData", "starting"))
            # Initialize the list of output AstroData objects
            adoutput_list = []
            # Loop over each input AstroData object in the input list
            for ad in rc.getInputs(style="AD"):
                # Check whether the validateData primitive has been run
                # previously
                if ad.phuGetKeyValue("VALDATA"):
                    log.warning("%s has already been processed by " \
                                "validateData" % (ad.filename))
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
                # Call the validate_data_f2 user level function
                ad = sdz.validate_data_f2(adinput=ad,
                                          output_names=rc["output_names"],
                                          suffix=rc["suffix"],
                                          repair=rc["repair"])
                # Append the output AstroData object to the list of output
                # AstroData objects
                adoutput_list.append(ad)
            # Report the list of output AstroData objects to the reduction
            # context
            rc.reportOutput(adoutput_list)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
