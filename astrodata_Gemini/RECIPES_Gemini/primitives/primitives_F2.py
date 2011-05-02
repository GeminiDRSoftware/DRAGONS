import sys

from astrodata import AstroData
from astrodata.adutils import gemLog
from astrodata.ConfigSpace import lookupPath
from gempy import geminiTools as gt
from gempy.science import geminiScience as gs
from gempy.science import standardize as sdz
from primitives_GEMINI import GEMINIPrimitives

class F2Primitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the FLAMINGOS-2
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GEMINIPrimitives'.
    """
    astrotype = 'F2'
    
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
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        # Log the standard 'starting primitive' debug message
        log.debug(gt.logMessage('primitive', 'addBPM', 'starting'))
        try:
            # Load the BPM file into AstroData
            bpm = AstroData(lookupPath('Gemini/F2/BPM/F2_bpm.fits'))
            # Call the add_bpm user level function
            output = gs.add_bpm(adInputs=rc.getInputs(style='AD'),
                                BPMs=bpm,
                                suffix=rc['suffix'])
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
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        # Log the standard 'starting primitive' debug message
        log.debug(gt.logMessage('primitive', 'standardizeHeaders', 'starting'))
        try:
            # Call the standardize_headers_f2 user level function
            output = sdz.standardize_headers_f2(input=rc.getInputs(style='AD'),
                                                suffix=rc['suffix'])
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
        This primitive is used to make the changes and additions to the
        keywords in the headers of FLAMINGOS-2 data, specifically.
        
        :param loglevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type loglevel: integer or string
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        # Log the standard 'starting primitive' debug message
        log.debug(gt.logMessage('primitive', 'standardizeStructure',
                                'starting'))
        try:
            # Call the standardize_structure_f2 user level function
            output = sdz.standardize_headers_f2(input=rc.getInputs(style='AD'),
                                                suffix=rc['suffix'])
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
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        # Log the standard 'starting primitive' debug message
        log.debug(gt.logMessage('primitive', 'validateData', 'starting'))
        try:
            # Call the validate_data_f2 user level function
            output = sdz.validate_data_f2(input=rc.getInputs(style='AD'),
                                          suffix=rc['suffix'])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
