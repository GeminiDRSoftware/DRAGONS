from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import standardization as sdz
from primitives_GEMINI import GEMINIPrimitives

class F2Primitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the FLAMINGOS-2
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GEMINIPrimitives'.
    """
    astrotype = "F2"
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
    
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
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the standardizeHeaders primitive has been run
            # previously
            if ad.phu_get_key_value("SDZHDRSI"):
                log.warning("%s has already been processed by " \
                            "standardizeHeaders" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the standardize_headers_f2 user level function
            ad = sdz.standardize_headers_f2(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
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
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value("SDZSTRUC"):
                log.warning("%s has already been processed by " \
                            "standardizeStructure" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the standardize_structure_f2 user level function
            ad = sdz.standardize_structure_f2(adinput=ad,
                                              add_mdf=rc["add_mdf"],
                                              mdf=rc["mdf"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
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
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "validateData", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value("VALDATA"):
                log.warning("%s has already been processed by validateData" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the validate_data_f2 user level function
            ad = sdz.validate_data_f2(adinput=ad, repair=rc["repair"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
