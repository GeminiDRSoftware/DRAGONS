import numpy as np
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
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
    
    def standardizeInstrumentHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of FLAMINGOS-2 data, specifically.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeInstrumentHeaders",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeInstrumentHeaders"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeInstrumentHeaders primitive
            # has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by "\
                            "standardizeInstrumentHeaders" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Update the keywords in the headers that are specific to
            # FLAMINGOS-2
            log.status("Updating keywords that are specific to FLAMINGOS-2")

            # Filter name (required for IRAF?)
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="filter_name(stripID=True, pretty=True)",
                keyword="FILTER", extname="PHU")

            # Pixel scale
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="pixel_scale()", extname="PHU")

            # Read noise (new keyword, should it be written?)
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="read_noise()", extname="SCI")

            # Gain (new keyword, should it be written?)
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="gain()", extname="SCI")

            # Non linear level
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="non_linear_level()", extname="SCI")

            # Saturation level
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="saturation_level()", extname="SCI")

            # Dispersion axis (new keyword, should it be written?)
            if "IMAGE" not in ad.types:
                gt.update_key_from_descriptor(
                    adinput=ad, descriptor="dispersion_axis()", extname="SCI")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc 
    
    def standardizeStructure(self, rc):
        """
        This primitive is used to standardize the structure of FLAMINGOS-2
        data, specifically.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeStructure",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeStructure"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # First run addMDF if necessary to attach mdf files to the input
        for ad in rc.get_inputs_as_astrodata():
            if rc["attach_mdf"]:
                # Get the mdf parameter from the RC
                mdf = rc["mdf"]
                if mdf is not None:
                    rc.run("addMDF(mdf=%s)" % mdf)
                else:
                    rc.run("addMDF")

                # It only needs to run once, on all input, so break loop
                break

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by standardizeStructure" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Standardize the structure of the input AstroData
            # object. Raw FLAMINGOS-2 data have three dimensions (i.e.,
            # 2048x2048x1), so check whether the third dimension has a length
            # of one and remove it
            for ext in ad:
                if len(ext.data.shape) == 3:
                    # Remove the single-dimensional axis from the pixel data
                    ext.data = np.squeeze(ext.data)
                    if len(ext.data.shape) == 2:
                        log.fullinfo("Removed third dimension from %s" \
                                     % ad.filename)
                    if len(ext.data.shape) == 3:
                        # The np.squeeze method only removes a dimension from
                        # the array if it is equal to 1. In this case, the
                        # third dimension contains multiple datasets. Need to
                        # deal with this as some point
                        pass
                    log.debug("Dimensions of %s[%s,%d] = %s" \
                              % (ad.filename, ext.extname(), ext.extver(), \
                              str(ext.data.shape)))
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def validateData(self, rc):
        """
        This primitive is used to validate FLAMINGOS-2 data, specifically.

        :param repair: Set to True (the default) to repair the data 
                       Note: this feature does not work yet.
        :type repair: Python boolean
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["validateData"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "validateData", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by validateData" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Get the repair parameter from the RC
            repair = rc["repair"]

            # Validate the input AstroData object. ACTUALLY DO SOMETHING HERE?
            log.stdinfo("No validation required for FLAMINGOS-2")
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
