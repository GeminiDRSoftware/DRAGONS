import numpy as np

from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from primitives_GEMINI import GEMINIPrimitives


class GNIRSPrimitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the GNIRS
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GEMINIPrimitives'.
    """
    astrotype = "GNIRS"
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
    
    def standardizeInstrumentHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of GNIRS data, specifically.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeInstrumentHeaders",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeInstrumentHeaders"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeInstrumentHeaders primitive has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by "
                            "standardizeInstrumentHeaders" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to NIRI
            log.status("Updating keywords that are specific to GNIRS")
            
            # Filter name (required for IRAF?)
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="filter_name(stripID=True, pretty=True)",
              keyword="FILTER", extname="PHU")
            
            # Pixel scale
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="pixel_scale()", extname="PHU")
            
            # Read noise
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="read_noise()", extname="SCI")
            
            # Gain
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="gain()", extname="SCI")
            
            # Non linear level
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="non_linear_level()", extname="SCI")
            
            # Saturation level
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="saturation_level()", extname="SCI")
            
            # Dispersion axis 
            if "SPECT" in ad.types:
                gt.update_key_from_descriptor(
                  adinput=ad, descriptor="dispersion_axis()", extname="SCI")
            
            # The exposure time by coadds in prepared data
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="exposure_time()", extname="PHU")

            # Adding the WCS information to the pixel data header, since for
            # GNIRS images at least, it can be only in the PHU
            wcskey = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 
                      'CD2_1', 'CD1_2', 'CD2_2', 'MJD_OBS', 'CTYPE1', 'CTYPE2']
            for key in wcskey:
                if (key not in ad['SCI',1].header) and (key in ad.phu.header):
                    ad['SCI',1].set_key_value(key, ad.phu_get_key_value(key))
                
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects 
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc 
    
    def standardizeStructure(self, rc):
        """
        This primitive is used to standardize the structure of GNIRS
        data, specifically.
        
        :param attach_mdf: Set to True to attach an MDF extension to the input
                           AstroData object(s). If an input AstroData object
                           does not have an AstroData type of SPECT, no MDF
                           will be added, regardless of the value of this
                           parameter.
        :type attach_mdf: Python boolean
        :param mdf: The file name, including the full path, of the MDF(s) to
                    attach to the input AstroData object(s). If only one MDF is
                    provided, that MDF will be attached to all input AstroData
                    object(s). If more than one MDF is provided, the number of
                    MDFs must match the number of input AstroData objects. If
                    no MDF is provided, the primitive will attempt to determine
                    an appropriate MDF.
        :type mdf: string or list of strings
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeStructure",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeStructure"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Use a flag to determine whether to run addMDF
        attach_mdf = True
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by standardizeStructure"
                            % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Attach an MDF to each input AstroData object
            if rc["attach_mdf"] and attach_mdf:
                
                # Get the mdf parameter from the reduction context
                mdf = rc["mdf"]
                if mdf is not None:
                    rc.run("addMDF(mdf=%s)" % mdf)
                else:
                    rc.run("addMDF")
                
                # Since addMDF uses all the AstroData inputs from the reduction
                # context, it only needs to be run once in this loop
                attach_mdf = False
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects 
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def validateData(self, rc):
        """
        This primitive is used to validate GNIRS data, specifically.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "validateData", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["validateData"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by validateData"
                            % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Validate the input AstroData object.
            log.status("No validation required for %s" % ad.filename)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
