import numpy as np

from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from primitives_GEMINI import GEMINIPrimitives


class NIRIPrimitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the NIRI
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GEMINIPrimitives'.
    """
    astrotype = "NIRI"
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
    
    def nonlinearityCorrect(self, rc):
        """
        Run on raw or nprepared Gemini NIRI data, this script calculates and
        applies a per-pixel linearity correction based on the counts in the
        pixel, the exposure time, the read mode, the bias level and the ROI.
        Pixels over the maximum correctable value are set to BADVAL unless
        given the force flag. Note that you may use glob expansion in infile,
        however, any pattern matching characters (*,?) must be either quoted
        or escaped with a backslash. Do we need a badval parameter that defines
        a value to assign to uncorrectable pixels, or do we want to just add
        those pixels to the DQ plane with a specific value?
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "nonlinearityCorrect",
                                "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["nonlinearityCorrect"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Define a lookup dictionary for necessary values
        # This should be moved to the lookups area
        lincorlookup = {
            # In the following form for NIRI data:
            #("read_mode", naxis2, "well_depth_setting"):
            #    (maximum counts, exposure time correction, gamma, eta)
            ("Low Background", 1024, "Shallow"):
                (12000, 1.2662732, 7.3877618e-06, 1.940645271e-10),
            ("Medium Background", 1024, "Shallow"):
                (12000, 0.09442515154, 3.428783846e-06, 4.808353308e-10),
            ("Medium Background", 256, "Shallow"):
                (12000, 0.01029262589, 6.815415667e-06, 2.125210479e-10),
            ("High Background", 1024, "Shallow"):
                (12000, 0.009697324059, 3.040036696e-06, 4.640788333e-10),
            ("High Background", 1024, "Deep"):
                (21000, 0.007680816203, 3.581914163e-06, 1.820403678e-10),
            }

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the nonlinearityCorrect primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by nonlinearityCorrect" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Get the appropriate information using the descriptors
            coadds = ad.coadds().as_pytype()
            read_mode = ad.read_mode().as_pytype()
            total_exposure_time = ad.exposure_time()
            well_depth_setting = ad.well_depth_setting().as_pytype()
            if coadds is None or read_mode is None or \
                total_exposure_time is None or well_depth_setting is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the
                # exception.
                if hasattr(ad, "exception_info"):
                    raise ad.exception_info
            
            # Check the raw exposure time (i.e., per coadd). First, convert
            # the total exposure time returned by the descriptor back to
            # the raw exposure time
            exposure_time = total_exposure_time / coadds
            if exposure_time > 600.:
                log.critical("The raw exposure time is outside the " \
                             "range used to derive correction.")
                raise Errors.InvalidValueError()
            
            # Check the read mode and well depth setting values
            if read_mode == "Invalid" or well_depth_setting == "Invalid":
                raise Errors.CalcError()
            
            # Print the descriptor values
            log.fullinfo("The number of coadds = %s" % coadds)
            log.fullinfo("The read mode = %s" % read_mode)
            log.fullinfo("The total exposure time = %s" % total_exposure_time)
            log.fullinfo("The well depth = %s" % well_depth_setting)
            
            # Loop over each science extension in each input AstroData object
            for ext in ad[SCI]:
                
                # Get the size of the raw pixel data
                naxis2 = ext.get_key_value("NAXIS2")
                
                # Get the raw pixel data
                raw_pixel_data = ext.data
                
                # Divide the raw pixel data by the number of coadds
                if coadds > 1:
                    raw_pixel_data = raw_pixel_data / coadds
                
                # Determine the mean of the raw pixel data
                raw_mean_value = np.mean(raw_pixel_data, dtype=np.float64)
                log.fullinfo("The mean value of the raw pixel data in " \
                             "%s is %.8f" % (ext.filename, raw_mean_value))
                
                # Create the key used to access the coefficients that are
                # used to correct for non-linearity
                key = (read_mode, naxis2, well_depth_setting)
                
                # Get the coefficients from the lookup table
                if lincorlookup[key]:
                    maximum_counts, coeff1, coeff2, coeff3 = \
                        lincorlookup[key]
                else:
                    raise Errors.TableKeyError()
                log.fullinfo("Coefficients used = %.12f, %.9e, %.9e" \
                             % (coeff1, coeff2, coeff3))
                
                # Create a new array that contains the corrected pixel data
                corrected_pixel_data = raw_pixel_data + \
                    coeff2 * raw_pixel_data**2 + coeff3 * raw_pixel_data**3
                
                # nirlin replaces pixels greater than maximum_counts with 0
                # Set the pixels to 0 if they have a value greater than the
                # maximum counts
                #log.fullinfo("Setting pixels to zero if above %f" % \
                #    maximum_counts)
                #corrected_pixel_data[corrected_pixel_data > \
                # maximum_counts] = 0
                # Should probably add the above to the DQ plane
                
                # Multiply the corrected pixel data by the number of coadds
                if coadds > 1:
                    corrected_pixel_data *= coadds
                    
                # Correct for the exposure time issue by scaling the counts
                corrected_pixel_data *= exposure_time / (exposure_time + coeff1)
                
                # Write the corrected pixel data to the output object
                ext.data = corrected_pixel_data
                
                # Determine the mean of the corrected pixel data
                corrected_mean_value = np.mean(ext.data, dtype=np.float64)
                log.fullinfo("The mean value of the corrected pixel data in " \
                             "%s is %.8f" \
                             % (ext.filename, corrected_mean_value))
            
            # Correct the exposure time by adding coeff1 * coadds
            total_exposure_time = total_exposure_time + coeff1 * coadds
            log.fullinfo("The true total exposure time = %f" \
                         % total_exposure_time)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def standardizeInstrumentHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of NIRI data, specifically.
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
            log.status("Updating keywords that are specific to NIRI")
            
            # Filter name (required for IRAF?)
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="filter_name(stripID=True, pretty=True)",
                keyword="FILTER", extname="PHU", 
                keyword_comments=self.keyword_comments)
            
            # Pixel scale
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="pixel_scale()", extname="PHU",
                keyword_comments=self.keyword_comments)
            
            # Read noise (new keyword, should it be written?)
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="read_noise()", extname="SCI",
                keyword_comments=self.keyword_comments)
            
            # Gain (new keyword, should it be written?)
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="gain()", extname="SCI",
                keyword_comments=self.keyword_comments)
            
            # Non linear level
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="non_linear_level()", extname="SCI",
                keyword_comments=self.keyword_comments)
            
            # Saturation level
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="saturation_level()", extname="SCI",
                keyword_comments=self.keyword_comments)
            
            # Dispersion axis (new keyword, should it be written?)
            if "SPECT" in ad.types:
                gt.update_key_from_descriptor(
                    adinput=ad, descriptor="dispersion_axis()", extname="SCI",
                    keyword_comments=self.keyword_comments)
            
            # Convention is multiply the exposure time by coadds in prepared data
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="exposure_time()", extname="PHU",
                keyword_comments=self.keyword_comments)

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
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
        This primitive is used to standardize the structure of NIRI
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
            
            # Standardize the structure of the input AstroData object. Raw
            # NIRI data have three dimensions (e.g., 2048x2048x1), so
            # check whether the third dimension has a length of one and remove
            # it.
            for ext in ad:
                if len(ext.data.shape) == 3:
                    
                    # Remove the single-dimensional axis from the pixel data
                    log.status("Removing the third dimension from %s"
                                 % ad.filename)
                    ext.data = np.squeeze(ext.data)
                    
                    if len(ext.data.shape) == 3:
                        # The np.squeeze method only removes a dimension from
                        # the array if it is equal to 1. In this case, the
                        # third dimension contains multiple datasets. Need to
                        # deal with this as some point
                        pass
                    
                    log.debug("Dimensions of %s[%s,%d] = %s" % (
                      ad.filename, ext.extname(), ext.extver(),
                      ext.data.shape))
            
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
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
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
        This primitive is used to validate NIRI data, specifically.
        
        :param repair: Set to True to repair the data, if necessary. Note: this
                       feature does not work yet. 
        :type repair: Python boolean
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
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
