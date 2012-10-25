class docstrings:

    def airmass(self):
        """
        Return the airmass value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the mean airmass of the observation
        """
        pass
    
    def amp_read_area(self):
        """
        Return the amp_read_area value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the composite string containing the name of the detector
                 amplifier (ampname) and the readout area of the CCD (detsec) 
                 used for the observation
        """
        pass
    
    def array_section(self):
        """
        Return the array_section value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: list as default (i.e., format=None)
        :return: the array_section
        """
        pass
    
    def azimuth(self):
        """
        Return the azimuth value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the azimuth (in degrees between 0 and 360) of the observation
        """
        pass
    
    def bias_level(self):
        """
        Return the bias_level value.

        For GMOS, this value is looked up from values in lookup tables distributed
        with the data reduction package - the values in the header are incorrect.

        The values are referenced by date (to account for hardware and other
        modifications that affect the values), and from the gain and read speed settings
        that were configured for this exposure.

        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the bias_level
        """
        pass
    
    def camera(self):
        """
        Return the camera value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the camera used for the observation
        """
        pass
    
    def cass_rotator_pa(self):
        """
        Return the cass_rotator_pa value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the cassegrain rotator position angle (in degrees between -360
                 and 360) of the observation
        """
        pass
    
    def central_wavelength(self):
        """
        Return the central_wavelength value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param asMicrometers: set to True to return the central_wavelength 
                              value in units of Micrometers
        :type asMicrometers: Python boolean
        :param asNanometers: set to True to return the central_wavelength 
                             value in units of Nanometers
        :type asNanometers: Python boolean
        :param asAngstroms: set to True to return the central_wavelength 
                            value in units of Angstroms
        :type asAngstroms: Python boolean
        :param format: set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s)
        :return: the central wavelength (in meters as default) of the 
                 observation
        """
        pass
    
    def coadds(self):
        """
        Return the coadds value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the number of coadds used for the observation
        """
        pass
    
    def data_label(self):
        """
        Return the data_label value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the data label of the observation
        """
        pass
    
    def data_section(self):
        """
        Return the data_section value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful data_section 
                       value in the form [x1:x2,y1:y2] that uses 1-based 
                       indexing
        :type pretty: Python boolean
        :param format: set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: tuple of integers that use 0-based indexing in the form 
                (x1 - 1, x2 - 1, y1 - 1, y2 - 1) as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2] 
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the section of the data of the observation
        """
        pass
    
    def dec(self):
        """
        Return the dec value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the declination (in decimal degrees) of the observation
        """
        pass
    
    def decker(self):
        """
        Return the decker value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned decker value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       decker value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the decker position used for the observation
        """
        pass
    
    def detector_rois_requested(self):
        """
        Return the list of detector ROIs (Region of Interest)s that were
        requested for this exposure. Not all instrument support ROIs other
        than the full array, and of those that do, not all support the concept
        of multiple (separate) ROIs per exposure. Even instruments that do
        support this concept (eg the GMOSes) cannot necessarily read out completely
        arbitrary ROI configurations, so the actual regions read may or may not
        correspond exactly to (though should be a superset of) those requested.

        This descriptor provides a list or ROIs requested, in the form:
        [[x1, x2, y1, y2], ...]

        - These are physical, unbinned pixel co-ordinates, so will not correspond
        directly to image pixels if the binning is not 1x1.
 
        - These numbers are 1-based, not 0-based - the corner pixel is [1,1].
        
        - The ranges given are inclusive at both ends.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: list as default (i.e., format=None)
        :return: the detector_rois_requested
        """
        pass
    
    def detector_roi_setting(self):
        """
        This descriptor attempts to deduce the "Name" of the ROI
        (Region Of Interest), as defined by the selection in the OT.
        
        For example, with GMOS, this might be "Full Frame", "Central Spectrum", 
        "CCD2" etc. The string "Custom" will be returned for custom defined ROIs

        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the detector_roi_setting
        """
        pass
    
    def detector_section(self):
        """
        Return the detector_section value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful 
                       detector_section value in the form [x1:x2,y1:y2] that 
                       uses 1-based indexing
        :type pretty: Python boolean
        :param format: set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: tuple of integers that use 0-based indexing in the form 
                (x1 - 1, x2 - 1, y1 - 1, y2 - 1) as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2] 
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the detector section of the observation
        """
        pass
    
    def detector_name(self):
        """
        Return the detector name. For GMOS this is generally the CCD name from 
        the CCDNAME header. 

        This is a bit subtle - if we have a non-mosaiced GMOS image, there
        will be a CCDNAME keyword in each SCI extension, and that's what we want.
        If we have a mosaiced image, these won't be present as the SCI is a mosaic of
        several (usually three) detectors. In this case, we return the DETID keyword
        from the PHU, which is generally a string concatenation of the individual 
        ccd names or similar.

        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the detector_name
        """
        pass
    
    def detector_x_bin(self):
        """
        Return the detector_x_bin value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the binning of the x-axis of the detector used for the 
                 observation
        """
        pass
    
    def detector_y_bin(self):
        """
        Return the detector_y_bin value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the binning of the y-axis of the detector used for the 
                 observation
        """
        pass
    
    def disperser(self):
        """
        Return the disperser value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned disperser value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       disperser value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the disperser used for the observation
        """
        pass
    
    def dispersion(self):
        """
        Return the dispersion value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param asMicrometers: set to True to return the dispersion 
                              value in units of Micrometers
        :type asMicrometers: Python boolean
        :param asNanometers: set to True to return the dispersion 
                             value in units of Nanometers
        :type asNanometers: Python boolean
        :param asAngstroms: set to True to return the dispersion 
                            value in units of Angstroms
        :type asAngstroms: Python boolean
        :param format: set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the dispersion (in meters per pixel as default) of the 
                 observation
        """
        pass
    
    def dispersion_axis(self):
        """
        Return the dispersion_axis value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the dispersion axis (x = 1; y = 2; z = 3) of the observation
        """
        pass
    
    def elevation(self):
        """
        Return the elevation value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the elevation (in degrees) of the observation
        """
        pass
    
    def exposure_time(self):
        """
        Return the exposure_time value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the total exposure time (in seconds) of the observation
        """
        pass
    
    def filter_name(self):
        """
        Return the filter_name value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned filter_name value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       filter_name value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the unique, sorted filter name idenifier string used for the 
                 observation
        """
        pass
    
    def focal_plane_mask(self):
        """
        Return the focal_plane_mask value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned focal_plane_mask value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       focal_plane_mask value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the focal plane mask used for the observation
        """
        pass
    
    def gain(self):
        """
        Return the gain value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the gain (in electrons per ADU) of the observation
        """
        pass
    
    def grating(self):
        """
        Return the grating value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned grating value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       grating value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the grating used for the observation
        """
        pass
    
    def group_id(self):
        """
        Return the group_id value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the group_id
        """
        pass
    
    def gain_setting(self):
        """
        Return the gain_setting value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the gain setting of the observation
        """
        pass
    
    def local_time(self):
        """
        Return the local_time value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the local time (in HH:MM:SS.S) at the start of the observation
        """
        pass
    
    def mdf_row_id(self):
        """
        Return the mdf_row_id value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the corresponding reference row in the MDF
        """
        pass
    
    def nod_count(self):
        """
        Return the nod_count value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the number of nod and shuffle cycles in the nod and shuffle 
                 observation
        """
        pass
    
    def nod_pixels(self):
        """
        Return the nod_pixels value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the number of pixel rows the charge is shuffled by in the nod 
                 and shuffle observation
        """
        pass
    
    def nominal_atmospheric_extinction(self):
        """
        Return the nominal atmospheric extinction value. These are determined
        from lookup tables based on the telescope site (ie Gemini-North or
        Gemini-South) and the filter name.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the nominal_atmospheric_extinction
        """
        pass
    
    def nominal_photometric_zeropoint(self):
        """
        Return the nominal photometric zeropoint value. These are determined
        from lookup tables keyed on the detector names and filter names
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the nominal_photometric_zeropoint
        """
        pass
    
    def non_linear_level(self):
        """
        Return the non_linear_level value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the non linear level in the raw images (in ADU) of the 
                 observation
        """
        pass
    
    def observation_class(self):
        """
        Return the observation_class value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the class (either 'science', 'progCal', 'partnerCal', 'acq', 
                 'acqCal' or 'dayCal') of the observation
        """
        pass
    
    def observation_epoch(self):
        """
        Return the observation_epoch value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the epoch (in years) at the start of the observation
        """
        pass
    
    def observation_id(self):
        """
        Return the observation_id value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the ID (e.g., GN-2011A-Q-123-45) of the observation
        """
        pass
    
    def observation_type(self):
        """
        Return the observation_type value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the type (either 'OBJECT', 'DARK', 'FLAT', 'ARC', 'BIAS' or 
                 'MASK') of the observation
        """
        pass
    
    def overscan_section(self):
        """
        Return the overscan section of the image. These are the image pixel
        co-ordinates of the area that contain overscan data as opposed to
        actual pixel data.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: list as default (i.e., format=None)
        :return: the overscan_section
        """
        pass
    
    def pixel_scale(self):
        """
        Return the pixel_scale value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the pixel scale (in arcsec per pixel) of the observation
        """
        pass
    
    def prism(self):
        """
        Return the prism value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned prism value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       prism value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the prism used for the observation
        """
        pass
    
    def program_id(self):
        """
        Return the program_id value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the Gemini program ID (e.g., GN-2011A-Q-123) of the 
                 observation
        """
        pass
    
    def pupil_mask(self):
        """
        Return the pupil_mask value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the pupil mask used for the observation
        """
        pass
    
    def qa_state(self):
        """
        Return the qa_state value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the quality assessment state (either 'Undefined', 'Pass', 
                 'Usable', 'Fail' or 'CHECK') of the observation
        """
        pass
    
    def ra(self):
        """
        Return the ra value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the Right Ascension (in decimal degrees) of the observation
        """
        pass
    
    def raw_bg(self):
        """
        Return the raw_bg value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw background (either '20-percentile', '50-percentile', 
                 '80-percentile' or 'Any') of the observation
        """
        pass
    
    def raw_cc(self):
        """
        Return the raw_cc value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw cloud cover (either '50-percentile', '70-percentile', 
                 '80-percentile', '90-percentile' or 'Any') of the observation
        """
        pass
    
    def raw_iq(self):
        """
        Return the raw_iq value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw image quality (either '20-percentile', 
                 '70-percentile', '85-percentile' or 'Any') of the observation
        """
        pass
    
    def raw_wv(self):
        """
        Return the raw_wv value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw water vapour (either '20-percentile', 
                 '50-percentile', '80-percentile' or 'Any') of the observation
        """
        pass
    
    def read_mode(self):
        """
        Return the read_mode value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the read mode (either 'Very Faint Object(s)', 
                 'Faint Object(s)', 'Medium Object', 'Bright Object(s)', 
                 'Very Bright Object(s)', 'Low Background', 
                 'Medium Background', 'High Background' or 'Invalid') of the 
                 observation
        """
        pass
    
    def read_noise(self):
        """
        Return the read_noise value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the estimated readout noise (in electrons) of the observation
        """
        pass
    
    def read_speed_setting(self):
        """
        Return the read_speed_setting value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the read speed setting (either 'fast' or 'slow') of the 
                 observation
        """
        pass
    
    def requested_iq(self):
        """
        Return the requested Image Quality value. This is the Gemini IQ
        percentile IQ band (eg "20-percentile") parsed to an integer 
        percentile value - (eg 20). "Any" maps to 100.
        Note, this is the value requested by the PI as the worst acceptable,
        not the delivered value. Smaller is better.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested_iq
        """
        pass
    
    def requested_cc(self):
        """
        Return the requested Cloud Cover value. This is the Gemini CC
        percentile CC band (eg "50-percentile") parsed to an integer 
        percentile value - (eg 50). "Any" maps to 100.
        Note, this is the value requested by the PI as the worst acceptable,
        not the delivered value. Smaller is better.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested_cc
        """
        pass
    
    def requested_wv(self):
        """
        Return the requested Water Vapor value. This is the Gemini WV
        percentile WV band (eg "50-percentile") parsed to an integer 
        percentile value - (eg 50). "Any" maps to 100.
        Note, this is the value requested by the PI as the worst acceptable,
        not the delivered value. Smaller is better.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested_wv
        """
        pass
    
    def requested_bg(self):
        """
        Return the requested sky Background value. This is the Gemini BG
        percentile BG band (eg "50-percentile") parsed to an integer 
        percentile value - (eg 50). "Any" maps to 100.
        Note, this is the value requested by the PI as the worst acceptable,
        not the delivered value. Smaller is better.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested_bg
        """
        pass
    
    def saturation_level(self):
        """
        Return the saturation_level value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the saturation level in the raw images (in ADU) of the 
                 observation
        """
        pass
    
    def slit(self):
        """
        Return the slit value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param stripID: set to True to remove the component ID from the 
                        returned slit value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       slit value
        :type pretty: Python boolean
        :rtype: string as default (i.e., format=None)
        :return: the slit used for the observation
        """
        pass
    
    def ut_datetime(self):
        """
        Return the ut_datetime value
        
        This descriptor attempts to figure out the datetime even when the
        headers are malformed or not present. It tries just about every header
        combination that could allow it to determine an appropriate datetime
        for the file in question. This makes it somewhat specific to Gemini
        data, in that the headers it looks at, and the assumptions it makes in
        trying to parse their values, are those known to occur in Gemini data.
        Note that some of the early gemini data, and that taken from lower
        level engineering interfaces, lack standard headers. Also the format
        and occurence of various headers has changed over time, even on the
        same instrument. If strict is set to True, the date or time are
        determined from valid FITS keywords. If it cannot be determined, None
        is returned. If dateonly or timeonly are set to True, then a
        datetime.date object or datetime.time object, respectively, is
        returned, containing only the date or time, respectively. These two
        interplay with strict in the sense that if strict is set to True and a
        date can be determined but not a time, then this function will return
        None unless the dateonly flag is set, in which case it will return the
        valid date. The dateonly and timeonly flags are intended for use by
        the ut_date and ut_time descriptors.
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :param strict: set to True to not try to guess the date or time
        :type strict: Python boolean
        :param dateonly: set to True to return a datetime.date
        :type dateonly: Python boolean
        :param timeonly: set to True to return a datetime.time
        :param timeonly: Python boolean
        :rtype: datetime.datetime (dateonly=False and timeonly=False)
        :rtype: datetime.time (timeonly=True)
        :rtype: datetime.date (dateonly=True)
        :return: the UT date and time at the start of the observation
        """
        pass
    
    def ut_time(self):
        """
        Return the ut_time value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the UT time at the start of the observation
        """
        pass
    
    def wavefront_sensor(self):
        """
        Return the wavefront_sensor value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the wavefront sensor (either 'AOWFS', 'OIWFS', 'PWFS1', 
                 'PWFS2', some combination in alphebetic order separated with 
                 an ampersand or None) used for the observation
        """
        pass
    
    def wavelength_band(self):
        """
        Return the wavelength band value. This only applies to spectroscopy
        data and gives the band (filter) name within which the central wavelength
        of the spectrum falls. 
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the wavelength_band
        """
        pass
    
    def wavelength_reference_pixel(self):
        """
        Return the wavelength_reference_pixel value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the reference pixel of the central wavelength of the 
                 observation
        """
        pass
    
    def well_depth_setting(self):
        """
        Return the well_depth_setting value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the well depth setting (either 'Shallow', 'Deep' or 
                 'Invalid') of the observation
        """
        pass
    
    def x_offset(self):
        """
        Return the x_offset value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the x offset of the observation
        """
        pass
    
    def y_offset(self):
        """
        Return the y_offset value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the y offset of the observation
        """
        pass
