class docstrings:

    def airmass(self):
        """
        Return the airmass value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the mean airmass of the observation
        """
        pass
    
    def ao_seeing(self):
        """
        Return the AO-estimated seeing
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the AO-estimated seeing of the observation in arcseconds       
        """
        pass
    
    def amp_read_area(self):
        """
        Return the amp_read_area value
        
        :param dataset: the dataset
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
        :return: the composite string containing the name of the array
                 amplifier and the readout area of the array used for the
                 observation 
        """
        pass
    
    def array_name(self):
        """
        Return the array_name value
        
        :param dataset: the dataset
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
        :return: the name of each array used for the observation
        """
        pass
    
    def array_section(self):
        """
        Return the array_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful array_section 
                       value in the form [x1:x2,y1:y2] that uses 1-based 
                       indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2]
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the unbinned section of the array that was used to observe the
                 data
        """
        pass
    
    def azimuth(self):
        """
        Return the azimuth value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the azimuth (in degrees between 0 and 360) of the observation
        """
        pass
    
    def camera(self):
        """
        Return the camera value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        :param format: the return format
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
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the unique identifying name (e.g., GN-2003A-C-2-52-003) of the
                 observation
        """
        pass
    
    def data_section(self):
        """
        Return the data_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful data_section 
                       value in the form [x1:x2,y1:y2] that uses 1-based 
                       indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2]
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the section of the pixel data extensions that contains the
                 data observed
        """
        pass
    
    def decker(self):
        """
        Return the decker value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned decker value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       decker value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the decker position used for the observation
        """
        pass
    
    def dec(self):
        """
        Return the dec value, defined for most Gemini instruments at the central pixel
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the declination (in decimal degrees) of the observation
        """
        pass
    
    def detector_name(self):
        """
        Return the detector_name value
        
        :param dataset: the dataset
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
        :return: the name of the detector used for the observation
        """
        pass
    
    def detector_roi_setting(self):
        """
        Return the detector_roi_setting value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the human-readable description of the detector Region Of
                 Interest (ROI) setting (either 'Full Frame', 'CCD2', 'Central
                 Spectrum', 'Central Stamp', 'Custom', 'Undefined' or 'Fixed'),
                 which corresponds to the name of the ROI in the OT
        """
        pass
    
    def detector_rois_requested(self):
        """
        Return the detector_rois_requested value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: list containing a list of integers (corresponding to unbinned
                pixels) that uses 1-bases indexing in the form [x1, x2, y1, y2]
                as default (i.e., format=None) 
        :return: the requested detector Region Of Interest (ROI)s of the
                 observation 
        """
        pass
    
    def detector_section(self):
        """
        Return the detector_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful 
                       detector_section value in the form [x1:x2,y1:y2] that 
                       uses 1-based indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2] 
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the unbinned section of the detector that was used to observe
                 the data
        """
        pass
    
    def detector_x_bin(self):
        """
        Return the detector_x_bin value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned disperser value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       disperser value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the disperser used for the observation
        """
        pass
    
    def dispersion_axis(self):
        """
        Return the dispersion_axis value
        
        :param dataset: the dataset
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
        :return: the dispersion axis (along rows, x = 1; along columns, y = 2;
                 along planes, z = 3) of the observation
        """
        pass
    
    def dispersion(self):
        """
        Return the dispersion value
        
        :param dataset: the dataset
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
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
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
    
    def elevation(self):
        """
        Return the elevation value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        :return: the total exposure time (in seconds) of the observation
        """
        pass
    
    def filter_name(self):
        """
        Return the filter_name value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned filter_name value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       filter_name value
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the unique filter name identifier string used for the 
                 observation; when multiple filters are used, the filter names
                 are concatenated with an ampersand
        """
        pass
    
    def focal_plane_mask(self):
        """
        Return the focal_plane_mask value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned focal_plane_mask value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       focal_plane_mask value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the focal plane mask used for the observation
        """
        pass
    
    def gain(self):
        """
        Return the gain value
        
        :param dataset: the dataset
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
    
    def gain_setting(self):
        """
        Return the gain_setting value
        
        :param dataset: the dataset
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
        :return: the gain setting (either 'high' or 'low') of the observation
        """
        pass
    
    def gcal_lamp(self):
        """
        Return the lamp from which GCAL is sending out light. This takes into
        account the fact that the IR lamp is behind a shutter.

        :param dataset: the dataset
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
        :return: the lamp from which gcal is sending out light
        """
        pass

    def grating(self):
        """
        Return the grating value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned grating value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       grating value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the grating used for the observation
        """
        pass
    
    def group_id(self):
        """
        Return the group_id value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the unique string that describes which stack a dataset belongs
                 to; it is based on the observation_id
        """
        pass
    
    def is_ao(self):
        """
        Return True if the observation uses AO
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: boolean as default (i.e., format=None)
        :return: True if the observation uses adaptive optics, False otherwise        
        """
        pass
    
    def local_time(self):
        """
        Return the local_time value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the local time (in HH:MM:SS.S) at the start of the observation
        """
        pass
    
    def lyot_stop(self):
        """
        Return the lyot_stop value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the lyot stop used for the observation
        """
        pass
    
    def mdf_row_id(self):
        """
        Return the mdf_row_id value
        
        :param dataset: the dataset
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
        :return: the corresponding reference row in the Mask Definition File
                 (MDF)
        """
        pass
    
    def nod_count(self):
        """
        Return the nod_count value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        Return the nominal_atmospheric_extinction value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None) 
        :return: the nominal atmospheric extinction (defined as coeff *
                 (airmass - 1.0), where coeff is the site and filter specific
                 nominal atmospheric extinction coefficient) of the observation
        """
        pass
    
    def nominal_photometric_zeropoint(self):
        """
        Return the nominal_photometric_zeropoint value
        
        :param dataset: the dataset
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
        :return: the nominal photometric zeropoint of the observation
        """
        pass
    
    def non_linear_level(self):
        """
        Return the non_linear_level value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the ID (e.g., GN-2011A-Q-123-45) of the observation; it is
                 used by group_id
        """
        pass
    
    def observation_type(self):
        """
        Return the observation_type value
        
        :param dataset: the dataset
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
        Return the overscan_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful
                       overscan_section value in the form [x1:x2,y1:y2] that
                       uses 1-based indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2] 
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the section of the pixel data extensions that contains the
                 overscan data
        """
        pass
    
    def pixel_scale(self):
        """
        Return the pixel_scale value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned prism value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       prism value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the prism used for the observation
        """
        pass
    
    def program_id(self):
        """
        Return the program_id value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned pupil mask value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful
                       pupil mask value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the pupil mask used for the observation
        """
        pass
    
    def qa_state(self):
        """
        Return the qa_state value
        
        :param dataset: the dataset
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
        Return the ra value, defined for most Gemini instruments at the central pixel
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw background (as an integer percentile value) of the
                 observation
        """
        pass
    
    def raw_cc(self):
        """
        Return the raw_cc value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw cloud cover (as an integer percentile value) of the
                 observation
        """
        pass
    
    def raw_iq(self):
        """
        Return the raw_iq value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw image quality (as an integer percentile value) of the
                 observation
        """
        pass
    
    def raw_wv(self):
        """
        Return the raw_wv value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw water vapour (as an integer percentile value) of the
                 observation
        """
        pass
    
    def read_mode(self):
        """
        Return the read_mode value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: GNIRS/NIFS: one of
                 'Very Faint Object(s)', 
                 'Faint Object(s)', 
                 'Medium Object(s)', 
                 'Bright Object(s)', 
                 'Very Bright Object(s)', 

                 NIRI: one of 
                 'Low Background', 
                 'Medium Background', 
                 'High Background',
                 'Invalid'

                 GMOS: one of 
                 'Normal',
                 'Bright',
                 'Acquisition',
                 'Engineering'

        """
        pass
    
    def read_noise(self):
        """
        Return the read_noise value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the read speed setting (either 'fast' or 'slow') of the 
                 observation
        """
        pass
    
    def requested_bg(self):
        """
        Return the requested_bg value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested background (as an integer percentile value) of 
                 the observation
        """
        pass
    
    def requested_cc(self):
        """
        Return the requested_cc value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested cloud cover (as an integer percentile value) of
                 the observation
        """
        pass
    
    def requested_iq(self):
        """
        Return the requested_iq value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested image quality (as an integer percentile value)
                 of the observation
        """
        pass
    
    def requested_wv(self):
        """
        Return the requested_wv value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested water vapour (as an integer percentile value) of
                 the observation
        """
        pass
    
    def saturation_level(self):
        """
        Return the saturation_level value
        
        :param dataset: the dataset
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
        :return: the saturation level (in ADU) of the observation
        """
        pass
    
    def slit(self):
        """
        Return the slit value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned slit value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       slit value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the name of the slit used for the observation
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
        and occurrence of various headers has changed over time, even on the
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param strict: set to True to not try to guess the date or time
        :type strict: Python boolean
        :param dateonly: set to True to return a datetime.date
        :type dateonly: Python boolean
        :param timeonly: set to True to return a datetime.time
        :param timeonly: Python boolean
        :param format: the return format
        :type format: string
        :rtype: datetime.datetime (dateonly=False and timeonly=False)
        :rtype: datetime.time (timeonly=True)
        :rtype: datetime.date (dateonly=True)
        :return: the UT date and time at the start of the observation
        """
        pass
    
    def ut_time(self):
        """
        Return the ut_time value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the wavefront sensor (either 'AOWFS', 'OIWFS', 'PWFS1', 
                 'PWFS2', some combination in alphabetic order separated with 
                 an ampersand or None) used for the observation
        """
        pass
    
    def wavelength_band(self):
        """
        Return the wavelength_band value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the wavelength band name (e.g., J, V, R, N) of the observation
        """
        pass
    
    def wavelength_reference_pixel(self):
        """
        Return the wavelength_reference_pixel value
        
        :param dataset: the dataset
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
        :return: the 1-based reference pixel of the central wavelength of the 
                 observation
        """
        pass
    
    def well_depth_setting(self):
        """
        Return the well_depth_setting value
        
        :param dataset: the dataset
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
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the telescope offset in x (in arcsec) of the observation
        """
        pass
    
    def y_offset(self):
        """
        Return the y_offset value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the telescope offset in y (in arcsec) of the observation
        """
        pass
