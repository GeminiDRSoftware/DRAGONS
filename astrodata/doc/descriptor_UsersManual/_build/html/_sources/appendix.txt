.. _Appendix_typewalk:

****************************************
A Complete List of Available Descriptors
****************************************

Descriptors are designed such that essential keyword values that describe a
particular concept can be accessed from the headers of any dataset in a
consistent manner, regardless of which instrument was used to obtain the
data. This is particularly useful for Gemini data, since the majority of
keywords used to describe a particular concept at Gemini are not uniform
between the instruments.

``airmass``

- the mean airmass of the observation

``amp_read_area``

- the composite string containing the name of the array amplifier and the
  readout area of the array used for the observation

``array_section``

- the unbinned section (in the form of a Python list of integers that uses
  0-based indexing as default) of the array that was used to observe the data

``azimuth``

- the azimuth (in degrees between 0 and 360) of the observation

``bias_level``

- the bias level (in ADU) of the observation

``camera``

- the camera used for the observation

``cass_rotator_pa``

- the cassegrain rotator position angle (in degrees between -360 and 360) of
  the observation 

``central_wavelength``

- the central wavelength (in meters as default) of the observation

``coadds``

- the number of coadds used for the observation

``data_label``

- the unique identifying name (e.g., GN-2003A-C-2-52-003) of the observation

``data_section``

- the section (in the form of a Python list of integers that uses 0-based
  indexing as default) of the pixel data extensions that contains the data
  observed

``decker``

- the decker position used for the observation

``dec``

- the declination (in decimal degrees) of the observation

``detector_name``

- the name of each array used for the observation

``detector_roi_setting``

- the human-readable description of the detector Region Of Interest (ROI)
  setting (either 'Full Frame', 'CCD2', 'Central Spectrum', 'Central Stamp',
  'Custom', 'Undefined' or 'Fixed'), which corresponds to the name of the ROI
  in the OT 

``detector_rois_requested``

- the requested detector Region Of Interest (ROI)s of the observation

``detector_section``

- the unbinned section (in the form of a Python list of integers that uses
  0-based indexing as default) of the detector that was used to observe the
  data 

``detector_x_bin``

- the binning of the x-axis of the detector used for the observation

``detector_y_bin``

- the binning of the y-axis of the detector used for the observation

``disperser``

- the disperser used for the observation - difference between
  disperser/grating/prism? when should someone use disperser? ***********************************************

``dispersion_axis``

- the dispersion axis (along rows, x = 1; along columns, y = 2; along planes,
  z = 3) of the observation.

``dispersion``

- the dispersion (in meters per pixel as default) of the observation

``elevation``

- the elevation (in degrees) of the observation

``exposure_time``

- the total exposure time (in seconds) of the observation

``filter_name``

- the unique filter name identifier string used for the observation; when
  multiple filters are used, the filter names are concatenated with an
  ampersand

``focal_plane_mask``

- the focal plane mask used for the observation

``gain``

- the gain (in electrons per ADU) of the observation

``gain_setting``

- the gain setting (either 'high' or 'low') of the observation

``grating``

- the grating used for the observation

``group_id``

- the unique string that describes which stack a dataset belongs to; it is
  based on the observation_id 

``instrument``

- the instrument used for the observation

``local_time``

- the local time (in HH:MM:SS.S) at the start of the observation

``mdf_row_id``

- the corresponding reference row in the Mask Definition File (MDF)

``nod_count``

- the number of nod and shuffle cycles in the nod and shuffle observation

``nod_pixels``

- the number of pixel rows the charge is shuffled by in the nod and shuffle
  observation 

``nominal_atmospheric_extinction``

- the nominal atmospheric extinction (defined as coeff * (airmass - 1.0), where
  coeff is the site and filter specific nominal atmospheric extinction
  coefficient) of the observation 

``nominal_photometric_zeropoint``

- the nominal photometric zeropoint of the observation

``non_linear_level``

- the non linear level in the raw images (in ADU) of the observation

``object``

- the name of the target object observed

``observation_class``

- the class (either 'science', 'progCal', 'partnerCal', 'acq', 'acqCal' or
  'dayCal') of the observation 

``observation_epoch``

- the epoch (in years) at the start of the observation

``observation_id``

- the ID (e.g., GN-2011A-Q-123-45) of the observation; it is used by group_id

``observation_type``

- the type (either 'OBJECT', 'DARK', 'FLAT', 'ARC', 'BIAS' or 'MASK') of the
  observation 

``overscan_section``

- the section (in the form of a Python list of integers that uses 0-based
  indexing as default) of the pixel data extensions that contains the overscan
  data

``pixel_scale``

- the pixel scale (in arcsec per pixel) of the observation

``prism``

- the prism used for the observation

``program_id``

- the Gemini program ID (e.g., GN-2011A-Q-123) of the observation

``pupil_mask``

- the pupil mask used for the observation

``qa_state``

- the quality assessment state (either 'Undefined', 'Pass', 'Usable', 'Fail' or
  'CHECK') of the observation 

``ra``

- the Right Ascension (in decimal degrees) of the observation

``raw_bg``

- the raw background (as an integer percentile value) of the observation 

``raw_cc``

- the raw cloud cover (as an integer percentile value) of the observation 

``raw_iq``

- the raw image quality (as an integer percentile value) of the observation 

``raw_wv``

- the raw water vapour (as an integer percentile value) of the observation 

``read_mode``

- the read mode (either 'Very Faint Object(s)', 'Faint Object(s)', 'Medium
  Object', 'Bright Object(s)', 'Very Bright Object(s)', 'Low Background',
  'Medium Background', 'High Background' or 'Invalid') of the observation

``read_noise``

- the estimated readout noise (in electrons) of the observation

``read_speed_setting``

- the read speed setting (either 'fast' or 'slow') of the observation

``requested_bg``

- the requested background (as an integer percentile value) of the observation

``requested_cc``

- the requested cloud cover (as an integer percentile value) of the observation

``requested_iq``

- the requested image quality (as an integer percentile value) of the
  observation

``requested_wv``

- the requested water vapour (as an integer percentile value) of the
  observation

``saturation_level``

- the saturation level (in ADU) of the observation

``slit``

- the name of the slit used for the observation

``telescope``

- the telescope used for the observation

``ut_date``

- the UT date (as a datetime object) at the start of the observation

``ut_datetime``

- the UT date and time (as a datetime object) at the start of the observation

``ut_time``

- the UT time (as a datetime object) at the start of the observation

``wavefront_sensor``

- the wavefront sensor (either 'AOWFS', 'OIWFS', 'PWFS1', 'PWFS2', some
  combination in alphabetic order separated with an ampersand or None) used for
  the observation 

``wavelength_band``

- the wavelength band name (e.g., J, V, R, N) of the observation

``wavelength_reference_pixel``

- the 1-based reference pixel of the central wavelength of the observation

``well_depth_setting``

- the well depth setting (either 'Shallow', 'Deep' or 'Invalid') of the
  observation 

``x_offset``

- the telescope offset in x (in arcsec) of the observation

``y_offset``

- the telescope offset in y (in arcsec) of the observation

.. _Appendix_CI:

**********************************************************
An Example Function from ``CalculatorInterface_GEMINI.py``
**********************************************************

The example function below is auto-generated by the
``mkCalculatorInterface`` script. The ``CalculatorInterface_GEMINI.py`` file
should never be edited directly.

::

    def airmass(self, format=None, **args):
        """
        Return the airmass value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the mean airmass of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_airmass"
            keyword = None
            if key in keydict.keys():
                keyword = keydict[key]
                
            if not hasattr(self.descriptor_calculator, "airmass"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for airmass")
                    raise Errors.DescriptorError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.airmass(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)
            
            
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "airmass",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret
        
        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, \"exception_info\"):
                    setattr(self, \"exception_info\", sys.exc_info()[1])
                return None
        except:
            raise

.. _Appendix_descriptor:

**********************************************************
An Example Descriptor Function from ``GMOS_Descriptor.py``
**********************************************************

::

  from astrodata import Errors
  from GMOS_Keywords import GMOS_KeyDict
  from GEMINI_Descriptors import GEMINI_DescriptorCalc
  
  class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
      # Updating the global key dictionary with the local key dictionary
      # associated with this descriptor class
      _update_stdkey_dict = GMOS_KeyDict
      
      def __init__(self):
          GEMINI_DescriptorCalc.__init__(self)
      
      def detector_x_bin(self, dataset, **args):
          # Since this descriptor function accesses keywords in the headers of
          # the pixel data extensions, always return a dictionary where the key
          # of the dictionary is an (EXTNAME, EXTVER) tuple
          ret_detector_x_bin = {}
          
          # Determine the ccdsum keyword from the global keyword dictionary 
          keyword = self.get_descriptor_key("key_ccdsum")
          
          # Get the value of the ccdsum keyword from the header of each pixel
          # data extension as a dictionary 
          ccdsum_dict = gmu.get_key_value_dict(dataset, keyword)
          
          if ccdsum_dict is None:
              # The get_key_value_dict() function returns None if a value
              # cannot be found and stores the exception info. Re-raise the
              # exception. It will be dealt with by the CalculatorInterface.
              if hasattr(dataset, "exception_info"):
                  raise dataset.exception_info
          
          for ext_name_ver, ccdsum in ccdsum_dict.iteritems():
              if ccdsum is None:
                  detector_x_bin = None
              else:
                  # Use the binning of the x-axis integer as the value
                  detector_x_bin, detector_y_bin = ccdsum.split()
              
              # Update the dictionary with the binning of the x-axis value
              ret_detector_x_bin.update({ext_name_ver:detector_x_bin})
          
          return ret_detector_x_bin
