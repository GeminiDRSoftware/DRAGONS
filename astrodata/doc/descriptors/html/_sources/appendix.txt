.. _Appendix_typewalk:

****************************************
A Complete List of Available Descriptors
****************************************

Descriptors are designed such that essential keyword values that describe a
particular concept can be accessed from the headers of a given dataset in a
consistent manner, regardless of which instrument was used to obtain the
data. This is particularly useful for Gemini data, since the majority of
keywords used to describe a particular concept at Gemini are not uniform
between the instruments.

::

  shell> typewalk -l
  Available Descriptors
          airmass
          amp_read_area
          azimuth
          camera
          cass_rotator_pa
          central_wavelength
          coadds
          data_label
          data_section
          dec
          decker
          detector_section
          detector_x_bin
          detector_y_bin
          disperser
          dispersion
          dispersion_axis
          elevation
          exposure_time
          filter_name
          focal_plane_mask
          gain
          gain_setting
          grating
          group_id
          instrument
          local_time
          mdf_row_id
          nod_count
          nod_pixels
          non_linear_level
          object
          observation_class
          observation_epoch
          observation_id
          observation_type
          overscan_section
          pixel_scale
          prism
          program_id
          pupil_mask
          qa_state
          ra
          raw_bg
          raw_cc
          raw_iq
          raw_wv
          read_mode
          read_noise
          read_speed_setting
          saturation_level
          slit
          telescope
          ut_date
          ut_datetime
          ut_time
          wavefront_sensor
          wavelength_reference_pixel
          well_depth_setting
          x_offset
          y_offset

.. _Appendix_mkCI:

****************************
``mkCalculatorInterface.py``
****************************

::

  from datetime import datetime
  from descriptorDescriptionDict import asDictArgDict
  from descriptorDescriptionDict import descriptorDescDict
  from descriptorDescriptionDict import detailedNameDict
  from descriptorDescriptionDict import stripIDArgDict

  class DescriptorDescriptor:
      name = None
      description = None
      pytype = None
      unit = None
      
      thunkfuncbuff = """
      def %(name)s(self, format=None, **args):
          \"\"\"
          %(description)s
          \"\"\"
          try:
              self._lazyloadCalculator()
              keydict = self.descriptor_calculator._specifickey_dict
              #print hasattr(self.descriptor_calculator, "%(name)s")
              if not hasattr(self.descriptor_calculator, "%(name)s"):
                  key = "key_"+"%(name)s"
                  #print "mkCI10:",key, repr(keydict)
                  #print "mkCI12:", key in keydict
                  if key in keydict.keys():
                      retval = self.phu_get_key_value(keydict[key])
                      if retval is None:
                          if hasattr(self, "exception_info"):
                              raise self.exception_info
                  else:
                      msg = "Unable to find an appropriate descriptor function "
                      msg += "or a default keyword for %(name)s"
                      raise KeyError(msg)
              else:
                  retval = self.descriptor_calculator.%(name)s(self, **args)
              
              %(pytypeimport)s
              ret = DescriptorValue( retval, 
                                     format = format, 
                                     name = "%(name)s",
                                     ad = self,
                                     pytype = %(pytype)s )
              return ret
          except:
              if not hasattr(self, "exception_info"):
                  setattr(self, "exception_info", sys.exc_info()[1])
              if (self.descriptor_calculator is None 
                  or self.descriptor_calculator.throwExceptions == True):
                  raise
              else:
                  #print "NONE BY EXCEPTION"
                  self.exception_info = sys.exc_info()[1]
                  return None
      """
      def __init__(self, name=None, pytype=None):
          self.name = name
          if pytype:
              self.pytype = pytype
              rtype = pytype.__name__
          try:
              desc = descriptorDescDict[name]
          except:
              if rtype == 'str':
                  rtype = 'string'
              if rtype == 'int':
                  rtype = 'integer'
              try:
                  dname = detailedNameDict[name]
              except:
                  dname = name
              try:
                  asDictArg = asDictArgDict[name]
              except:
                  asDictArg = 'no'
              try:
                  stripIDArg = stripIDArgDict[name]
              except:
                  stripIDArg = 'no'

              if stripIDArg == 'yes':
                  desc = 'Return the %(name)s value\n' % {'name':name} + \
                         '        :param dataset: the data set\n' + \
                         '        :type dataset: AstroData\n' + \
                         '        :param stripID: set to True to remove the ' + \
                         'component ID from the \n                        ' + \
                         'returned %(name)s value\n' % {'name':name} + \
                         '        :type stripID: Python boolean\n' + \
                         '        :param pretty: set to True to return a ' + \
                         'human meaningful \n' + \
                         '                       %(name)s ' % {'name':name} + \
                         'value\n' + \
                         '        :type pretty: Python boolean\n' + \
                         '        :rtype: %(rtype)s ' % {'rtype':rtype} + \
                         'as default (i.e., format=None)\n' + \
                         '        :return: the %(dname)s' \
                         % {'dname':dname}
              elif asDictArg == 'yes':
                  desc = 'Return the %(name)s value\n' % {'name':name} + \
                         '        :param dataset: the data set\n' + \
                         '        :type dataset: AstroData\n' + \
                         '        :param format: the return format\n' + \
                         '                       set to as_dict to return a ' + \
                         'dictionary, where the number ' + \
                         '\n                       of dictionary elements ' + \
                         'equals the number of pixel data ' + \
                         '\n                       extensions in the image. ' + \
                         'The key of the dictionary is ' + \
                         '\n                       an (EXTNAME, EXTVER) ' + \
                         'tuple, if available. Otherwise, ' + \
                         '\n                       the key is the integer ' + \
                         'index of the extension.\n' + \
                         '        :type format: string\n' + \
                         '        :rtype: %(rtype)s ' % {'rtype':rtype} + \
                         'as default (i.e., format=None)\n' + \
                         '        :rtype: dictionary containing one or more ' + \
                         '%(rtype)s(s) ' % {'rtype':rtype} + \
                         '(format=as_dict)\n' + \
                         '        :return: the %(dname)s' \
                         % {'dname':dname}

              else:
                  desc = 'Return the %(name)s value\n' % {'name':name} + \
                         '        :param dataset: the data set\n' + \
                         '        :type dataset: AstroData\n' + \
                         '        :param format: the return format\n' + \
                         '        :type format: string\n' + \
                         '        :rtype: %(rtype)s ' % {'rtype':rtype} + \
                         'as default (i.e., format=None)\n' + \
                         '        :return: the %(dname)s' \
                         % {'dname':dname}
                  
          self.description = desc
          
      def funcbody(self):
          if self.pytype:
              pytypestr = self.pytype.__name__
          else:
              pytypestr = "None"
          if pytypestr == "datetime":
              pti = "from datetime import datetime"
          else:
              pti = ""
          #print "mkC150:", pti
          ret = self.thunkfuncbuff % {'name':self.name,
                                      'pytypeimport': pti,
                                      'pytype': pytypestr,
                                      'description':self.description}
          return ret
          
  DD = DescriptorDescriptor
          
  descriptors =   [   DD("airmass", pytype=float),
                      DD("amp_read_area", pytype=str),
                      DD("azimuth", pytype=float),
                      DD("camera", pytype=str),
                      DD("cass_rotator_pa", pytype=float),
                      DD("central_wavelength", pytype=float),
                      DD("coadds", pytype=int),
                      DD("data_label", pytype=str),
                      DD("data_section", pytype=list),
                      DD("dec", pytype=float),
                      DD("decker", pytype=str),
                      DD("detector_section", pytype=list),
                      DD("detector_x_bin", pytype=int),
                      DD("detector_y_bin", pytype=int),
                      DD("disperser", pytype=str),
                      DD("dispersion", pytype=float),
                      DD("dispersion_axis", pytype=int),
                      DD("elevation", pytype=float),
                      DD("exposure_time", pytype=float),
                      DD("filter_name", pytype=str),
                      DD("focal_plane_mask", pytype=str),
                      DD("gain", pytype=float),
                      DD("grating", pytype=str),
                      DD("group_id", pytype=str),
                      DD("gain_setting", pytype=str),
                      DD("instrument", pytype=str),
                      DD("local_time", pytype=str),
                      DD("mdf_row_id", pytype=int),
                      DD("nod_count", pytype=int),
                      DD("nod_pixels", pytype=int),
                      DD("non_linear_level", pytype=int),
                      DD("object", pytype=str),
                      DD("observation_class", pytype=str),
                      DD("observation_epoch", pytype=str),
                      DD("observation_id", pytype=str),
                      DD("observation_type", pytype=str),
                      DD("overscan_section", pytype=list),
                      DD("pixel_scale", pytype=float),
                      DD("prism", pytype=str),
                      DD("program_id", pytype=str),
                      DD("pupil_mask", pytype=str),
                      DD("qa_state", pytype=str),
                      DD("ra", pytype=float),
                      DD("raw_bg", pytype=str),
                      DD("raw_cc", pytype=str),
                      DD("raw_iq", pytype=str),
                      DD("raw_wv", pytype=str),
                      DD("read_mode", pytype=str),
                      DD("read_noise", pytype=float),
                      DD("read_speed_setting", pytype=str),
                      DD("saturation_level", pytype=int),
                      DD("slit", pytype=str),
                      DD("telescope", pytype=str),
                      DD("ut_date", pytype=datetime),
                      DD("ut_datetime", pytype=datetime),
                      DD("ut_time", pytype=datetime),
                      DD("wavefront_sensor", pytype=str),
                      DD("wavelength_reference_pixel", pytype=float),
                      DD("well_depth_setting", pytype=str),
                      DD("x_offset", pytype=float),
                      DD("y_offset", pytype=float),
                  ]

  wholeout = """import sys
  from astrodata import Descriptors
  from astrodata.Descriptors import DescriptorValue
  from astrodata import Errors

  class CalculatorInterface:

      descriptor_calculator = None
  %(descriptors)s
  # UTILITY FUNCTIONS, above are descriptor thunks            
      def _lazyloadCalculator(self, **args):
          '''Function to put at top of all descriptor members
          to ensure the descriptor is loaded.  This way we avoid
          loading it if it is not needed.'''
          if self.descriptor_calculator is None:
              self.descriptor_calculator = Descriptors.get_calculator(self, **args)
  """
  out = ""

  for dd in descriptors:
      out += dd.funcbody()
      
  finalout = wholeout % {"descriptors": out}

  print finalout

.. _Appendix_CI:

***************************************************
An Example Function from ``CalculatorInterface.py``
***************************************************

The example function below is auto-generated by the
``mkCalculatorInterface.py`` file. The ``CalculatorInterface.py`` file should
never be edited directly.

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
            if not hasattr(self.descriptor_calculator, "airmass"):
                key = "key_"+"airmass"
                if key in keydict.keys():
                    retval = self.phu_get_key_value(keydict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info
                else:
                    msg = "Unable to find an appropriate descriptor function "
                    msg += "or a default keyword for airmass"
                    raise KeyError(msg)
            else:
                retval = self.descriptor_calculator.airmass(self, **args)
            
            
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "airmass",
                                   ad = self,
                                   pytype = float )
            return ret
        except:
            if (self.descriptor_calculator is None 
                or self.descriptor_calculator.throwExceptions == True):
                raise
            else:
                #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()[1]
                return None

.. _Appendix_SDKD:

********************************
``StandardDescriptorKeyDict.py``
********************************

::

  globalStdkeyDict = {
      "key_airmass":"AIRMASS",
      "key_amp_read_area":"AMPROA",
      "key_azimuth":"AZIMUTH",
      "key_camera":"CAMERA",
      "key_cass_rotator_pa":"CRPA",
      "key_central_wavelength":"CWAVE",
      "key_coadds":"COADDS",
      "key_data_label":"DATALAB",
      "key_data_section":"DATASEC",
      "key_dec":"DEC",
      "key_decker":"DECKER",
      "key_detector_section":"DETSEC",
      "key_detector_x_bin":"XCCDBIN",
      "key_detector_y_bin":"YCCDBIN",
      "key_disperser":"DISPERSR",
      "key_dispersion":"WDELTA",
      "key_dispersion_axis":"DISPAXIS",
      "key_elevation":"ELEVATIO",
      "key_exposure_time":"EXPTIME",
      "key_filter_name":"FILTNAME",
      "key_focal_plane_mask":"FPMASK",
      "key_gain":"GAIN",
      "key_gain_setting":"GAINSET",
      "key_grating":"GRATING",
      "key_group_id":"GROUPID",
      "key_instrument":"INSTRUME",
      "key_local_time":"LT",
      "key_mdf_row_id":"MDFROW",
      "key_nod_count":"NODCOUNT",
      "key_nod_pixels":"NODPIX",
      "key_non_linear_level":"NONLINEA",
      "key_object":"OBJECT",
      "key_observation_class":"OBSCLASS",
      "key_observation_epoch":"OBSEPOCH",
      "key_observation_id":"OBSID",
      "key_observation_type":"OBSTYPE",
      "key_overscan_section":"OVERSSEC",
      "key_pixel_scale":"PIXSCALE",
      "key_prism":"PRISM",
      "key_program_id":"GEMPRGID",
      "key_pupil_mask":"PUPILMSK",
      "key_qa_state":"QASTATE",
      "key_ra":"RA",
      "key_raw_bg":"RAWBG",
      "key_raw_cc":"RAWCC",
      "key_raw_iq": "RAWIQ",
      "key_raw_wv":"RAWWV",
      "key_read_mode":"READMODE",
      "key_read_noise":"RDNOISE",
      "key_read_speed_setting":"RDSPDSET",
      "key_saturation_level":"SATLEVEL",
      "key_slit":"SLIT",
      "key_telescope":"TELESCOP",
      "key_ut_date":"DATE-OBS",
      "key_ut_datetime":"DATETIME",
      "key_ut_time":"UT",
      "key_wavefront_sensor":"WFS",
      "key_wavelength_reference_pixel":"WREFPIX",
      "key_well_depth_setting":"WELDEPTH",
      "key_x_offset":"XOFFSET",
      "key_y_offset":"YOFFSET",
      }

.. _Appendix_descriptor:

**********************************************************
An Example Descriptor Function from ``GMOS_Descriptor.py``
**********************************************************

::

  from astrodata import Errors
  from StandardGMOSKeyDict import stdkeyDictGMOS
  from GEMINI_Descriptor import GEMINI_DescriptorCalc
  
  class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
      # Updating the global key dictionary with the local key dictionary
      # associated with this descriptor class
      _update_stdkey_dict = stdkeyDictGMOS

      def __init__(self):
          GEMINI_DescriptorCalc.__init__(self)
        
      def detector_x_bin(self, dataset, **args):

          # Since this descriptor function accesses keywords in the headers of
          # the pixel data extensions, always return a dictionary where the key
          # of the dictionary is an (EXTNAME, EXTVER) tuple.
          ret_detector_x_bin = {}

          # Loop over the science extensions in the dataset
          for ext in dataset["SCI"]:

              # Get the ccdsum value from the header of each pixel data
              # extension. The ccdsum keyword is defined in the local key
              # dictionary (stdkeyDictGMOS) but is read from the updated global
              # key dictionary (self.get_descriptor_key())
              ccdsum = ext.get_key_value(self.get_descriptor_key("key_ccdsum"))

              if ccdsum is None:
                  # The get_key_value() function returns None if a value cannot
                  # be found and stores the exception info. Re-raise the
                  # exception. It will be dealt with by the CalculatorInterface
                  if hasattr(ext, "exception_info"):
                      raise ext.exception_info

              detector_x_bin, detector_y_bin = ccdsum.split()

              # Return a dictionary with the binning of the x-axis integer as 
              # the value
              ret_detector_x_bin.update({
                  (ext.extname(), ext.extver()):int(detector_x_bin)})

          if ret_detector_x_bin == {}:
              # If the dictionary is still empty, the AstroData object was not
              # autmatically assigned a "SCI" extension and so the above for 
              # loop was not entered
              raise Errors.CorruptDataError()
      
          return ret_detector_x_bin
