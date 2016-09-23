.. appendixC:

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
