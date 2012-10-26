import re

from astrodata import Descriptors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy.library import astrotools

from PHOENIX_Keywords import PHOENIX_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class PHOENIX_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = PHOENIX_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def dec(self, dataset, **args):
        # Determine the declination keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_dec")
        
        # Get the value of the declination keyword from the header of the PHU
        dec = dataset.phu_get_key_value(keyword)
        
        if dec is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the declination float
        ret_dec = float(astrotools.degsextodec(dec))
        
        return ret_dec
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Determine the filter name keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_filter")
        
        # Get the value of the filter name keyword from the header of the PHU
        filter_name = dataset.phu_get_key_value(keyword)
        
        if filter_name is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if pretty:
            stripID = True
        if stripID:
            # Return the stripped filter name string
            ret_filter_name = string.removeComponentID(filter_name)
        else:
            # Return the filter name string
            ret_filter_name = str(filter_name)
        
        return ret_filter_name
    
    def ra(self, dataset, **args):
        # Determine the R.A. keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_ra")
        
        # Get the value of the R.A keyword from the header of the PHU
        ra = dataset.phu_get_key_value(keyword)
        
        if ra is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the declination float
        ret_ra = float(astrotools.degsextodec(ra))
        
        return ret_ra
