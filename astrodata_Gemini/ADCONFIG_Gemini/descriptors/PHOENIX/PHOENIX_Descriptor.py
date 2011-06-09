import re

from astrodata import Descriptors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import astrotools

from StandardPHOENIXKeyDict import stdkeyDictPHOENIX
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class PHOENIX_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = stdkeyDictPHOENIX
    
    def dec(self, dataset, **args):
        # Get the declination from the header of the PHU
        dec = dataset.phu_get_key_value(self.get_descriptor_key("key_dec"))
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
        # Get the filter name value from the header of the PHU. The filter name
        # keyword is defined in the local key dictionary (stdkeyDictPHOENIX)
        # but are read from the updated global key dictionary
        # (self.get_descriptor_key())
        filter_name = dataset.phu_get_key_value(
            self.get_descriptor_key("key_filter"))
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
        # Get the declination from the header of the PHU
        ra = dataset.phu_get_key_value(self.get_descriptor_key("key_ra"))
        if ra is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        # Return the declination float
        ret_ra = float(astrotools.degsextodec(ra))
        
        return ret_ra
