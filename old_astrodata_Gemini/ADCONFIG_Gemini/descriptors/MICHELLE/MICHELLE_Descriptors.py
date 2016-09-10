from astrodata.utils import Errors
from astrodata.interface.Descriptors import DescriptorValue

from MICHELLE_Keywords import MICHELLE_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class MICHELLE_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = MICHELLE_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def exposure_time(self, dataset, **args):
        # Determine the exposure time and the number of extensions keywords
        # from the global keyword dictionary
        keyword1 = self.get_descriptor_key("key_exposure_time")
        keyword2 = self.get_descriptor_key("key_numext")
        
        # Get the value of the exposure time and the number of extensions
        # keywords from the header of the PHU
        exposure_time = dataset.phu_get_key_value(keyword1)
        extensions = dataset.phu_get_key_value(keyword2)
        
        if exposure_time is None or extensions is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
        
        # Return the exposure time float
        ret_exposure_time = float(exposure_time * coadds * extensions)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_exposure_time, name="exposure_time",
                                 ad=dataset)
        return ret_dv
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Determine the filter name keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_filter_name")
        
        # Get the value of the filter name keyword from the header of the PHU
        filter_name = dataset.phu_get_key_value(keyword)
        
        if filter_name is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # The MICHELLE filters don't have ID strings, so we just ignore the
        # stripID and pretty options
        if filter_name == "NBlock":
            ret_filter_name = "blank"
        else:
            # Return the filter name string
            ret_filter_name = str(filter_name)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_filter_name, name="filter_name",
                                 ad=dataset)
        return ret_dv
