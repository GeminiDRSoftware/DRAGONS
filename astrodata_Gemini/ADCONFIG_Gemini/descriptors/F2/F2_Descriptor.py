from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import string

from StandardF2KeyDict import stdkeyDictF2
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class F2_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = stdkeyDictF2
    
    f2ArrayDict = None
    f2ConfigDict = None
    
    def __init__(self):
        self.f2ArrayDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ArrayDict", "f2ArrayDict")
        self.nifsConfigDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ConfigDict", "f2ConfigDict")
        GEMINI_DescriptorCalc.__init__(self)
    
    def data_section(self, dataset, pretty=False, **args):
        data_section = "[1:2048,1:2048]"
        if pretty:
            # Return the data section string that uses 1-based indexing as the
            # value
            ret_data_section = data_section
        else:
            # Return the data section list that used 0-based, non-inclusive
            # indexing as the value
            ret_data_section = string.sectionStrToIntList(data_section)
        
        return ret_data_section
        
    def gain(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self._specifickey_dict)
        lnrs = dataset.phu_get_key_value(self._specifickey_dict["key_lnrs"])
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if lnrs in getattr(self, "f2ArrayDict"):
            gain = self.f2ArrayDict[lnrs][1]
        else:
            raise Errors.TableKeyError()
        # Return the gain float
        ret_gain = float(gain)
        
        return ret_gain
    
    f2ArrayDict = None

    def instrument(self, dataset, **args):
        ret_instrument = "F2"
        
        return ret_instrument
    
    def non_linear_level(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self._specifickey_dict)
        lnrs = dataset.phu_get_key_value(self._specifickey_dict["key_lnrs"])
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        # Get the saturation level using the appropriate descriptor
        saturation_level = dataset.saturation_level()
        if saturation_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if lnrs in getattr(self, "f2ArrayDict"):
            non_linear_fraction = self.f2ArrayDict[lnrs][3]
        else:
            raise Errors.TableKeyError()
        # Return the read noise float
        ret_non_linear_level = int(saturation_level * non_linear_fraction)
        
        return ret_non_linear_level
    
    f2ArrayDict = None
    
    def read_noise(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self._specifickey_dict)
        lnrs = dataset.phu_get_key_value(self._specifickey_dict["key_lnrs"])
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if lnrs in getattr(self, "f2ArrayDict"):
            read_noise = self.f2ArrayDict[lnrs][0]
        else:
            raise Errors.TableKeyError()
        # Return the read noise float
        ret_read_noise = float(read_noise)
        
        return ret_read_noise
    
    f2ArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self._specifickey_dict)
        lnrs = dataset.phu_get_key_value(self._specifickey_dict["key_lnrs"])
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if lnrs in getattr(self, "f2ArrayDict"):
            saturation_level = self.f2ArrayDict[lnrs][2]
        else:
            raise Errors.TableKeyError()
        # Return the read noise float
        ret_saturation_level = int(saturation_level)
        
        return ret_saturation_level
    
    f2ArrayDict = None
    
    def x_offset(self, dataset, **args):
        # Get the y offset from the header of the PHU.
        y_offset = dataset.phu_get_key_value(globalStdkeyDict["key_yoffset"])
        if y_offset is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        ret_x_offset = -y_offset
        
        return ret_x_offset
    
    def y_offset(self, dataset, **args):
        # Get the x offset from the header of the PHU.
        x_offset = dataset.phu_get_key_value(globalStdkeyDict["key_xoffset"])
        if x_offset is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        ret_y_offset = -x_offset
        
        return ret_y_offset
