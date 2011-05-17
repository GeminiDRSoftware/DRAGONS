from astrodata import Descriptors
from astrodata import Errors
from astrodata.Calculator import Calculator

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardMICHELLEKeyDict import stdkeyDictMICHELLE
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class MICHELLE_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    globalStdkeyDict.update(stdkeyDictMICHELLE)
    
    def __init__(self):
        pass
    
    def exposure_time(self, dataset, **args):
        # Get the exposure time and the number of extensions from the header of
        # the PHU. The exposure time and the number of extensions keywords are
        # defined in the local key dictionary (stdkeyDictMICHELLE) but are read
        # from the updated global key dictionary (globalStdkeyDict)
        exposure_time = \
            dataset.phu_get_key_value(globalStdkeyDict['key_exposure_time'])
        extensions = dataset.phu_get_key_value(globalStdkeyDict['key_numext'])
        if exposure_time is None or extensions is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
        # Return the exposure time float
        ret_exposure_time = float(exposure_time * coadds * extensions)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the filter name value from the header of the PHU. The filter name
        # keyword is defined in the local key dictionary (stdkeyDictMICHELLE)
        # but is read from the updated global key dictionary (globalStdkeyDict)
        filter_name = \
            dataset.phu_get_key_value(globalStdkeyDict['key_filter_name'])
        if filter_name is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # The MICHELLE filters don't have ID strings, so we just ignore the
        # stripID and pretty options
        if filter_name == 'NBlock':
            ret_filter_name = 'blank'
        else:
            # Return the filter name string
            ret_filter_name = str(filter_name)
        
        return ret_filter_name
    
    def read_mode(self, dataset, **args):
        # Get the read mode from the header of the PHU. The read mode is
        # defined in the local key dictionary (stdkeyDictMICHELLE) but is read
        # from the updated global key dictionary (globalStdkeyDict)
        read_mode = dataset.phu_get_key_value(globalStdkeyDict['key_read_mode'])
        if read_mode is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Return the read mode string
        ret_read_mode = str(read_mode)
        
        return ret_read_mode
