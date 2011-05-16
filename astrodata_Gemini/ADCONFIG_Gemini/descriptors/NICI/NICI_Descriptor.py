from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import string

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardNICIKeyDict import stdkeyDictNICI
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NICI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    globalStdkeyDict.update(stdkeyDictNICI)
    
    def __init__(self):
        pass
        
    def exposure_time(self, dataset, **args):
        # Get the two number of coadds and the two exposure time values from
        # the header of the PHU. The two number of coadds and the two exposure
        # time keywords are defined in the local key dictionary
        # (stdkeyDictNICI) but are read from the updated global key dictionary
        # (globalStdkeyDict)
        # Note: 2010 data has the keywords in the PHU but earlier data (not
        # sure where the line is) doesn't. Need to find out when the keyword
        # locations changed ...
        key_exposure_time_r = globalStdkeyDict['key_exposure_time_r']
        exposure_time_r = dataset.phu_get_key_value(key_exposure_time_r)
        key_exposure_time_b = globalStdkeyDict['key_exposure_time_b']
        exposure_time_b = dataset.phu_get_key_value(key_exposure_time_b)
        coadds_r = dataset.phu_get_key_value(globalStdkeyDict['key_coadds_r'])
        coadds_b = dataset.phu_get_key_value(globalStdkeyDict['key_coadds_b'])
        if exposure_time_r is None or exposure_time_b is None or \
            coadds_r is None or coadds_b is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Return a dictionary with the exposure time keyword names as the key
        # and the total exposure time float as the value
        ret_exposure_time = {}
        total_exposure_time_r = float(exposure_time_r * coadds_r)
        total_exposure_time_b = float(exposure_time_b * coadds_b)
        ret_exposure_time.update({key_exposure_time_r:total_exposure_time_r, \
            key_exposure_time_b:total_exposure_time_b})
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the two filter name values from the header of the PHU. The two
        # filter name keywords are defined in the local key dictionary
        # (stdkeyDictNICI) but are read from the updated global key dictionary
        # (globalStdkeyDict)
        key_filter_r = globalStdkeyDict['key_filter_r']
        key_filter_b = globalStdkeyDict['key_filter_b']
        filter_r = dataset.phu_get_key_value(key_filter_r)
        filter_b = dataset.phu_get_key_value(key_filter_b)
        if filter_r is None or filter_b is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            stripID = True
        if stripID:
            # Strip the component _id from the two filter name values
            filter_r = string.removeComponentID(filter_r)
            filter_b = string.removeComponentID(filter_b)
        # Return a dictionary with the keyword names as the key and the filter
        # name string as the value
        ret_filter_name = {}
        ret_filter_name.update({key_filter_r:str(filter_r), \
            key_filter_b:str(filter_b)})
        
        return ret_filter_name
    
    def pixel_scale(self, dataset, **args):
        # Return the pixel scale float
        ret_pixel_scale = float(0.018)
        
        return ret_pixel_scale
