from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy.gemini import gemini_metadata_utils as gmu

from NICI_Keywords import NICI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class NICI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = NICI_KeyDict
     
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def exposure_time(self, dataset, **args):
        # Determine the two number of coadds and the two exposure time keywords
        # from the global keyword dictionary
        keyword1 = self.get_descriptor_key("key_exposure_time_r")
        keyword2 = self.get_descriptor_key("key_exposure_time_b")
        keyword3 = self.get_descriptor_key("key_coadds_r")
        keyword4 = self.get_descriptor_key("key_coadds_b")
        
        # Get the value of the two number of coadds and the two exposure time
        # keywords from the header of the PHU
        #
        # Note: 2010 data has the keywords in the PHU but earlier data (not
        # sure where the line is) doesn't. Need to find out when the keyword
        # locations changed ...
        exposure_time_r = dataset.phu_get_key_value(keyword1)
        exposure_time_b = dataset.phu_get_key_value(keyword2)
        coadds_r = dataset.phu_get_key_value(keyword3)
        coadds_b = dataset.phu_get_key_value(keyword4)
        
        if exposure_time_r is None or exposure_time_b is None or \
            coadds_r is None or coadds_b is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return a dictionary with the exposure time keyword names as the key
        # and the total exposure time float as the value
        ret_exposure_time = {}
        total_exposure_time_r = float(exposure_time_r * coadds_r)
        total_exposure_time_b = float(exposure_time_b * coadds_b)
        ret_exposure_time.update(
            {key_exposure_time_r:total_exposure_time_r,
             key_exposure_time_b:total_exposure_time_b})
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_filter_name = {}
        
        # For NICI, the red filter is defined in the first science extension,
        # while the blue filter is defined in the second science extension. Get
        # the two filter name values from the header of each pixel data
        # extension
        count = 0
        for ext in dataset:
            # Assigning a "SCI" extension doesn't work for NICI right now ...
            if count == 0:
                filter_r = ext.get_key_value(
                    self.get_descriptor_key("key_filter_r"))
                filter_b = None
                count += 1
            else:
                filter_b = ext.get_key_value(
                    self.get_descriptor_key("key_filter_b"))
        
        if (filter_r is None) or (filter_b is None):
            # The get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
            else:
                raise Errors.Error("No second science extension")
        
        if pretty:
            stripID = True
        if stripID:
            # Strip the component ID from the two filter name values
            filter_r = gmu.removeComponentID(filter_r)
            filter_b = gmu.removeComponentID(filter_b)
        
        # Return a dictionary with the dispersion axis integer as the value
        ret_filter_name.update(
            {("SCI", 1):str(filter_r), ("SCI", 2):str(filter_b)})
        
        if ret_filter_name == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_filter_name
    
    def pixel_scale(self, dataset, **args):
        # Return the pixel scale float
        ret_pixel_scale = float(0.018)
        
        return ret_pixel_scale
