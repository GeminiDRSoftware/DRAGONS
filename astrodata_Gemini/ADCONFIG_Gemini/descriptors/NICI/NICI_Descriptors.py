from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.Descriptors import DescriptorValue

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
        
        if None in [exposure_time_r, exposure_time_b, coadds_r, coadds_b]:
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_exposure_time, name="exposure_time",
                                 ad=dataset)
        return ret_dv
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_filter_name = {}
        
        # For NICI, the red filter is defined in the first science extension,
        # while the blue filter is defined in the second science extension.
        #
        # Determine the filter name keyword from the global keyword dictionary 
        keyword1 = self.get_descriptor_key("key_filter_r")
        keyword2 = self.get_descriptor_key("key_filter_b")
        
        # Get the value of the filter name keyword from the header of each
        # pixel data extension as a dictionary where the key of the dictionary
        # is an ("*", EXTVER) tuple
        filter_dict = gmu.get_key_value_dict(adinput=dataset,
                                             keyword=[keyword1, keyword2])
        
        filter_r_dict = filter_dict[keyword1]
        filter_b_dict = filter_dict[keyword2]
        
        # The following code contains duplication so that the case when a user
        # loops over extensions in an AstroData object and calls the
        # filter_name descriptor on each extension, i.e.,
        #
        #   for ext in ad:
        #       print ext.filter_name()
        #
        # is handled appropriately. There may be a better way to do this ...
        if filter_r_dict is None:
            if filter_b_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            else:
                for ext_name_ver, filter_b in filter_b_dict.iteritems():
                    if filter_b is None:
                        raw_filter = None
                    else:
                        raw_filter = filter_b
                    
                    if pretty:
                        stripID = True
                    if stripID:
                        # Strip the component ID from the filter name value
                        if raw_filter is not None:
                            filter = gmu.removeComponentID(raw_filter)
                    else:
                        filter = raw_filter
                    
                    # Update the dictionary with the filter name value
                    ret_filter_name.update({ext_name_ver:filter})
        else:
            for ext_name_ver, filter_r in filter_r_dict.iteritems():
                if filter_b_dict is not None:
                    filter_b = filter_b_dict[ext_name_ver]
                else:
                    filter_b = None
                
                if filter_r is None and filter_b is None:
                    raw_filter = None
                elif filter_r is None and filter_b is not None:
                    raw_filter = filter_b
                elif filter_r is not None and filter_b is None:
                    raw_filter = filter_r
                else:
                    # Both filter_r and filter_b are defined for a single
                    # extension, which is incorrect 
                    raise Errors.CorruptDataError()
                
                if pretty:
                    stripID = True
                if stripID:
                    # Strip the component ID from the filter name value
                    if raw_filter is not None:
                        filter = gmu.removeComponentID(raw_filter)
                else:
                    filter = raw_filter
                
                # Update the dictionary with the filter name value
                ret_filter_name.update({ext_name_ver:filter})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_filter_name, name="filter_name",
                                 ad=dataset)
        return ret_dv
    
    def pixel_scale(self, dataset, **args):
        # Return the pixel scale float
        ret_pixel_scale = float(0.018)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    def read_mode(self, dataset, **args):
        # For NICI data, raise an exception if the read_mode descriptor called,
        # since it is not relevant for NICI data.
        raise Errors.ExistError()
