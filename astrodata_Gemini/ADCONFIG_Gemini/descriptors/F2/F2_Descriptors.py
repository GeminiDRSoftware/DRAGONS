import math
from time import strptime
from datetime import datetime

from astrodata.utils import Errors
from astrodata.utils import Lookups

from astrodata.interface.Descriptors import DescriptorValue
from astrodata.interface.structuredslice import pixel_exts, bintable_exts

from gempy.gemini import gemini_metadata_utils as gmu

from F2_Keywords import F2_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

# ------------------------------------------------------------------------------
class F2_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = F2_KeyDict
    
    f2ArrayDict = None
    f2ConfigDict = None
    
    def __init__(self):
        self.f2ArrayDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ArrayDict", "f2ArrayDict")
        self.nifsConfigDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ConfigDict", "f2ConfigDict")
        GEMINI_DescriptorCalc.__init__(self)
    
    def data_section(self, dataset, pretty=False, **args):
        raw_data_section = "[1:2048,1:2048]"
        
        if pretty:
            # Use the data section string that uses 1-based indexing as the
            # value in the form [x1:x2,y1:y2] 
            ret_data_section = raw_data_section
        else:
            # Use the data section list that used 0-based, non-inclusive
            # indexing as the value in the form [x1, x2, y1, y2]
            ret_data_section = gmu.sectionStrToIntList(raw_data_section)
            
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_data_section, name="data_section",
                                 ad=dataset)
        return ret_dv
    
    array_section = data_section
    detector_section = data_section
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the UT date using the appropriate descriptor
        ut_date_dv = dataset.ut_date()
        
        if ut_date_dv.is_none():
            # The descriptor functions return None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ut_date = str(ut_date_dv)
        
        obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
        
        # Old commissioning data was taken before March 1, 2010
        old_ut_date = datetime(2010, 3, 1, 0, 0)
        
        if obs_ut_date > old_ut_date:
            # Determine the two filter name keywords from the global keyword
            # dictionary
            keyword1 = self.get_descriptor_key("key_filter1")
            keyword2 = self.get_descriptor_key("key_filter2")
            
        else:
            # Make sure the filter_name descriptor is backwards compatible with
            # old engineering data
            #
            # Determine the two filter name keywords from the global keyword
            # dictionary
            keyword1 = self.get_descriptor_key("key_old_filter1")
            keyword2 = self.get_descriptor_key("key_old_filter2")
            
        # Get the value of the two filter name keywords from the header of the
        # PHU 
        filter1 = dataset.phu_get_key_value(keyword1)
        filter2 = dataset.phu_get_key_value(keyword2)
        
        if None in [filter1, filter2]:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if stripID or pretty:
            # Strip the component ID from the two filter name values
            filter1 = gmu.removeComponentID(filter1)
            filter2 = gmu.removeComponentID(filter2)
        
        filter = []
        if pretty:
            # Remove any filters that have the value "open" or "Open"
            if "open" not in filter1 and "Open" not in filter1:
                filter.append(str(filter1))
            if "open" not in filter2 and "Open" not in filter2:
                filter.append(str(filter2))
            if len(filter) == 0:
                filter = ["open"]
            
            if "Block" in filter1 or "Block" in filter2:
                filter = ["blank"]
            if "Dark" in filter1 or "Dark" in filter2:
                filter = ["blank"]
            if "DK" in filter1 or "DK" in filter2:
                filter = ["dark"]
        else:
            filter = [filter1, filter2]
        
        if len(filter) > 1:
            # Concatenate the filter names with "&"
            ret_filter_name = "%s&%s" % (filter[0], filter[1])
        else:
            ret_filter_name = str(filter[0])
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_filter_name, name="filter_name",
                                 ad=dataset)
        return ret_dv
    
    def gain(self, dataset, **args):
        # Determine the number of non-destructive read pairs keyword (lnrs)
        # from the global keyword dictionary
        keyword = self.get_descriptor_key("key_lnrs")
        
        # Get the number of non-destructive read pairs from the header of the
        # PHU
        lnrs = dataset.phu_get_key_value(keyword)
        
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain, name="gain", ad=dataset)
        
        return ret_dv
    
    f2ArrayDict = None
    
    def instrument(self, dataset, **args):
        ret_instrument = "F2"
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_instrument, name="instrument", ad=dataset)
        
        return ret_dv
    
    def lyot_stop(self, dataset, stripID=False, pretty=False, **args):
        # Determine the lyot stop keywords from the global keyword dictionary
        keyword = self.get_descriptor_key("key_lyot_stop")
        
        # Get the value of the lyot stop keywords from the header of the PHU 
        lyot_stop = dataset.phu_get_key_value(keyword)
        
        if lyot_stop is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if stripID or pretty:
            # Strip the component ID from the two filter name values
            ret_lyot_stop = gmu.removeComponentID(lyot_stop)
        else:
            ret_lyot_stop = lyot_stop
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_lyot_stop, name="lyot_stop", ad=dataset)
        
        return ret_dv
    
    def non_linear_level(self, dataset, **args):
        # Determine the number of non-destructive read pairs keyword (lnrs)
        # from the global keyword dictionary
        keyword = self.get_descriptor_key("key_lnrs")
        
        # Get the number of non-destructive read pairs from the header of the
        # PHU
        lnrs = dataset.phu_get_key_value(keyword)
        
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the saturation level using the appropriate descriptor
        saturation_level_dv = dataset.saturation_level()
        
        if saturation_level_dv.is_none():
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
        ret_non_linear_level = int(saturation_level_dv * non_linear_fraction)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_non_linear_level, name="non_linear_level",
                                 ad=dataset)
        return ret_dv
    
    f2ArrayDict = None
    
    def observation_type(self, dataset, **args):
        # Determine the observation type keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_observation_type")
        
        # Get the value of the observation type from the header of the PHU
        observation_type = dataset.phu_get_key_value(keyword)
        
        if observation_type is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Sometimes FLAMINGOS-2 dark frames have the observation type
        # incorrectly set to OBJECT. Ensure that DARK is returned as the
        # observation type for dark frames.
        if "F2_DARK" in dataset.types:
            ret_observation_type = "DARK"
        else:
            ret_observation_type = observation_type
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_observation_type, name="observation_type",
                                 ad=dataset)
        return ret_dv
    
    def pixel_scale(self, dataset, **args):
        # First try to calculate the pixel scale using the values of the WCS
        # matrix elements keywords
        #
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_pixel_scale_dict = {}
        
        # Determine the WCS matrix elements keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_cd11")
        keyword2 = self.get_descriptor_key("key_cd12")
        keyword3 = self.get_descriptor_key("key_cd21")
        keyword4 = self.get_descriptor_key("key_cd22")
        
        # Get the value of the WCS matrix elements keywords from the header of
        # each pixel data extension as a dictionary where the key of the
        # dictionary is an ("*", EXTVER) tuple
        cd_dict = gmu.get_key_value_dict(
            adinput=dataset, keyword=[keyword1, keyword2, keyword3, keyword4])
        
        cd11_dict = cd_dict[keyword1]
        cd12_dict = cd_dict[keyword2]
        cd21_dict = cd_dict[keyword3]
        cd22_dict = cd_dict[keyword4]
        
        if cd11_dict is None:
            # Get the pixel scale value using the value of the pixel scale
            # keyword in the PHU
            pixel_scale = self._get_pixel_scale_from_header(dataset=dataset)
            
            # Loop over the pixel data extensions in the dataset
            pixel_scale_dict = {}
            for ext in dataset[pixel_exts]:
                # Update the dictionary with the pixel_scale value
                pixel_scale_dict.update(
                    {(ext.extname(), ext.extver()): pixel_scale})
            
            # Instantiate the DescriptorValue (DV) object
            dv = DescriptorValue(pixel_scale_dict)
            
            # Create a new dictionary where the key of the dictionary is an
            # EXTVER integer
            extver_dict = dv.collapse_by_extver()
            
            if not dv.validate_collapse_by_extver(extver_dict):
                # The validate_collapse_by_extver function returns False if the
                # values in the dictionary with the same EXTVER are not equal
                raise Errors.CollapseError()
            
            ret_pixel_scale_dict = pixel_scale_dict
        
        else:
            for ext_name_ver, cd11 in cd11_dict.iteritems():
                cd12 = None
                if cd12_dict is not None:
                    cd12 = cd12_dict[ext_name_ver]
                cd21 = None
                if cd21_dict is not None:
                    cd21 = cd21_dict[ext_name_ver]
                cd22 = None
                if cd22_dict is not None:
                    cd22 = cd22_dict[ext_name_ver]
                
                pixel_scale = None
                if not None in [cd11, cd12, cd21, cd22]:
                    # Calculate the pixel scale using the WCS matrix elements
                    pixel_scale = 3600 * (
                      math.sqrt(math.pow(cd11, 2) + math.pow(cd12, 2)) +
                      math.sqrt(math.pow(cd21, 2) + math.pow(cd22, 2))) / 2
                
                if pixel_scale is None or pixel_scale == 0.0:
                    # Get the pixel scale value using the value of the pixel
                    # scale keyword
                    pixel_scale = self._get_pixel_scale_from_header(
                        dataset=dataset)
                
                # Update the dictionary with the pixel scale value
                ret_pixel_scale_dict.update({ext_name_ver: pixel_scale})
        
        # Instantiate the return DescriptorValue (DV) object using the newly
        # created dictionary
        ret_dv = DescriptorValue(ret_pixel_scale_dict, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    def _get_pixel_scale_from_header(self, dataset):
        # Determine the pixel scale keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_pixel_scale")
        
        # Get the value of the pixel scale keyword from the header of the PHU
        ret_pixel_scale = dataset.phu_get_key_value(keyword)
        
        if ret_pixel_scale is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        return ret_pixel_scale
    
    def read_mode(self, dataset, **args):
        # Determine the number of non-destructive read pairs (lnrs) keyword
        # from the global keyword dictionary
        keyword = self.get_descriptor_key("key_lnrs")
        
        # Get the values of the number of non-destructive read pairs from the
        # header of the PHU
        lnrs = dataset.phu_get_key_value(keyword)
        
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the read mode integer
        ret_read_mode = int(lnrs)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_mode, name="read_mode", ad=dataset)
        
        return ret_dv
    
    def read_noise(self, dataset, **args):
        # Determine the number of non-destructive read pairs keyword (lnrs)
        # from the global keyword dictionary
        keyword = self.get_descriptor_key("key_lnrs")
        
        # Get the number of non-destructive read pairs from the header of the
        # PHU
        lnrs = dataset.phu_get_key_value(keyword)
        
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_noise, name="read_noise", ad=dataset)
        
        return ret_dv
    
    f2ArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Determine the number of non-destructive read pairs keyword (lnrs)
        # from the global keyword dictionary
        keyword = self.get_descriptor_key("key_lnrs")
        
        # Get the number of non-destructive read pairs from the header of the
        # PHU
        lnrs = dataset.phu_get_key_value(keyword)
        
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
        
        # Return the saturation_level integer
        ret_saturation_level = int(saturation_level)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_saturation_level, name="saturation_level",
                                 ad=dataset)
        return ret_dv
     
    f2ArrayDict = None
    
    def wavelength_band(self, dataset, **args):
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = Lookups.get_lookup_table(
                "Gemini/F2/F2FilterWavelength", "filter_wavelength")
            filter = str(dataset.filter_name(pretty=True))
            if filter in filter_table:
                ctrl_wave = filter_table[filter]
            else:
                raise Errors.TableKeyError()
        else:
            ctrl_wave = dataset.central_wavelength(asMicrometers=True)
        
        min_diff = None
        band = None
        
        for std_band, std_wave in self.std_wavelength_band.items():
            diff = abs(std_wave - ctrl_wave)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                band = std_band
        
        if band is None:
            raise Errors.CalcError()
        else:
            ret_wavelength_band = band
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_wavelength_band, name="wavelength_band",
                                 ad=dataset)
        return ret_dv
    
    def x_offset(self, dataset, **args):
        # Determine the y offset keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_y_offset")
        
        # Get the y offset from the header of the PHU
        y_offset = dataset.phu_get_key_value(keyword)
        
        if y_offset is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ret_x_offset = -y_offset
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_x_offset, name="x_offset", ad=dataset)
        
        return ret_dv
    
    def y_offset(self, dataset, **args):
        # Determine the x offset keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_x_offset")
        
        # Get the x offset from the header of the PHU.
        x_offset = dataset.phu_get_key_value(keyword)
        
        if x_offset is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ret_y_offset = -x_offset
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_y_offset, name="y_offset", ad=dataset)
        
        return ret_dv
