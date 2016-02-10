import math
from time import strptime
from datetime import datetime

from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.slices import pixel_exts
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_metadata_utils as gmu

from F2_Keywords import F2_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

import GemCalcUtil

import pywcs
from gempy.gemini.coordinate_utils import toicrs

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
        self.f2ConfigDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ConfigDict", "f2ConfigDict")
        GEMINI_DescriptorCalc.__init__(self)
    
    # This is a slightly modified copy of the GEMINI central_wavelength
    # descriptor. All we do here is fix the value with the K-long filter
    # which comes out as 0 in the header. See FRS #36101
    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):
        # For most Gemini data, the central wavelength is recorded in
        # micrometers
        input_units = "micrometers"

        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]

        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the central wavelength in the default units of meters.
            output_units = "meters"

        # Determine the central wavelength keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_central_wavelength")

        # Get the value of the central wavelength keyword from the header of
        # the PHU
        raw_central_wavelength = dataset.phu_get_key_value(keyword)

        if dataset.phu_get_key_value('FILTER1') == 'K-long_G0812':
            raw_central_wavelength = 2.2

        if raw_central_wavelength is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        else:
            central_wavelength = float(raw_central_wavelength)

        # Validate the central wavelength value
        if central_wavelength < 0.0:
            raise Errors.InvalidValueError()
        else:
            # Use the utilities function convert_units to convert the central
            # wavelength value from the input units to the output units
            ret_central_wavelength = GemCalcUtil.convert_units(
              input_units=input_units, input_value=central_wavelength,
              output_units=output_units)

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_central_wavelength,
                                 name="central_wavelength", ad=dataset)
        return ret_dv

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
    
    def group_id(self, dataset, **args):
        # essentially a copy of the NIRI group_id descriptor, 
        # adapted for F2.
        
        # Descriptors used for all image types
        unique_id_descriptor_list_all = ["read_mode", "detector_section"]
                                         

        # List to format descriptor calls using 'pretty=True' parameter
        call_pretty_version_list = ["filter_name", "disperser",
                                    "focal_plane_mask"]

        # Descriptors to be returned as an ordered list using descriptor
        # 'as_list' method.
        convert_to_list_list = ["detector_section"]

        # Other descriptors required for spectra 
        required_spectra_descriptors = ["disperser", "focal_plane_mask"]
        if "SPECT" in dataset.types:
            unique_id_descriptor_list_all.extend(required_spectra_descriptors)

        # Additional descriptors required for each frame type
        dark_id = ["exposure_time", "coadds"]
        flat_id = ["filter_name", "camera", "exposure_time", "observation_id"]
        flat_twilight_id = ["filter_name", "camera"]
        science_id = ["observation_id", "filter_name", "camera", "exposure_time"]
        ## !!! KL: added exposure_time to science_id for QAP.  The sky subtraction
        ## !!! seems unable to deal with difference exposure time circa Sep 2015.
        ## !!! The on-target dither sky-sub falls over completely.
        ## !!! Also, we do not have a fully tested scale by exposure routine.

        # This is used for imaging flats and twilights to distinguish between
        # the two type
        additional_item_to_include = None
        
        # Update the list of descriptors to be used depending on image type
        ## This requires updating to cover all spectral types
        ## Possible updates to the classification system will make this usable
        ## at the Gemini level
        data_types = dataset.types
        if "F2_DARK" in  data_types:
            id_descriptor_list = dark_id
        elif "F2_IMAGE_FLAT" in data_types:
            id_descriptor_list = flat_id
            additional_item_to_include = "F2_IMAGE_FLAT"
        elif "F2_IMAGE_TWILIGHT" in data_types:
            id_descriptor_list = flat_twilight_id
            additional_item_to_include = "F2_IMAGE_TWILIGHT"            
        else:
            id_descriptor_list = science_id

        # Add in all of the common descriptors required
        id_descriptor_list.extend(unique_id_descriptor_list_all)

        # Form the group_id
        descriptor_object_string_list = []
        for descriptor in id_descriptor_list:
            # Prepare the descriptor call
            if descriptor in call_pretty_version_list:
                end_parameter = "(pretty=True)"
            else:
                end_parameter = "()"
            descriptor_call = ''.join([descriptor, end_parameter])

            # Call the descriptor
            exec ("descriptor_object = dataset.{0}".format(descriptor_call))

            # Check for a returned descriptor value object with a None value
            if descriptor_object.is_none():
                # The descriptor functions return None if a value cannot be found
                # and stores the exception info. Re-raise the exception. It
                # will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info

            # In some cases require the information as a list
            if descriptor in convert_to_list_list:
                descriptor_object = descriptor_object.as_list()

            # Convert DV value to a string and store
            descriptor_object_string_list.append(str(descriptor_object))

        # Add in any none descriptor related information
        if additional_item_to_include is not None:
            descriptor_object_string_list.append(additional_item_to_include)
            
        # Create the final group_id string
        ret_group_id = '_'.join(descriptor_object_string_list)            

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_group_id, name="group_id", ad=dataset)
        
        return ret_dv

        
    
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

    def nominal_photometric_zeropoint(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_nominal_photometric_zeropoint = {}
        
        table = Lookups.get_lookup_table("Gemini/F2/Nominal_Zeropoints",
                                         "nominal_zeropoints")
        # Get the values of the gain, detector name and filter name using the
        # appropriate descriptors. Use as_pytype() to return the values as the
        # default python type rather than an object.
        gain = dataset.gain().as_pytype()
        camera = dataset.camera(pretty=True).as_pytype()
        filter_name = dataset.filter_name(pretty=True).as_pytype()
        
        if gain is None or camera is None or filter_name is None:
            # The descriptor functions return None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the value of the BUNIT keyword from the header of each pixel data
        # extension as a dictionary where the key of the dictionary is an
        # ("*", EXTVER) tuple 
        bunit_dict = gmu.get_key_value_dict(adinput=dataset, keyword="BUNIT")
        
        for ext_name_ver, bunit in bunit_dict.iteritems():
            # If bunit is "electron" or None, set the gain factor to 0.0 
            gain_factor = 0.0
            
            if bunit == "adu":
                gain_factor = 2.5 * math.log10(gain)
                
            nominal_zeropoint_key = (filter_name, camera)
            
            if nominal_zeropoint_key in table:
                nominal_photometric_zeropoint = (
                    table[nominal_zeropoint_key] - gain_factor)
            else:
                raise Errors.TableKeyError()
            
            # Update the dictionary with the nominal photometric zeropoint
            # value 
            ret_nominal_photometric_zeropoint.update({
                ext_name_ver:nominal_photometric_zeropoint})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nominal_photometric_zeropoint,
                                 name="nominal_photometric_zeropoint",
                                 ad=dataset)
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

        if lnrs in getattr(self, "f2ArrayDict"):
            non_linear_fraction = self.f2ArrayDict[lnrs][3]
        else:
            raise Errors.TableKeyError()
        
        # Get the saturation level using the appropriate descriptor
        saturation_level_dv = dataset.saturation_level()
        
        if saturation_level_dv.is_none():
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
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
        
        # Get the gain using the appropriate descriptor, to return in units
        # of ADU
        gain = dataset.gain()
        if gain is None:
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        # Return the saturation_level integer
        ret_saturation_level = int(saturation_level / gain)
        
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
    
    def wcs_ra(self, dataset, **args):
        ext = dataset['SCI', 1]
        wcs = pywcs.WCS(ext.header)
        # Pixel coordinate of cass rotator center from Andy Stephens from gacq
        (x, y) = (1034, 1054)
        # Data can be 3d - Assume 1 on the z axis
        if ext.get_key_value('NAXIS') == 3:
            result = wcs.wcs_pix2sky([[x,y,1]], 1)
        else:
            result = wcs.wcs_pix2sky([[x,y]], 1)
        ra = result[0][0]
        dec = result[0][1]

        # As of 2015-09-01 F2 is broken in non sidereal mode in that
        # The WCS purports to be in FK5 but seems to be actually in APPT
        # Once this is fixed, we will need a conditional on ut_datetime here
        if 'NON_SIDEREAL' in dataset.types:
            (ra, dec) = toicrs('APPT', ra, dec, ut_datetime=dataset.ut_datetime().as_pytype())

        return DescriptorValue(ra, name="wcs_ra", ad=dataset)


    def wcs_dec(self, dataset, **args):
        ext = dataset['SCI', 1]
        wcs = pywcs.WCS(ext.header)
        # Pixel coordinate of cass rotator center from Andy Stephens from gacq
        (x, y) = (1034, 1054)
        # Data is 3d. Assume 1 on the z axis
        result = wcs.wcs_pix2sky([[x,y,1]], 1)
        ra = result[0][0]
        dec = result[0][1]

        # As of 2015-09-01 F2 is broken in non sidereal mode in that
        # The WCS purports to be in FK5 but seems to be actually in APPT
        # Once this is fixed, we will need a conditional on ut_datetime here
        if 'NON_SIDEREAL' in dataset.types:
            (ra, dec) = toicrs('APPT', ra, dec, ut_datetime=dataset.ut_datetime().as_pytype())

        return DescriptorValue(dec, name="wcs_dec", ad=dataset)

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
