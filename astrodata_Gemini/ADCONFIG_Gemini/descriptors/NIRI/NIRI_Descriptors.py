import math

from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_metadata_utils as gmu

import GemCalcUtil 
from NIRI_Keywords import NIRI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

from astrodata_Gemini.ADCONFIG_Gemini.lookups.NIRI import NIRISpecDict
from astrodata_Gemini.ADCONFIG_Gemini.lookups.NIRI import NIRIFilterMap
from astrodata_Gemini.ADCONFIG_Gemini.lookups.NIRI import Nominal_Zeropoints
from astrodata_Gemini.ADCONFIG_Gemini.lookups.NIRI import NIRIFilterWavelength

# replaces the former lookup of the FITS bintable, nsappwavepp.fits
from astrodata_Gemini.ADCONFIG_Gemini.lookups.IR import nsappwavepp
# ------------------------------------------------------------------------------
class NIRI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = NIRI_KeyDict
    
    niriFilternameMapConfig = None
    niriFilternameMap = {}
    niriSpecDict = None
    
    def __init__(self):
        self.niriSpecDict = NIRISpecDict.niriSpecDict
        self.niriFilternameMapConfig = NIRIFilterMap.niriFilternameMapConfig
        self.nsappwave = nsappwavepp.nsappwavepp
        
        filternamemap = {}
        for line in self.niriFilternameMapConfig:
            linefiltername = gmu.filternameFrom([line[1], line[2], line[3]])
            filternamemap.update({linefiltername:line[0]})
        self.niriFilternameMap = filternamemap
        
        GEMINI_DescriptorCalc.__init__(self)
    
    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):
        # Currently for NIRI data, the central wavelength is recorded in
        # angstroms
        input_units = "angstroms"
        
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
            # return the central wavelength in the default units of meters
            output_units = "meters"
        
        # The central_wavelength from nsappwave can only be obtained from data
        # that does not have an AstroData Type of IMAGE
        if "IMAGE" not in dataset.types:
            
            # Get the focal plane mask and disperser values using the
            # appropriate descriptors
            focal_plane_mask = dataset.focal_plane_mask()
            disperser = dataset.disperser(stripID=True)
            
            if focal_plane_mask is None or disperser is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info

            # Get the central wavelength value from the nsappwave lookup
            # table
            count = 0
            for row in self.nsappwave:
                if (focal_plane_mask == row["MASK"] and
                    disperser == row["GRATING"]):
                    count += 1
                    if row["LAMBDA"]:
                        raw_central_wavelength = float(row["LAMBDA"])
                    else:
                        raise Errors.TableValueError()
            if count == 0:
                raise Errors.TableKeyError()
            
            # Use the utilities function convert_units to convert the central
            # wavelength value from the input units to the output units
            ret_central_wavelength = GemCalcUtil.convert_units(
                input_units=input_units,
                input_value=float(raw_central_wavelength),
                output_units=output_units)
        else:
            raise Errors.DescriptorTypeError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_central_wavelength,
                                 name="central_wavelength", ad=dataset)
        return ret_dv
    
    def data_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_data_section = {}
        
        # Determine the region of interest keyword from the global keyword
        # dictionary 
        keyword1 = self.get_descriptor_key("key_lowrow")
        keyword2 = self.get_descriptor_key("key_hirow")
        keyword3 = self.get_descriptor_key("key_lowcol")
        keyword4 = self.get_descriptor_key("key_hicol")
        
        # Get the value of the region of interest keyword from the header of
        # each pixel data extension as a dictionary where the key of the
        # dictionary is an ("*", EXTVER) tuple
        coord_dict = gmu.get_key_value_dict(
            adinput=dataset, keyword=[keyword1, keyword2, keyword3, keyword4])
        
        x_start_dict = coord_dict[keyword1]
        x_end_dict = coord_dict[keyword2]
        y_start_dict = coord_dict[keyword3]
        y_end_dict = coord_dict[keyword4]
        
        if None in [x_start_dict, x_end_dict, y_start_dict, y_end_dict]:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        for ext_name_ver, x_start in x_start_dict.iteritems():
            x_end = x_end_dict[ext_name_ver]
            y_start = y_start_dict[ext_name_ver]
            y_end = y_end_dict[ext_name_ver]
            
            if None in [x_start, x_end, y_start, y_end]:
                data_section = None
            elif pretty:
                # Return a dictionary with the data section string that uses
                # 1-based indexing as the value in the form [x1:x2,y1:y2] 
                data_section = "[%d:%d,%d:%d]" % (
                    x_start + 1, x_end + 1, y_start + 1, y_end + 1)
            else:
                # Return a dictionary with the data section list that uses
                # 0-based, non-inclusive indexing as the value in the form
                # [x1, x2, y1, y2]
                data_section = [x_start, x_end + 1, y_start, y_end + 1]
            
            # Update the dictionary with the data section value
            ret_data_section.update({ext_name_ver:data_section})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_data_section, name="data_section",
                                 ad=dataset)
        return ret_dv
    
    array_section = data_section
    detector_section = data_section
    
    def detector_roi_setting(self, dataset, **args):
        roi_setting = "Custom"
        data_section = dataset.data_section().as_pytype()
        
        # The data_section list uses 0-based, non-inclusive indexing
        if data_section == [0, 256, 0, 256]:
            roi_setting = "Central256"
        if data_section == [0, 512, 0, 512]:
            roi_setting = "Central512"
        if data_section == [0, 768, 0, 768]:
            roi_setting = "Central768"
        if data_section == [0, 1024, 0, 1024]:
            roi_setting = "Full Frame"
        
        ret_roi_setting = roi_setting
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_roi_setting, name="roi_setting",
                                 ad=dataset)
        return ret_dv
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        
        # Disperser can only ever be in key_filter3 because the other two
        # wheels are in an uncollimated beam. Determine the filter name keyword
        # from the global keyword dictionary
        keyword = self.get_descriptor_key("key_filter3")
        
        # Get the value of the filter name keyword from the header of the PHU
        filter3 = dataset.phu_get_key_value(keyword)
        
        if filter3 is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Check if the filter name contains the string "grism". If it does, set
        # the disperser to the filter name value
        if "grism" in filter3:
            disperser = filter3
        else:
            # If the filter name value does not contain the string "grism",
            # return MIRROR like GMOS
            disperser = "MIRROR"
        
        if stripID and disperser is not "MIRROR":
            # Return the disperser string with the component ID stripped
            ret_disperser = gmu.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_disperser, name="disperser", ad=dataset)
        
        return ret_dv
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            # To match against the lookup table to get the pretty name, we
            # need the component IDs attached
            stripID = False
        
        # Determine the three filter name keyword from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_filter1")
        keyword2 = self.get_descriptor_key("key_filter2")
        keyword3 = self.get_descriptor_key("key_filter3")
        
        # Get the values of the three filter name keywords from the header of
        # the PHU
        filter1 = dataset.phu_get_key_value(keyword1)
        filter2 = dataset.phu_get_key_value(keyword2)
        filter3 = dataset.phu_get_key_value(keyword3)
        
        if filter1 is None or filter2 is None or filter3 is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if stripID:
            filter1 = gmu.removeComponentID(filter1)
            filter2 = gmu.removeComponentID(filter2)
            filter3 = gmu.removeComponentID(filter3)
        
        # Create list of filter values
        filters = [filter1, filter2, filter3]
        
        if pretty:
            # To match against the lookup table, the filter list must be sorted
            filters.sort()
            filter_name = gmu.filternameFrom(filters)
            if filter_name in self.niriFilternameMap:
                ret_filter_name = str(self.niriFilternameMap[filter_name])
            else:
                ret_filter_name = str(filter_name)
        else:
            ret_filter_name = gmu.filternameFrom(filters)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_filter_name, name="filter_name",
                                 ad=dataset)
        return ret_dv
    
    def gain(self, dataset, **args):
        # Get the gain value from the lookup table
        if "gain" in getattr(self, "niriSpecDict"):
            gain = self.niriSpecDict["gain"]
        else:
            raise Errors.TableKeyError()
        
        # Return the gain float
        ret_gain = float(gain)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain, name="gain", ad=dataset)
        
        return ret_dv

    def group_id(self, dataset, **args):
        # For NIRI image data, the group id contains the read_mode,
        # well_depth_setting, detector_section.
        # In addition flats, twilights and camera have the pretty version of the
        # filter_name included. For science data the pretty version of the
        # observation_id, filter_name and the camera are also included.
        #
        # Currently for spectroscopic data the disperser and focal_plane_mask is
        # included too

        # Descriptors used for all image types
        unique_id_descriptor_list_all = ["read_mode", "well_depth_setting",
                                         "detector_section"]
                                         

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
        flat_twilight_id = ["observation_id", "filter_name", "camera"]
        science_id = ["observation_id", "filter_name", "camera"]

        # This is used for imaging flats and twilights to distinguish between
        # the two type
        additional_item_to_include = None
        
        # Update the list of descriptors to be used depending on image type
        ## This requires updating to cover all spectral types
        ## Possible updates to the classification system will make this usable
        ## at the Gemini level
        data_types = dataset.types
        if "NIRI_DARK" in  data_types:
            id_descriptor_list = dark_id
        elif "NIRI_IMAGE_FLAT" in data_types:
            id_descriptor_list = flat_twilight_id
            additional_item_to_include = "NIRI_IMAGE_FLAT"
        elif "NIRI_IMAGE_TWILIGHT" in data_types:
            id_descriptor_list = flat_twilight_id
            additional_item_to_include = "NIRI_IMAGE_TWILIGHT"            
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
    
    niriSpecDict = None
    
    def nominal_photometric_zeropoint(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_nominal_photometric_zeropoint = {}
        table = Nominal_Zeropoints.nominal_zeropoints
        
        # Get the values of the gain, detector name and filter name using the
        # appropriate descriptors. Use as_pytype() to return the values as the
        # default python type rather than an object.
        gain = dataset.gain().as_pytype()
        camera = dataset.camera().as_pytype()
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
        # Get the saturation level using the appropriate descriptor
        saturation_level = dataset.saturation_level()
        
        if saturation_level is None:
            # The descriptor functions return None if a value cannot be found 
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # The array is non-linear at some fraction of the saturation level.
        # Get this fraction from the lookup table
        if "linearlimit" in getattr(self, "niriSpecDict"):
            linearlimit = self.niriSpecDict["linearlimit"]
        else:
            raise Errors.TableKeyError()
        
        # Return the non linear level integer
        ret_non_linear_level = int(saturation_level * linearlimit)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_non_linear_level, name="non_linear_level",
                                 ad=dataset)
        return ret_dv
    
    niriSpecDict = None
    
    def pixel_scale(self, dataset, **args):
        # Determine the WCS matrix elements keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_cd11")
        keyword2 = self.get_descriptor_key("key_cd12")
        keyword3 = self.get_descriptor_key("key_cd21")
        keyword4 = self.get_descriptor_key("key_cd22")
        
        # Get the values of the WCS matrix elements keywords from the header of
        # the PHU
        cd11 = dataset.phu_get_key_value(keyword1)
        cd12 = dataset.phu_get_key_value(keyword2)
        cd21 = dataset.phu_get_key_value(keyword3)
        cd22 = dataset.phu_get_key_value(keyword4)
        
        if cd11 is None or cd12 is None or cd21 is None or cd22 is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Calculate the pixel scale using the WCS matrix elements
        pixel_scale = 3600 * (math.sqrt(math.pow(cd11, 2) +
                                        math.pow(cd12, 2)) +
                              math.sqrt(math.pow(cd21, 2) +
                                        math.pow(cd22, 2))) / 2
        
        # Return the pixel scale float
        ret_pixel_scale = float(pixel_scale)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    def pupil_mask(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        
        # Determine the filter name keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_filter3")
        
        # Get the value of the filter name keyword from the header of the PHU
        filter3 = dataset.phu_get_key_value(keyword)
        
        if filter3 is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Check if the filter name contains the string "grism". If it does, set
        # the disperser to the filter name value
        if filter3.startswith("pup"):
            pupil_mask = filter3
        else:
            # If the filter name value does not contain the string "grism",
            # return MIRROR like GMOS
            pupil_mask = "MIRROR"
        
        if stripID and pupil_mask is not "MIRROR":
            # Return the pupil mask string with the component ID stripped
            ret_pupil_mask = gmu.removeComponentID(pupil_mask)
        else:
            # Return the pupil_mask string
            ret_pupil_mask = str(pupil_mask)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pupil_mask, name="pupil_mask", ad=dataset)
        
        return ret_dv
    
    def read_mode(self, dataset, **args):
        # Determine the number of non-destructive read pairs (lnrs) and the
        # number of digital averages (ndavgs) keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_lnrs")
        keyword2 = self.get_descriptor_key("key_ndavgs")
        
        # Get the values of the number of non-destructive read pairs and the
        # number of digital averages keywords from the header of the PHU
        lnrs = dataset.phu_get_key_value(keyword1)
        ndavgs = dataset.phu_get_key_value(keyword2)
        
        if lnrs is None or ndavgs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs == 16 and ndavgs == 16:
            read_mode = "Low Background"
        elif lnrs == 1 and ndavgs == 16:
            read_mode = "Medium Background"
        elif lnrs == 1 and ndavgs == 1:
            read_mode = "High Background"
        else:
            read_mode = "Invalid"
        
        # Return the read mode string
        ret_read_mode = str(read_mode)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_mode, name="read_mode", ad=dataset)
        
        return ret_dv
    
    def read_noise(self, dataset, **args):
        # Get the read mode and the number of coadds using the appropriate
        # descriptors
        read_mode = dataset.read_mode()
        coadds = dataset.coadds()
        
        if read_mode is None or coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Use the value of the read mode to get the read noise from the lookup
        # table
        if read_mode == "Low Background":
            key = "lowreadnoise"
        if read_mode == "High Background":
            key = "readnoise"
        else:
            key = "medreadnoise"
        if key in getattr(self, "niriSpecDict"):
            read_noise = self.niriSpecDict[key]
        else:
            raise Errors.TableKeyError()
        
        # Return the read noise float
        ret_read_noise = float(read_noise * math.sqrt(coadds))
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_noise, name="read_noise", ad=dataset)
        
        return ret_dv
    
    niriSpecDict = None
    
    def saturation_level(self, dataset, **args):
        # Note that this descriptor does not currently account for any image 
        # stacking. If stacked images are averaged, this saturation level will 
        # be correct, but if they are added, the saturation level will need to 
        # be multiplied by the number of stacked images.
        
        # Get the number of coadds, the gain and the well depth setting values
        # using the appropriate descriptors
        coadds = dataset.coadds()
        gain = dataset.gain()
        well_depth_setting = dataset.well_depth_setting()
        
        if coadds is None or gain is None or well_depth_setting is None:
            # The descriptor functions return None if a value cannot be found 
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Use the value of the well depth setting to get the well depth from
        # the lookup table
        if well_depth_setting == "Shallow":
            key = "shallowwell"
        if well_depth_setting == "Deep":
            key = "deepwell"
        if key in getattr(self, "niriSpecDict"):
            well = self.niriSpecDict[key]
        else:
            raise Errors.TableKeyError()
        
        # Return the saturation level integer in units of ADU
        ret_saturation_level = int(well * coadds / gain)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_saturation_level, name="saturation_level",
                                 ad=dataset)
        return ret_dv
    
    niriSpecDict = None
    
    def wavelength_band(self, dataset, **args):
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = NIRIFilterWavelength,filter_wavelength
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
    
    def well_depth_setting(self, dataset, **args):
        # Determine the VDDUC and VDETCOM detector bias voltage post exposure
        # (avdduc and avdet, respectively) keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_avdduc")
        keyword2 = self.get_descriptor_key("key_avdet")
        
        # Get the values of the VDDUC and VDETCOM detector bias voltage post
        # exposure keywords from the header of the PHU
        avdduc = dataset.phu_get_key_value(keyword1)
        avdet = dataset.phu_get_key_value(keyword2)
        
        if avdduc is None or avdet is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        biasvolt = avdduc - avdet
        shallowbias = self.niriSpecDict["shallowbias"]
        deepbias = self.niriSpecDict["deepbias"]
        if abs(biasvolt - shallowbias) < 0.05:
            well_depth_setting = "Shallow"
        elif abs(biasvolt - deepbias) < 0.05:
            well_depth_setting = "Deep"
        else:
            well_depth_setting = "Invalid"
        
        # Return the well depth setting string
        ret_well_depth_setting = str(well_depth_setting)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_well_depth_setting,
                                 name="well_depth_setting", ad=dataset)
        return ret_dv
