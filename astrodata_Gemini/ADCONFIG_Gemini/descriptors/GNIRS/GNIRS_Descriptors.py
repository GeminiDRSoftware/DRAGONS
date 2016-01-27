import math, re

from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_metadata_utils as gmu

from GNIRS_Keywords import GNIRS_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

# ------------------------------------------------------------------------------
class GNIRS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GNIRS_KeyDict
    
    gnirsArrayDict = None
    gnirsConfigDict = None
    
    def __init__(self):
        self.gnirsArrayDict = Lookups.get_lookup_table(
            "Gemini/GNIRS/GNIRSArrayDict", "gnirsArrayDict")
        self.gnirsConfigDict = Lookups.get_lookup_table(
            "Gemini/GNIRS/GNIRSConfigDict", "gnirsConfigDict")
        GEMINI_DescriptorCalc.__init__(self)

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
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        
        # If the acquisition mirror is in, the disperser is "MIRROR"
        if dataset.phu_get_key_value('ACQMIR') == 'In':
            return DescriptorValue('MIRROR', name="disperser", ad=dataset)

        # GNIRS contains two dispersers - the grating and the prism. Get the
        # grating and the prism values using the appropriate descriptors
        grating = dataset.grating(stripID=stripID, pretty=pretty).as_pytype()
        prism = dataset.prism(stripID=stripID, pretty=pretty).as_pytype()
        
        if grating is None or prism is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the disperser string
        # If the "prism" is the "MIRror" then don't include it
        if prism.startswith('MIR'):
            ret_disperser = str(grating)
        else:
            ret_disperser = "%s&%s" % (grating, prism)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_disperser, name="disperser", ad=dataset)
        
        return ret_dv
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        # For GNIRS, the focal plane mask is the combination of the slit
        # mechanism and the decker mechanism. Get the slit and the decker
        # values using the appropriate descriptors
        slit = dataset.slit(stripID=stripID, pretty=pretty).as_pytype()
        decker = dataset.decker(stripID=stripID, pretty=pretty).as_pytype()
        
        if slit is None or decker is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Sometimes (2010 rebuild?) we see "Acquisition" and sometimes "Acq"
        # In both slit and decker. Make them all consistent.
        slit = slit.replace('Acquisition', 'Acq')
        decker = decker.replace('Acquisition', 'Acq')

        if pretty:
            # Disregard the decker if it's in long slit mode
            if "Long" in decker:
                focal_plane_mask = slit
            # Append XD to the slit name if the decker is in XD mode
            elif "XD" in decker:
                focal_plane_mask = "%s%s" % (slit, "XD")
            elif "IFU" in slit and "IFU" in decker:
                focal_plane_mask = "IFU"
            elif "Acq" in slit and "Acq" in decker:
                focal_plane_mask = "Acq"
            else:
                focal_plane_mask = "%s&%s" % (slit, decker)
        else:
            focal_plane_mask = "%s&%s" % (slit, decker)
        
        ret_focal_plane_mask = str(focal_plane_mask)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_focal_plane_mask, name="focal_plane_mask",
                                 ad=dataset)
        return ret_dv

    def gain(self, dataset, **args):
        # Determine the read mode and well depth from their descriptors
        read_mode = dataset.read_mode().as_pytype()
        well_depth = dataset.well_depth_setting().as_pytype()
        
        if read_mode is None or well_depth is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if (read_mode, well_depth) in getattr(self, "gnirsArrayDict"):
            gain = self.gnirsArrayDict[(read_mode, well_depth)][1]
        else:
            raise Errors.TableKeyError()
        
        # Return the gain float
        ret_gain = float(gain)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain, name="gain", ad=dataset)
        
        return ret_dv
    
    gnirsArrayDict = None

    def grating(self, dataset, stripID=False, pretty=False, **args):
        """
        Note. A CC software change approx July 2010 changed the grating names
        to also include the camera, eg 32/mmSB_G5533 indicates the 32/mm
        grating with the Short Blue camera. This is unhelpful as if we wanted
        to know the camera, we'd call the camera descriptor. Thus, this
        descriptor function repairs the header values to only list the grating.
        """
        # Determine the grating keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_grating")
        
        # Get the value of the grating keyword from the header of the PHU
        grating = dataset.phu_get_key_value(keyword)
        
        if grating is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # The format of the grating string is currently (2011) nnn/mmCAM_Gnnnn
        # nnn is a 2 or 3 digit number (lines per mm)
        # /mm is literally "/mm"
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # _G is literally "_G"
        # nnnn is the 4 digit component ID.
        cre = re.compile("([\d/m]+)([A-Z]*)(_G)(\d+)")
        m = cre.match(grating)
        if m:
            parts = m.groups()
            ret_grating = "%s%s%s" % (parts[0], parts[2], parts[3])
        else:
            # If the regex didn't match, just pass through the raw value
            ret_grating = grating
        if stripID or pretty:
            ret_grating = gmu.removeComponentID(ret_grating)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_grating, name="grating", ad=dataset)
        
        return ret_dv

    def group_id(self, dataset, **args):
        # For GNIRS image data, the group id contains the read_mode,
        # well_depth_setting, detector_section.
        # In addition flats, twilights and camera have the pretty version of the
        # filter_name included. For science data the pretty version of the
        # observation_id, filter_name and the camera are also included.

        # Descriptors used for all image types
        unique_id_descriptor_list_all = ["well_depth_setting", 
                                         "detector_section", "disperser",
                                         "focal_plane_mask"]
                                         

        # List to format descriptor calls using 'pretty=True' parameter
        call_pretty_version_list = ["filter_name", "disperser",
                                    "focal_plane_mask"]

        # Descriptors to be returned as an ordered list using descriptor
        # 'as_list' method.
        convert_to_list_list = ["detector_section"]

        # Additional descriptors required for each frame type
        dark_id = ["read_mode", "exposure_time", "coadds"]
        flat_id = ["observation_id", "filter_name", "camera", "read_mode"]
        science_id = ["observation_id", "filter_name", "camera", "read_mode"]

        # This is used for imaging flats and twilights to distinguish between
        # the two types
        additional_item_to_include = None
        
        # Update the list of descriptors to be used depending on image type
        data_types = dataset.types
        if "GNIRS_DARK" in  data_types:
            id_descriptor_list = dark_id
        elif "GNIRS_IMAGE_FLAT" in data_types:
            id_descriptor_list = flat_id
            additional_item_to_include = "GNIRS_IMAGE_FLAT"
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

    def nominal_photometric_zeropoint(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_nominal_photometric_zeropoint = {}
        
        table = Lookups.get_lookup_table("Gemini/GNIRS/Nominal_Zeropoints",
                                         "nominal_zeropoints")
        # Get the values of the gain, detector name and filter name using the
        # appropriate descriptors. 
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
        
        # Determine the read mode and well depth from their descriptors
        read_mode = dataset.read_mode().as_pytype()
        well_depth = dataset.well_depth_setting().as_pytype()
        if read_mode is None or well_depth is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if (read_mode, well_depth) in getattr(self, "gnirsArrayDict"):
            linear_limit = self.gnirsArrayDict[(read_mode, well_depth)][3]
        else:
            raise Errors.TableKeyError()
        
        # Return the non-linear level int
        ret_non_linear_level = int(linear_limit * saturation_level)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_non_linear_level, name="non_linear_level", 
                                 ad=dataset)
        
        return ret_dv
    
    gnirsArrayDict = None

    def pixel_scale(self, dataset, **args):
        
        # Get the camera using the appropriate descriptor
        camera = dataset.camera().as_pytype()
        
        if camera is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        # Imaging or darks
        if any(type in dataset.type() for type in ["GNIRS_IMAGE", "GNIRS_DARK"]):
            # The format of the camera string is either Short or Long, 
            # followed by either Red or Blue, then _G and 4 digits
            cre = re.compile("^(Short|Long)(Red|Blue)(_G)(\d+)$")
            m = cre.match(camera)
            cameratype = m.group(1)
            if cameratype == "Short":
                ret_pixel_scale = 0.15
            if cameratype == "Long":
                ret_pixel_scale = 0.05
            if ret_pixel_scale is None:
                raise Exception('No camera match for imaging mode')
            
        #Spectroscopy
        else:
            
            # Determine the prism, decker and disperser keywords from the 
            # global keyword dictionary
            keyword1 = self.get_descriptor_key("key_prism")
            keyword2 = self.get_descriptor_key("key_decker")
            keyword3 = self.get_descriptor_key("key_grating")

            # Get the value of the prism, decker and disperser keywords from 
            # the header of the PHU 
            prism = dataset.phu_get_key_value(keyword1)
            decker = dataset.phu_get_key_value(keyword2)
            disperser = dataset.phu_get_key_value(keyword3)
        
            if prism is None or decker is None or disperser is None:
                # The phu_get_key_value() function returns None if a value 
                # cannot be found and stores the exception info. Re-raise 
                # the exception. It will be dealt with by the 
                # CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
                
            pixel_scale_key = (prism, decker, disperser, camera)
            if pixel_scale_key in getattr(self, "gnirsConfigDict"):
                row = self.gnirsConfigDict[pixel_scale_key]
            else:
                raise Errors.TableKeyError()
        
            if float(row[2]):
                ret_pixel_scale = float(row[2])
            else:
                raise Errors.TableValueError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    gnirsConfigDict = None
    
    def prism(self, dataset, stripID=False, pretty=False, **args):
        """
        Note. A CC software change approx July 2010 changed the prism names to
        also include the camera, eg 32/mmSB_G5533 indicates the 32/mm grating
        with the Short Blue camera. This is unhelpful as if we wanted to know
        the camera, we'd call the camera descriptor. Thus, this descriptor
        function repairs the header values to only list the prism.
        """
        # Determine the prism keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_prism")
        
        # Get the value of the prism keyword from the header of the PHU.
        prism = dataset.phu_get_key_value(keyword)
        
        if prism is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # The format of the prism string is currently (2011) [CAM+]prism_Gnnnn
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # + is a literal "+"
        # prism is the actual prism name
        # nnnn is the 4 digit component ID.
        # The change from the old style is the camer prefix, which we drop here
        cre = re.compile("([LBSR]*\+)*([A-Z]*_G\d+)")
        m = cre.match(prism)
        if m:
            ret_prism = m.group(2)

        if stripID or pretty:
                ret_prism = gmu.removeComponentID(ret_prism)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_prism, name="prism", ad=dataset)
        
        return ret_dv
    
    def ra(self, dataset, **args):
        # In general, the GNIRS WCS is the way to go. But sometimes the DC
        # has a bit of a senior moment and the WCS is miles off (presumably
        # still has values from the previous observation or something. Who knows.
        # So we do a sanity check on it and use the target values if it's messed up
        wcs_ra = dataset.wcs_ra().as_pytype()
        target_ra = dataset.target_ra(offset=True, icrs=True).as_pytype()
        delta = abs(wcs_ra - target_ra)
        # wraparound?
        if delta > 180.0:
            delta = abs(delta-360.0)
        delta *= 3600.0 # put in arcsecs

        # And account for cos(dec) factor
        delta /= math.cos(math.radians(dataset.dec().as_pytype()))

        # If more than 1000" arcsec different, WCS is probably bad
        if delta > 1000:
            return dataset.target_ra(offset=True, icrs=True)
        else:
            return dataset.wcs_ra()

    def dec(self, dataset, **args):
        # In general, the GNIRS WCS is the way to go. But sometimes the DC
        # has a bit of a senior moment and the WCS is miles off (presumably
        # still has values from the previous observation or something. Who knows.
        # So we do a sanity check on it and use the target values if it's messed up
        wcs_dec = dataset.wcs_dec().as_pytype()
        target_dec = dataset.target_dec(offset=True, icrs=True).as_pytype()
        delta = abs(wcs_dec - target_dec)
        # wraparound?
        if delta > 180.0:
            delta = abs(delta-360.0)
        delta *= 3600.0 # put in arcsecs

        # If more than 1000" arcsec different, WCS is probably bad
        if delta > 1000:
            return dataset.target_dec(offset=True, icrs=True)
        else:
            return dataset.wcs_dec()

    def read_mode(self, dataset, **args):
        # Determine the number of non-destructive read pairs (lnrs) and the
        # number of digital averages (ndavgs) keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_lnrs")
        keyword2 = self.get_descriptor_key("key_ndavgs")
        
        # Get the value of the number of non-destructive read pairs and the
        # number of digital averages keywords from the header of the PHU
        lnrs = dataset.phu_get_key_value(keyword1)
        ndavgs = dataset.phu_get_key_value(keyword2)
        
        if lnrs is None or ndavgs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs == 32 and ndavgs == 16:
            read_mode = "Very Faint Objects"
        elif lnrs == 16 and ndavgs == 16:
            read_mode = "Faint Objects"
        elif lnrs == 1 and ndavgs == 16:
            read_mode = "Bright Objects"
        elif lnrs == 1 and ndavgs == 1:
            read_mode = "Very Bright Objects"
        else:
            read_mode = "Invalid"
        
        ret_read_mode = str(read_mode)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_mode, name="read_mode", ad=dataset)
        
        return ret_dv
 
    def read_noise(self, dataset, **args):
        # Determine the read mode and well depth from their descriptors
        read_mode = dataset.read_mode().as_pytype()
        well_depth = dataset.well_depth_setting().as_pytype()
        
        if read_mode is None or well_depth is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if (read_mode, well_depth) in getattr(self, "gnirsArrayDict"):
            read_noise = self.gnirsArrayDict[(read_mode, well_depth)][0]
        else:
            raise Errors.TableKeyError()
        
        coadds = dataset.coadds()
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the read noise float
        ret_read_noise = float(read_noise * math.sqrt(coadds))
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_noise, name="read_noise", ad=dataset)
        
        return ret_dv
    
    gnirsArrayDict = None
   
    def saturation_level(self, dataset, **args):
        # Get the gain and number of coadds using the appropriate descriptor
        gain = dataset.gain()
        coadds = dataset.coadds()
        if gain is None or coadds is None:
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        # Determine the read mode and well depth from their descriptors
        read_mode = dataset.read_mode().as_pytype()
        well_depth = dataset.well_depth_setting().as_pytype()
        if read_mode is None or well_depth is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        if (read_mode, well_depth) in getattr(self, "gnirsArrayDict"):
            well = self.gnirsArrayDict[(read_mode, well_depth)][2]
        else:
            raise Errors.TableKeyError()
        
        # Return the saturation level in units of ADU
        ret_saturation_level = int(well * coadds / gain) 
               
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_saturation_level, name="saturation_level", 
                                 ad=dataset)
        
        return ret_dv
    
    gnirsArrayDict = None
   
    def slit(self, dataset, stripID=False, pretty=False, **args):
        """
        Note that in GNIRS all the slits are machined into one physical piece
        of metal, which is on a slide - the mechanism simply slides the slide
        along to put the right slit in the beam. Thus all the slits have the
        same component ID as they're they same physical component.

        Note that in the ~2010 rebuild, the slit names were changed to remove
        the space - ie "1.00 arcsec" -> "1.00arcsec"
        So here, we remove the space all the time for consistency.
        """
        # Determine the slit keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_slit")
        
        # Get the value of the slit keyword from the header of the PHU
        slit = dataset.phu_get_key_value(keyword)
        
        if slit is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        slit = slit.replace(' ', '')
        
        if stripID or pretty:
            ret_slit = gmu.removeComponentID(slit)
        else:
            ret_slit = str(slit)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_slit, name="slit", ad=dataset)
        
        return ret_dv
    
    def wavelength_band(self, dataset, **args):
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = Lookups.get_lookup_table(
                "Gemini/GNIRS/GNIRSFilterWavelength", "filter_wavelength")
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
        # Determine the bias value (biasvolt) keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if abs(0.3 - abs(biasvolt)) < 0.1:
            well_depth_setting = "Shallow"
        elif abs(0.6 - abs(biasvolt)) < 0.1:
            well_depth_setting = "Deep"
        else:
            well_depth_setting = "Invalid"
        
        # Return the well depth setting string
        ret_well_depth_setting = str(well_depth_setting)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_well_depth_setting,
                                 name="well_depth_setting", ad=dataset)
        return ret_dv
