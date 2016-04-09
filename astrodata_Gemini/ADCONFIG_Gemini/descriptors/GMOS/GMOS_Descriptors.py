import math
import numpy as np
from datetime import datetime
from time import strptime

from astrodata.utils import Errors
from astrodata.interface.slices import pixel_exts
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_data_calculations as gdc
from gempy.gemini import gemini_metadata_utils as gmu

import GemCalcUtil

from GMOS_Keywords import GMOS_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import ROItable
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSAmpTables
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import Nominal_Zeropoints
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSPixelScale
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSReadModes
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSThresholdValues
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSFilterWavelength

# ------------------------------------------------------------------------------
class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GMOS_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def amp_read_area(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_amp_read_area_dict = {}
        
        # Determine the name of the detector amplifier keyword (ampname) from
        # the global keyword dictionary 
        keyword = self.get_descriptor_key("key_ampname")
        
        # Get the value of the name of the detector amplifier keyword from the
        # header of each pixel data extension as a dictionary where the key of
        # the dictionary is an ("*", EXTVER) tuple
        ampname_dict = gmu.get_key_value_dict(adinput=dataset, keyword=keyword)
        
        if ampname_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the pretty (1-based indexing) readout area of the CCD using the
        # appropriate descriptor
        detector_section_dv = dataset.detector_section(pretty=True)
        
        if detector_section_dv.is_none():
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface. 
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Use as_dict() to return the detector section value as a dictionary
        # where the key of the dictionary is an ("*", EXTVER) tuple 
        detector_section_dict = detector_section_dv.as_dict()
        
        for ext_name_ver, ampname in ampname_dict.iteritems():
            detector_section = detector_section_dict[ext_name_ver]
            
            if ampname is None or detector_section is None:
                amp_read_area = None
            else:
                # Use the composite amp_read_area string as the value
                amp_read_area = "'%s':%s" % (ampname, detector_section)
            
            # Update the dictionary with the amp_read_area value
            ret_amp_read_area_dict.update({ext_name_ver:amp_read_area})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_amp_read_area_dict, name="amp_read_area",
                                 ad=dataset)
        return ret_dv
    
    def array_name(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        array_name_dict = {}
        
        # Determine the name of the array keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_array_name")
        
        # Get the value of the name of the array keyword from the header of
        # each pixel data extension as a dictionary where the key of the
        # dictionary is an ("*", EXTVER) tuple
        array_name_dict = gmu.get_key_value_dict(adinput=dataset,
                                                 keyword=keyword)
        if array_name_dict is None:
            # It is possible that the data have been mosaiced, which means that
            # the name of the array keyword no longer exists in the pixel data
            # extensions. Instead, determine the value of the detector name
            # using the appropriate descriptor
            detector_name_dv = dataset.detector_name()
            
            if detector_name_dv.is_none():
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface. 
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            ret_array_name = detector_name_dv
        else:
            ret_array_name = array_name_dict
        
        # Instantiate the return DescriptorValue (DV) object using the newly
        # created dictionary
        ret_dv = DescriptorValue(ret_array_name, name="array_name", ad=dataset)
        
        return ret_dv
    
    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):
        # Currently for GMOS data, the central wavelength is recorded in
        # nanometers
        input_units = "nanometers"
        
        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units.
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
        
        # GMOS has a grating wavelength keyword GRWLEN and also a 
        # Central wavelength header CENTWAVE. Generally they are the same, 
        # except for when the slit is offset from the center of the focal plane
        # as with IFU-1. Central wavelength is the one we want, but this header
        # was only added circa 2007, so older data do not have it, in which case
        # fall back to grating wavelenth. Hopefully in a future version we can
        # add the calculation to correct the value for older data.
        
        # Get the value of the central wavelength keyword from the header of
        # the PHU
        raw_central_wavelength = dataset.phu_get_key_value('CENTWAVE')
        if raw_central_wavelength is None:
            # Probaly it's old data that doesn't have the CENTWAVE header
            raw_central_wavelength = dataset.phu_get_key_value('GRWLEN')
        
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
    
    def detector_name(self, dataset, pretty=False, **args):
        # Determine the name of the detector keyword from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_detector_name")
        
        # Get the value of the name of the detector keyword from the header of
        # the PHU
        detector_name = dataset.phu_get_key_value(keyword1)
        
        if detector_name is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if pretty:
            # Define relationship between the type of the detector and the
            # pretty name of the detector
            pretty_detector_name_dict = {
                "SDSU II CCD": "EEV",
                "SDSU II e2v DD CCD42-90": "e2vDD",
                "S10892": "Hamamatsu",
                }
            
            # Determine the type of the detector keyword from the global
            # keyword dictionary
            keyword2 = self.get_descriptor_key("key_detector_type")
            
            # Get the value of the type of the detector keyword from the header
            # of the PHU
            detector_type = dataset.phu_get_key_value(keyword2)
            
            # Return the pretty name of the detector
            if detector_type in pretty_detector_name_dict:
                ret_detector_name = pretty_detector_name_dict[detector_type]
            else:
                errmsg = ("'{0}' key not found in "
                          "{1}".format(detector_type,
                                       "pretty_detector_name_dict"))
                                                          
                raise Errors.TableKeyError(errmsg)
        else:
            # Return the name of the detectory
            ret_detector_name = detector_name
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_name, name="detector_name",
                                 ad=dataset)
        return ret_dv
    
    def detector_rois_requested(self, dataset, **args):
        # This parses the DETROx GMOS headers and returns a list of ROIs in the
        # form [x1, x2, y1, y2]. These are in physical pixels - should be
        # irrespective of binning. These are 1-based, and inclusive, 2 pixels,
        # starting at pixel 2 would be [2, 3].
        ret_detector_rois_requested_list = []
        
        # Must be single digit ROI number
        for i in range(1, 10):
            x1 = dataset.phu_get_key_value("DETRO%sX" % i)
            xs = dataset.phu_get_key_value("DETRO%sXS" % i)
            y1 = dataset.phu_get_key_value("DETRO%sY" % i)
            ys = dataset.phu_get_key_value("DETRO%sYS" % i)
            
            if x1 is not None:
                # The headers are in the form of a start position and size
                # so make them into start and end pixels here
                xs *= int(dataset.detector_x_bin())
                ys *= int(dataset.detector_y_bin())
                ret_detector_rois_requested_list.append(
                    [x1, x1+xs-1, y1, y1+ys-1])
            else:
                break
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_rois_requested_list,
                                 name="detector_rois_requested", ad=dataset)
        return ret_dv
    
    def detector_roi_setting(self, dataset, **args):
        """
        Attempts to deduce the Name of the ROI, more or less as per the options
        you can select in the OT. Only considers the first ROI.
        
        """
        # Get the lookup table containing the ROI sections
        gmosRoiSettings = ROItable.gmosRoiSettings
        ret_detector_roi_setting = "Undefined"
        rois = dataset.detector_rois_requested()
        if rois.is_none():
            # The descriptor functions return None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        rois = rois.as_list()
        if rois:
            roi = rois[0]
            
            # If we don't recognise it, it's "Custom"
            ret_detector_roi_setting = "Custom"
            
            for s in gmosRoiSettings.keys():
                if roi in gmosRoiSettings[s]:
                    ret_detector_roi_setting = s
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_roi_setting,
                                 name="detector_roi_setting", ad=dataset)
        return ret_dv
    
    def detector_x_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_detector_x_bin_dict = {}
        
        # Determine the ccdsum keyword from the global keyword dictionary 
        keyword = self.get_descriptor_key("key_ccdsum")
        
        # Get the value of the ccdsum keyword from the header of each pixel
        # data extension as a dictionary where the key of the dictionary is an
        # ("*", EXTVER) tuple
        ccdsum_dict = gmu.get_key_value_dict(adinput=dataset, keyword=keyword)
        
        if ccdsum_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        for ext_name_ver, ccdsum in ccdsum_dict.iteritems():
            if ccdsum is None:
                detector_x_bin = None
            else:
                # Use the binning of the x-axis integer as the value
                x_bin, y_bin = ccdsum.split()
                detector_x_bin = int(x_bin)
            
            # Update the dictionary with the binning of the x-axis value
            ret_detector_x_bin_dict.update({ext_name_ver:detector_x_bin})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_x_bin_dict,
                                 name="detector_x_bin", ad=dataset)
        return ret_dv
    
    def detector_y_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_detector_y_bin_dict = {}
        
        # Determine the ccdsum keyword from the global keyword dictionary 
        keyword = self.get_descriptor_key("key_ccdsum")
        
        # Get the value of the ccdsum keyword from the header of each pixel
        # data extension as a dictionary where the key of the dictionary is an
        # ("*", EXTVER) tuple
        ccdsum_dict = gmu.get_key_value_dict(adinput=dataset, keyword=keyword)
        
        if ccdsum_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        for ext_name_ver, ccdsum in ccdsum_dict.iteritems():
            if ccdsum is None:
                detector_y_bin = None
            else:
                # Use the binning of the y-axis integer as the value
                x_bin, y_bin = ccdsum.split()
                detector_y_bin = int(y_bin)
            
            # Update the dictionary with the binning of the y-axis value
            ret_detector_y_bin_dict.update({ext_name_ver:detector_y_bin})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_y_bin_dict,
                                 name="detector_y_bin", ad=dataset)
        return ret_dv
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        # Determine the disperser keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_disperser")
        
        # Get the value of the disperser keyword from the header of the PHU
        disperser = dataset.phu_get_key_value(keyword)
        
        if disperser is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if pretty:
            # If pretty=True, use stripID then additionally remove the
            # trailing "+" from the string
            stripID = True
        
        if stripID:
            if pretty:
                # Return the stripped and pretty disperser string
                ret_disperser = gmu.removeComponentID(disperser).strip("+")
            else:
                # Return the stripped disperser string
                ret_disperser = gmu.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_disperser, name="disperser", ad=dataset)
        
        return ret_dv
    
    def dispersion(self, dataset, asMicrometers=False, asNanometers=False,
                   asAngstroms=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_dispersion_dict = {}
        
        # Currently for GMOS data, the dispersion is recorded in meters (?)
        input_units = "meters"
        
        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # dispersion in these units.
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the dispersion in the default units of meters.
            output_units = "meters"
        
        # Determine the dispersion keyword from the global keyword dictionary 
        keyword = self.get_descriptor_key("key_dispersion")
        
        # Get the value of the dispersion keyword from the header of each pixel
        # data extension as a dictionary where the key of the dictionary is an
        # ("*", EXTVER) tuple
        dispersion_dict = gmu.get_key_value_dict(adinput=dataset,
                                                 keyword=keyword)
        if dispersion_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        for ext_name_ver, raw_dispersion in dispersion_dict.iteritems():
            if raw_dispersion is None:
                dispersion = None
            else:
                # Use the utilities function convert_units to convert the
                # dispersion value from the input units to the output units
                dispersion = float(GemCalcUtil.convert_units(
                    input_units=input_units, input_value=float(raw_dispersion),
                    output_units=output_units))
            
            # Update the dictionary with the dispersion value
            ret_dispersion_dict.update({ext_name_ver:dispersion})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_dispersion_dict, name="dispersion",
                                 ad=dataset)
        return ret_dv
    
    def dispersion_axis(self, dataset, **args):
        # The GMOS dispersion axis should always be 1
        ret_dispersion_axis = 1
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_dispersion_axis, name="dispersion_axis",
                                 ad=dataset)
        return ret_dv
    
    def exposure_time(self, dataset, **args):
        # Determine the exposure time keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_exposure_time")
        
        # Get the value of the exposure time keyword from the header of the PHU
        exposure_time = dataset.phu_get_key_value(keyword)
        
        if exposure_time is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Sanity check for times when the GMOS DC is stoned
        if exposure_time > 10000. or exposure_time < 0.:
            raise Errors.InvalidValueError()
        else:
            # Return the exposure time float
            ret_exposure_time = float(exposure_time)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_exposure_time, name="exposure_time",
                                 ad=dataset)
        return ret_dv
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        # Determine the focal plane mask keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_focal_plane_mask")
        
        # Get the focal plane mask value from the header of the PHU
        focal_plane_mask = dataset.phu_get_key_value(keyword)
        
        if focal_plane_mask is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if focal_plane_mask == "None":
            ret_focal_plane_mask = "Imaging"
        else:
            # Return the focal plane mask string
            ret_focal_plane_mask = str(focal_plane_mask)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_focal_plane_mask, name="focal_plane_mask",
                                 ad=dataset)
        return ret_dv
    
    def gain(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_gain_dict = {}
        
        # If the data have been prepared, take the gain value directly from the
        # appropriate keyword. At some point, a check for the STDHDRSI header
        # keyword should be added, since the function that overwrites the gain
        # keyword also writes the STDHDRSI keyword.
        if "PREPARED" in dataset.types:
            
            # Determine the gain keyword from the global keyword dictionary 
            keyword = self.get_descriptor_key("key_gain")
            
            # Get the value of the gain keyword from the header of each pixel
            # data extension as a dictionary where the key of the dictionary is
            # an ("*", EXTVER) tuple
            gain_dict = gmu.get_key_value_dict(adinput=dataset,
                                               keyword=keyword)
            if gain_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            ret_gain_dict = gain_dict
        
        else:
            # Get the lookup table containing the gain values by amplifier
            gmosampsGain = GMOSAmpTables.gmosampsGain
            gmosampsGainBefore20150826 = GMOSAmpTables.gmosampsGainBefore20150826
            gmosampsGainBefore20060831 = GMOSAmpTables.gmosampsGainBefore20060831
            
            # Determine the amplifier integration time keyword (ampinteg) from
            # the global keyword dictionary
            keyword = self.get_descriptor_key("key_ampinteg")
            
            # Get the value of the amplifier integration time keyword from the
            # header of the PHU
            ampinteg = dataset.phu_get_key_value(keyword)
            
            if ampinteg is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Get the UT date, gain setting and read speed setting values using
            # the appropriate descriptors
            ut_date_dv = dataset.ut_date()
            gain_setting_dv = dataset.gain_setting()
            read_speed_setting_dv = dataset.read_speed_setting()
            
            if (ut_date_dv.is_none() or gain_setting_dv.is_none() or
                read_speed_setting_dv.is_none()):
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Use as_dict() and as_pytype() to return the values as a
            # dictionary where the key of the dictionary is an ("*", EXTVER)
            # tuple and the default python type, respectively, rather than an
            # object
            ut_date = str(ut_date_dv)
            gain_setting_dict = gain_setting_dv.as_dict()
            read_speed_setting = read_speed_setting_dv.as_pytype()
            
            obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
            # These dates really shouldn't be hard wired so sloppily all over
            # the place (including gempy gemini_data_calculations.py) but that
            # goes as far as the dictionary names so leave it for a possible
            # future clean up of how the dictionaries are keyed.
            change_2015_ut = datetime(2015, 8, 26, 0, 0)
            change_2006_ut = datetime(2006, 8, 31, 0, 0)
            
            # Determine the name of the detector amplifier keyword (ampname)
            # from the global keyword dictionary 
            keyword = self.get_descriptor_key("key_ampname")
            
            # Get the value of the name of the detector amplifier keyword from
            # the header of each pixel data extension as a dictionary where the
            # key of the dictionary is an ("*", EXTVER) tuple
            ampname_dict = gmu.get_key_value_dict(adinput=dataset,
                                                  keyword=keyword)
            if ampname_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            for ext_name_ver, ampname in ampname_dict.iteritems():
                gain_setting = gain_setting_dict[ext_name_ver]
                
                if ampname is None or gain_setting is None:
                    gain = None
                else:
                    gain_key = (read_speed_setting, gain_setting, ampname)
                    
                    if obs_ut_date > change_2015_ut:
                        gain_dict = gmosampsGain
                    elif obs_ut_date > change_2006_ut:
                        gain_dict = gmosampsGainBefore20150826
                    else:
                        gain_dict = gmosampsGainBefore20060831

                    if gain_key in gain_dict:
                        gain = gain_dict[gain_key]
                    else:
                        raise Errors.TableKeyError()

                # Update the dictionary with the gain value
                ret_gain_dict.update({ext_name_ver:gain})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain_dict, name="gain", ad=dataset)
        
        return ret_dv
    
    def gain_setting(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_gain_setting_dict = {}
        
        # If the data have not been prepared, take the raw gain value directly
        # from the appropriate keyword
        if "PREPARED" not in dataset.types:
            
            # Determine the gain keyword from the global keyword dictionary 
            keyword = self.get_descriptor_key("key_gain")
            
            # Get the value of the gain keyword from the header of each pixel
            # data extension as a dictionary where the key of the dictionary is
            # an ("*", EXTVER) tuple
            gain_dict = gmu.get_key_value_dict(adinput=dataset,
                                               keyword=keyword)
            if gain_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            for ext_name_ver, gain in gain_dict.iteritems():
                if gain is None:
                    gain_setting = None
                elif gain > 3.0:
                    gain_setting = "high"
                else:
                    gain_setting = "low"
                
                # Update the dictionary with the gain setting value
                ret_gain_setting_dict.update({ext_name_ver:gain_setting})
        
        else:
            # If the data have been prepared, take the gain setting value
            # directly from the appropriate keyword in the header of each pixel
            # data extension
            #
            # Determine the gain setting keyword from the global keyword
            # dictionary
            keyword = self.get_descriptor_key("key_gain_setting")
            
            # Get the value of the gain setting keyword from the header of each
            # pixel data extension as a dictionary where the key of the
            # dictionary is an ("*", EXTVER) tuple
            gain_setting_dict = gmu.get_key_value_dict(adinput=dataset,
                                                       keyword=keyword)

            if gain_setting_dict is not None:
                ret_gain_setting_dict = gain_setting_dict
            else:
                # The dataset was not processed using gemini_python. Try to get
                # the gain from the "GAINORIG" keyword in the header of each
                # pixel data extension as a dictionary where the key of the
                # dictionary is an ("*", EXTVER) tuple.
                gain_dict = gmu.get_key_value_dict(adinput=dataset,
                                                   keyword="GAINORIG")
                if gain_dict is None:
                    # Resort to getting the gain using the appropriate
                    # descriptor (this will use the updated gain value). Use
                    # as_dict() to return the gain value as a dictionary where
                    # the key of the dictionary is an ("*", EXTVER) tuple
                    # rather than an object. 
                    gain_dv = dataset.gain()
                    if gain_dv.is_none():
                        # The descriptor functions return None if a value
                        # cannot be found and stores the exception
                        # info. Re-raise the exception. It will be dealt with
                        # by the CalculatorInterface. 
                        if hasattr(dataset, "exception_info"):
                            raise dataset.exception_info
                    gain_dict = gain_dv.as_dict()
                    
                else:
                    for ext_name_ver, gain_orig in gain_dict.iteritems():
                        count_exts = 0
                        count_gain_orig = 0
                        if gain_orig == 1:
                            # If the gain from the "GAINORIG" keyword is equal
                            # to 1, the very original gain was written to
                            # "GAINMULT" 
                            count_gain_orig += 1
                        count_exts += 1
                    
                    if count_exts == count_gain_orig:
                        # The value of "GAINORIG" keyword is equal to 1 in all
                        # pixel data extensions. Try to get the gain from the
                        # "GAINMULT" keyword in the header of each pixel data
                        # extension as a dictionary where the key of the
                        # dictionary is an ("*", EXTVER) tuple.
                        gain_dict = gmu.get_key_value_dict(adinput=dataset,
                                                           keyword="GAINMULT")
                if gain_dict is None:
                    # The get_key_value_dict() function returns None if a value
                    # cannot be found and stores the exception info. Re-raise
                    # the exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(dataset, "exception_info"):
                        raise dataset.exception_info
                
                for ext_name_ver, gain in gain_dict.iteritems():
                    if gain is None:
                        gain_setting = None
                    elif gain > 3.0:
                        gain_setting = "high"
                    else:
                        gain_setting = "low"
                    
                    # Update the dictionary with the gain setting value
                    ret_gain_setting_dict.update({ext_name_ver:gain_setting})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain_setting_dict, name="gain_setting",
                                 ad=dataset)
        return ret_dv
    
    def group_id(self, dataset, **args):
        # For GMOS image data, the group id contains the detector_x_bin,
        # detector_y_bin, amp_read_area, gain_setting and read_speed_setting.
        # In addition flats and twilights have the pretty version of the
        # filter_name included. Also, for science data the pretty version of
        # the filter_name and the observation_id are also included.
        #
        # Currently for spectroscopic data the grating is included too.

        # Descriptors used for all frame types
        unique_id_descriptor_list_all = ["detector_x_bin", "detector_y_bin",
                                         "read_mode", "amp_read_area"]

        # List to format descriptor calls using 'pretty=True' parameter
        call_pretty_version_list = ["filter_name", "disperser"]

        # Descriptors to be returned as an ordered list using descriptor
        # 'as_list' method.
        convert_to_list_list = ["amp_read_area"]

        # Other descriptors required for spectra 
        required_spectra_descriptors = ["disperser"]

        ## This will probably require more thought in the future for spectral
        ## flats and twilights. Possibly even if it's required...
        if "SPECT" in dataset.types:
            unique_id_descriptor_list_all.extend(required_spectra_descriptors)

        # Additional descriptors required for each frame type
        bias_id = []
        dark_id = ["exposure_time"]
        flat_twilight_id = ["filter_name"]
        science_id = ["observation_id", "filter_name"]

        # Update the list of descriptors to be used depending on image type
        #
        ## This requires updating to cover all spectral types
        ## Possible updates to the classification system will make this usable
        ## at the Gemini level
        data_types = dataset.types
        if "GMOS_BIAS" in data_types:
            id_descriptor_list = bias_id
        elif "GMOS_DARK" in data_types:
            id_descriptor_list = dark_id
        elif ("GMOS_IMAGE_FLAT" in data_types or
              "GMOS_IMAGE_TWILIGHT" in data_types):
            id_descriptor_list = flat_twilight_id
        else:
            id_descriptor_list = science_id

        # Add in all of the common descriptors required
        id_descriptor_list.extend(unique_id_descriptor_list_all)

        # Form the group_id
        descriptor_dv_string_list = []
        for descriptor in id_descriptor_list:
            # Prepare the descriptor call
            if descriptor in call_pretty_version_list:
                end_parameter = "(pretty=True)"
            else:
                end_parameter = "()"
            descriptor_call = ''.join([descriptor, end_parameter])

            # Call the descriptor
            exec ("descriptor_dv = dataset.{0}".format(descriptor_call))

            # Check for a returned descriptor value object with a None value
            if descriptor_dv.is_none():
                # The descriptor functions return None if a value cannot be found
                # and stores the exception info. Re-raise the exception. It
                # will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info

            # In some cases require the information as a list
            if descriptor in convert_to_list_list:
                descriptor_dv = descriptor_dv.as_list()

            # Convert DV value to a string and store
            descriptor_dv_string_list.append(str(descriptor_dv))

        # Create the final group_id string
        ret_group_id = '_'.join(descriptor_dv_string_list)            

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_group_id, name="group_id", ad=dataset)
        
        return ret_dv

    def nod_count(self, dataset, **args):
        # The number of nod and shuffle cycles can only be obtained from nod
        # and shuffle data
        if "GMOS_NODANDSHUFFLE" in dataset.types:
            
            # Determine the number of nod and shuffle cycles keyword from the
            # global keyword dictionary, for positions A and B
            keyword1 = self.get_descriptor_key("key_a_nod_count")
            keyword2 = self.get_descriptor_key("key_b_nod_count")
            
            # Get the value of the number of nod and shuffle cycles keyword
            # from the header of the PHU, for positions A and N
            a_nod_count = dataset.phu_get_key_value(keyword1)
            b_nod_count = dataset.phu_get_key_value(keyword2)

            if a_nod_count is None or b_nod_count is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Return the nod count integers as a tuple
            ret_nod_count = (int(a_nod_count),int(b_nod_count))
        else:
            raise Errors.DescriptorTypeError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nod_count, name="nod_count", ad=dataset)
        
        return ret_dv
    
    def nod_offsets(self, dataset, **args):
        # Get the offsets (in arcsec) from the default telescope position
        # For the observations in beams A and B
        if "GMOS_NODANDSHUFFLE" in dataset.types:
            # These are the modern keywords
            ayoff = dataset.phu_get_key_value("NODAYOFF")
            byoff = dataset.phu_get_key_value("NODBYOFF")
            
            # But earlier, it was assumed that only B would be offset
            if ayoff is None:
                ayoff = 0.0
                byoff = dataset.phu_get_key_value("NODYOFF")

            if byoff is None:
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info

            # Return a tuple
            ret_nod_offsets = (ayoff,byoff)
        else:
            raise Errors.DescriptorTypeError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nod_pixels, name="nod_offsets", ad=dataset)
        
        return ret_dv
    
    def nod_pixels(self, dataset, **args):
        # The number of pixel rows the charge is shuffled by can only be
        # obtained from nod and shuffle data
        if "GMOS_NODANDSHUFFLE" in dataset.types:
            
            # Determine the number of pixel rows the charge is shuffled by
            # keyword from the global keyword dictionary
            keyword = self.get_descriptor_key("key_nod_pixels")
            
            # Get the value of the number of pixel rows the charge is shuffled
            # by keyword from the header of the PHU
            nod_pixels = dataset.phu_get_key_value(keyword)
            
            if nod_pixels is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Return the nod pixels integer
            ret_nod_pixels = int(nod_pixels)
        else:
            raise Errors.DescriptorTypeError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nod_pixels, name="nod_pixels", ad=dataset)
        
        return ret_dv
    
    def nominal_photometric_zeropoint(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_nominal_photometric_zeropoint_dict = {}
        
        # Get the lookup table containing the nominal zeropoints
        table = Nominal_Zeropoints.nominal_zeropoints
        
        # Get the values of the gain, detector name and filter name using the
        # appropriate descriptors.
        gain_dv = dataset.gain()
        array_name_dv = dataset.array_name()
        filter_name_dv = dataset.filter_name(pretty=True)
        
        if (gain_dv.is_none() or array_name_dv.is_none() or
            filter_name_dv.is_none()):
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the value of the BUNIT keyword from the header of each pixel data
        # extension as a dictionary where the key of the dictionary is an
        # ("*", EXTVER) tuple 
        bunit_dict = gmu.get_key_value_dict(adinput=dataset, keyword="BUNIT")

        # Loop over extvers in one of the DVs as they will all have the same
        # extvers. Then use get_value method to get the value from a given DV
        # for a particular extver
        for extver in gain_dv.ext_vers():
            # key used for bunit dict and for instantiating outuput DV
            ext_name_ver = ("*", extver)
            
            # Obtain the values from the three DVs for the current extver
            array_name = array_name_dv.get_value(extver)
            gain = gain_dv.get_value(extver)
            filter_name = filter_name_dv.get_value(extver)

            if array_name is None:
                nominal_photometric_zeropoint = None
            else:
                # Determine whether data are in ADU or electrons
                if bunit_dict is not None:
                    bunit = bunit_dict[ext_name_ver]
                else:
                    bunit = None
                
                # If bunit is "electron" or None, set the gain factor to 0.0 
                gain_factor = 0.0
                if bunit == "adu":
                    gain_factor = 2.5 * math.log10(gain)

                nominal_zeropoint_key = (array_name, filter_name)
                
                if nominal_zeropoint_key in table:
                    nominal_photometric_zeropoint = (
                        table[nominal_zeropoint_key] - gain_factor)
                else:
                    raise Errors.TableKeyError()
            
            # Update the dictionary with the nominal photometric zeropoint
            # value 
            ret_nominal_photometric_zeropoint_dict.update({
                ext_name_ver: nominal_photometric_zeropoint})

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nominal_photometric_zeropoint_dict,
                                 name="nominal_photometric_zeropoint",
                                 ad=dataset)
        return ret_dv
    
    def non_linear_level(self, dataset, **args):
        # Set the non linear level equal to the saturation level for GMOS
        non_linear_level_dv = dataset.saturation_level()
        
        if non_linear_level_dv.is_none():
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ret_non_linear_level = non_linear_level_dv
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_non_linear_level, name="non_linear_level",
                                 ad=dataset)
        return ret_dv
    
    def overscan_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_overscan_section_dict = {}
        
        # Determine the overscan section keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_overscan_section")
        
        # Get the value of the overscan section keyword from the header of each
        # pixel data extension as a dictionary where the key of the dictionary
        # is an ("*", EXTVER) tuple
        overscan_section_dict = gmu.get_key_value_dict(adinput=dataset,
                                                       keyword=keyword)
        if overscan_section_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        dict = overscan_section_dict.iteritems()
        for ext_name_ver, raw_overscan_section in dict:
            if raw_overscan_section is None:
                overscan_section = None
            elif pretty:
                # Use the overscan section string that uses 1-based indexing as
                # the value in the form [x1:x2,y1:y2]
                overscan_section = str(raw_overscan_section)
            else:
                # Use the overscan section list that uses 0-based,
                # non-inclusive indexing as the value in the form
                # [x1, x2, y1, y2] 
                overscan_section = gmu.sectionStrToIntList(
                    raw_overscan_section)
            
            # Update the dictionary with the array section value
            ret_overscan_section_dict.update({ext_name_ver:overscan_section})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_overscan_section_dict,
                                 name="overscan_section") #, ad=dataset)
        return ret_dv
    
    def pixel_scale(self, dataset, **args):
        # Get the lookup table containing the pixel scale values
        gmosPixelScales = GMOSPixelScale.gmosPixelScales
        
        # Get the values of the instrument and the binning of the y-axis using
        # the appropriate descriptors
        instrument_dv = dataset.instrument()
        detector_y_bin_dv = dataset.detector_y_bin()
        
        # Determine the detector type keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_detector_type")
        
        # Get the value for the detector type keyword from the header of the
        # PHU
        detector_type = dataset.phu_get_key_value(keyword)
        
        if (instrument_dv.is_none() or detector_y_bin_dv.is_none() or
            detector_type is None):
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Use as_pytype() to return the instrument value as the default python
        # type, rather than an object
        instrument = instrument_dv.as_pytype()
        
        # Get the unbinned pixel scale (in arcseconds per unbinned pixel) from
        # the lookup table 
        pixel_scale_key = (instrument, detector_type)
        
        if pixel_scale_key in gmosPixelScales:
            raw_pixel_scale = gmosPixelScales[pixel_scale_key]
        else:
            raise Errors.TableKeyError()
        
        # Return the binned pixel scale value
        ret_pixel_scale = float(detector_y_bin_dv * raw_pixel_scale)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    def read_mode(self, dataset, **args):
        # There are currently four ways to set the read mode for the GMOS
        # instruments' detector, which are determined by the gain setting and
        # the read speed setting.

        # Read mode name mapping: 
        # key=read_mode; value= [ gain_setting, read_speed_setting ]
        #
        # mapping_dict moved to lookups
        # As there was no separate definitions for EEV and e2v CCDs,
        # 'default' applies for both EEV and the super old e2v detector
        # names.
        # 04-06-2015, kra
        read_mode_table = GMOSReadModes.read_mode_map
        detector = dataset.detector_name(pretty=True).as_pytype()

        if detector == 'Hamamatsu':
            read_mode_mapping_dict = read_mode_table[detector]
        else:
            read_mode_mapping_dict = read_mode_table['default']

        # Required descriptors
        read_mode_descriptors = ["gain_setting", "read_speed_setting"]
        read_mode_dvs = []
        for descriptor_name in read_mode_descriptors:
            exec ("read_mode_dvs.append(dataset.{0}())".format(descriptor_name))

        if any(dv.is_none() for dv in read_mode_dvs):
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        # Obtain the read mode from the look up table.
        read_mode_dvs_strings = [str(dv) for dv in read_mode_dvs]
        for key, value in read_mode_mapping_dict.iteritems():
            if read_mode_dvs_strings == value:
                read_mode_string = key
                break
        else:
            raise Errors.TableKeyError("{0!s} is not a valid value in the {2} "
                                       "dictionary: {3!r}".format(
                    read_mode_dvs_strings, "read_mode_mapping_dict",
                    read_mode_mapping_dict))

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(read_mode_string, name="read_mode",
                                 ad=dataset)
        return ret_dv
    
    def read_noise(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_read_noise_dict = {}
        
        # If the data have been prepared, take the read noise value directly
        # from the appropriate keyword. At some point, a check for the STDHDRSI
        # header keyword should be added, since the function that overwrites
        # the read noise keyword also writes the STDHDRSI keyword.
        if "PREPARED" in dataset.types:
            
            # Determine the read noise keyword from the global keyword
            # dictionary
            keyword = self.get_descriptor_key("key_read_noise")
            
            # Get the value of the read noise keyword from the header of each
            # pixel data extension as a dictionary where the key of the
            # dictionary is an ("*", EXTVER) tuple
            read_noise_dict = gmu.get_key_value_dict(adinput=dataset,
                                                     keyword=keyword)
            if read_noise_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            for ext_name_ver, raw_read_noise in read_noise_dict.iteritems():
                if raw_read_noise is None:
                    read_noise = None
                else:
                    read_noise = float(raw_read_noise)
                
                # Update the dictionary with the read noise value
                ret_read_noise_dict.update({ext_name_ver:read_noise})
        else:
            # Get the lookup table containing the read noise values by
            # amplifier
            gmosampsRdnoise = GMOSAmpTables.gmosampsRdnoise
            gmosampsRdnoiseBefore20150826 = GMOSAmpTables.gmosampsRdnoiseBefore20150826
            gmosampsRdnoiseBefore20060831 = GMOSAmpTables.gmosampsRdnoiseBefore20060831
            
            # Get the UT date, gain setting and read speed setting values using
            # the appropriate descriptors
            ut_date_dv = dataset.ut_date()
            gain_setting_dv = dataset.gain_setting()
            read_speed_setting_dv = dataset.read_speed_setting()
            
            if (ut_date_dv.is_none() or gain_setting_dv.is_none() or
                read_speed_setting_dv.is_none()):
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Use as_dict() and as_pytype() to return the values as a
            # dictionary and the default python type, respectively, rather than
            # an object
            ut_date = str(ut_date_dv)
            gain_setting_dict = gain_setting_dv.as_dict()
            read_speed_setting = read_speed_setting_dv.as_pytype()
            
            obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
            change_2015_ut = datetime(2015, 8, 26, 0, 0)
            change_2006_ut = datetime(2006, 8, 31, 0, 0)
            # Determine the name of the detector amplifier keyword (ampname)
            # from the global keyword dictionary 
            keyword = self.get_descriptor_key("key_ampname")
            
            # Get the value of the name of the detector amplifier keyword from
            # the header of each pixel data extension as a dictionary where the
            # key of the dictionary is an ("*", EXTVER) tuple
            ampname_dict = gmu.get_key_value_dict(adinput=dataset,
                                                  keyword=keyword)
            if ampname_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            for ext_name_ver, ampname in ampname_dict.iteritems():
                gain_setting = gain_setting_dict[ext_name_ver]
                
                if ampname is None or gain_setting is None:
                    read_noise = None
                else:
                    read_noise_key = (
                        read_speed_setting, gain_setting, ampname)

                    if obs_ut_date > change_2015_ut:
                        read_noise_dict = gmosampsRdnoise
                    elif obs_ut_date > change_2006_ut:
                        read_noise_dict = gmosampsRdnoiseBefore20150826
                    else:
                        read_noise_dict = gmosampsRdnoiseBefore20060831

                    if read_noise_key in read_noise_dict:
                        read_noise = read_noise_dict[read_noise_key]
                    else:
                        raise Errors.TableKeyError()
                
                # Update the dictionary with the read noise value
                ret_read_noise_dict.update({ext_name_ver:read_noise})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_noise_dict, name="read_noise",
                                 ad=dataset)
        return ret_dv
    
    def read_speed_setting(self, dataset, **args):
        # Determine the amplifier integration time keyword (ampinteg) from the
        # global keyword dictionary
        keyword = self.get_descriptor_key("key_ampinteg")
        ampinteg = dataset.phu_get_key_value(keyword)

        if ampinteg is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

       # ampinteg depends on CCD type (EEV, Hamamatsu)
        detector = dataset.detector_name(pretty=True)
        if detector == "Hamamatsu":
            ret_read_speed_setting = "slow" if ampinteg > 8000 else "fast"
        else:
            ret_read_speed_setting = "slow" if ampinteg > 2000 else "fast"

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_speed_setting,
                                 name="read_speed_setting", ad=dataset)
        return ret_dv

    def saturation_level(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_saturation_level_dict = {}
        
        # Get the lookup table containing the saturation values by amplifier
        gmosThresholds = GMOSThresholdValues.gmosThresholds

        # The hard limit for saturation is the controller digitization limit
        controller_limit = 65535
        
        # Determine the name of the bias image and the name of the dark image
        # keywords from the global keyword dictionary 
        keyword1 = self.get_descriptor_key("key_bias_image")
        keyword2 = self.get_descriptor_key("key_dark_image")
        
        # Get the value of the the name of the bias image and the name of the
        # dark image keywords from the header of the PHU 
        bias_image = dataset.phu_get_key_value(keyword1)
        dark_image = dataset.phu_get_key_value(keyword2)
        
        # Determine the name of the detector amplifier (ampname) and the
        # overscan value keywords from the global keyword dictionary
        keyword3 = self.get_descriptor_key("key_ampname")
        keyword4 = self.get_descriptor_key("key_overscan_value")
        
        # Get the value of the name of the detector amplifier, the overscan
        # value and the BUNIT keywords from the header of each pixel data
        # extension as a dictionary, where the key of the dictionary is an
        # ("*", EXTVER) tuple
        keyword_dict = gmu.get_key_value_dict(
            adinput=dataset, keyword=[keyword3, keyword4, "BUNIT"],
            dict_key_extver=True)
        
        ampname_dict = keyword_dict[keyword3]
        
        if ampname_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        overscan_dict = keyword_dict[keyword4]
        bunit_dict = keyword_dict["BUNIT"]
        
        # Get the name of the detector, the gain and the binning of the x-axis
        # and y-axis values using the appropriate descriptors
        detector_name_dv = dataset.detector_name(pretty=True)

        gain_dv = dataset.gain()
        detector_x_bin_dv = dataset.detector_x_bin()
        detector_y_bin_dv = dataset.detector_y_bin()
        
        if (detector_name_dv.is_none() or gain_dv.is_none() or
            detector_x_bin_dv.is_none() or detector_y_bin_dv.is_none()):
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Determine the bin factor. If the bin factor is great than 2, the
        # saturation level will be equal to the controller digitization limit.
        bin_factor = detector_x_bin_dv * detector_y_bin_dv

        for extver, ampname in ampname_dict.iteritems():
            gain = gain_dv.get_value(extver=extver)
            
            # Determine whether it is required to calculate the bias level
            # (bias level calculations can take some time)
            overscan = None
            if overscan_dict is not None:
                overscan = overscan_dict[extver]
            if overscan is not None:
                # The overscan was subtracted from the data
                data_contains_bias = False
            elif bias_image is not None or dark_image is not None:
                # The overscan was not subtracted from the data, but the bias
                # or dark was subtracted from the data
                data_contains_bias = False
            else:
                # The data still contains a bias level
                data_contains_bias = True
            
            if ((not data_contains_bias) or
                (not detector_name_dv == "EEV" and data_contains_bias and
                 bin_factor <= 2)):
                # Calculate the bias level
                bias_level = gdc.get_bias_level(adinput=dataset, estimate=True)
            # Correct the controller limit for bias level and units
            processed_limit = controller_limit
            if not data_contains_bias:
                processed_limit -= bias_level[extver]
            
            # Check units of data (i.e., ADU vs. electrons)
            bunit = None
            if bunit_dict is not None:
                bunit = bunit_dict[extver]
            if bunit == "electron" or bunit == "electrons":
                processed_limit *= gain
            
            if detector_name_dv == "EEV" or bin_factor > 2:
                # For old EEV CCDs, use the detector limit
                saturation = processed_limit
            else:
                # For new GMOS-N CCDs, look up the saturation limit from
                # the table, then correct for binning, units, and bias level
                
                # Get the base saturation value from the lookup table
                if ampname in gmosThresholds:
                    # Saturation value from the lookup table is an integer 
                    # with units of electrons
                    saturation = gmosThresholds[ampname]
                else:
                    # This error condition will be hit for all mosaicked
                    # or tiled data
                    errmsg = ("'{0}' key not found "
                              "in '{1}'".format(ampname, "gmosThresholds"))
                    raise Errors.TableKeyError(errmsg)
                
                # Correct the saturation level for binning
                saturation = saturation * bin_factor
                
                # The saturation level is reported in electrons; convert
                # it to ADU if needed
                if bunit not in ("electron", "electrons"):
                    saturation = saturation / gain
                
                # The saturation level does not contain the bias; add it
                # in if necessary
                if data_contains_bias:
                    saturation += bias_level[extver]
                
                # Check whether the value is now over the controller limit;
                # if so, set it to the hard limit
                if saturation > processed_limit:
                    saturation = processed_limit
            
            ret_saturation_level_dict.update({extver: saturation})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_saturation_level_dict,
                                 name="saturation_level", ad=dataset)
        return ret_dv
    
    def wavelength_band(self, dataset, **args):
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = GMOSFilterWavelength.filter_wavelength
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
        # Return the RA derived from the WCS
        # This GMOS version simply returns the reference value for each coordinate
        # from the WCS. This is simplistic, but very robust and is good for GMOS
        # Try the first science extension
        crval = dataset['SCI', 1].get_key_value('CRVAL1')
        ctype = dataset['SCI', 1].get_key_value('CTYPE1')

        # If None, try the PHU
        crval = dataset.phu_get_key_value('CRVAL1') if crval is None else crval
        ctype = dataset.phu_get_key_value('CTYPE1') if ctype is None else ctype

        if crval is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if ctype != 'RA---TAN':
            raise Errors.InvalidValueError

        return DescriptorValue(crval, name="wcs_ra", ad=dataset)

    def wcs_dec(self, dataset, **args):
        # Return the DEC derived from the WCS
        # This generic version simply returns ther REFERENCE VALUE
        # from the WCS. This is simplistic, but very robust and is good for GMOS
        # Try the first science extension
        crval = dataset['SCI', 1].get_key_value('CRVAL2')
        ctype = dataset['SCI', 1].get_key_value('CTYPE2')

        # If None, try the PHU
        crval = dataset.phu_get_key_value('CRVAL2') if crval is None else crval
        ctype = dataset.phu_get_key_value('CTYPE2') if ctype is None else ctype

        if crval is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if ctype != 'DEC--TAN':
            raise Errors.InvalidValueError

        return DescriptorValue(crval, name="wcs_dec", ad=dataset)

