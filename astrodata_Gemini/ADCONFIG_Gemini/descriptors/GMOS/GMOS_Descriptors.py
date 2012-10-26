from datetime import datetime
from time import strptime
import math
import numpy as np

from astrodata import Errors
from astrodata import Lookups
from gempy.gemini import gemini_metadata_utils as gmu
import GemCalcUtil

from GMOS_Keywords import GMOS_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GMOS_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def amp_read_area(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_amp_read_area = {}
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            # Determine the name of the detector amplifier keyword (ampname)
            # from the global keyword dictionary
            keyword = self.get_descriptor_key("key_ampname")
            
            # Get the value of the name of the detector amplifier keyword from
            # the header of each pixel data extension 
            ampname = ext.get_key_value(keyword)
            
            # Get the pretty (1-based indexing) readout area of the CCD
            # (detsec) using the appropriate descriptor
            detsec = ext.detector_section(pretty=True)
            
            if ampname is None or detsec is None:
                amp_read_area = None
            else:
                # Use the composite amp_read_area string as the value
                amp_read_area = "'%s':%s" % (ampname, detsec)
            
            # Update the dictionary with the amp_read_area value
            ret_amp_read_area.update({
                (ext.extname(), ext.extver()):amp_read_area})
        
        if ret_amp_read_area == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_amp_read_area
    
    def bias_level(self, dataset, **args):
        # Get the static bias lookup table
        gmosampsBias, gmosampsBiasBefore20060831 = Lookups.get_lookup_table(
            "Gemini/GMOS/GMOSAmpTables", "gmosampsBias",
            "gmosampsBiasBefore20060831")
        
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_bias_level = {}
        
        # Get the UT date using the appropriate descriptor
        ut_date = str(dataset.ut_date())
        
        if ut_date is None:
            # The descriptor functions return None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)
        
        # Get the gain setting and read speed setting values using the
        # appropriate descriptors.
        read_speed_setting_dv = dataset.read_speed_setting()
        gain_setting_dv = dataset.gain_setting()
        if read_speed_setting_dv is None or gain_setting_dv is None:
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        else:
            read_speed_setting_dict = read_speed_setting_dv.dict_val
            gain_setting_dict = gain_setting_dv.dict_val
        
        # Get the gain dictionary for the dataset
        gain_dv = dataset.gain()
        if gain_dv is None:
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        else:
            gain_dict = gain_dv.dict_val
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            dict_key = (ext.extname(),ext.extver())
            
            # Determine the name of the detector amplifier keyword (ampname)
            # from the global keyword dictionary
            keyword = self.get_descriptor_key("key_ampname")
            
            # Get the value of the name of the detector amplifier keyword from
            # the header of each pixel data extension 
            ampname = ext.get_key_value(keyword)
            
            if ampname is None:
                bias_level = None
            
            # Check whether data has been overscan-subtracted
            overscan = ext.get_key_value(
                self.get_descriptor_key("key_overscan_value"))
            
            # Check whether a raw bias was written to the header
            # (eg. in the prepare step)
            raw_bias = ext.get_key_value(
                self.get_descriptor_key("key_bias_level"))
            
            # Get the gain, gain_setting, and read_speed_setting for the
            # extension
            gain = gain_dict[dict_key]
            gain_setting = gain_setting_dict[dict_key]
            read_speed_setting = read_speed_setting_dict[dict_key]
            
            # Get the approximate bias level for the extension
            if overscan is not None:
                # Use the average overscan level as the bias level
                bias_level = overscan
            elif raw_bias is not None:
                # Use the previously calculated bias level
                # (written in prepare step)
                bias_level = raw_bias
            else:
                # Otherwise, use static bias levels
                # from the lookup table
                bias_key = (read_speed_setting, gain_setting, ampname)
                if obs_ut_date > old_ut_date:
                    if bias_key in gmosampsBias:
                        static_bias = gmosampsBias[bias_key]
                    else:
                        raise Errors.TableKeyError()
                else:
                    if bias_key in gmosampsBiasBefore20060831:
                        static_bias = gmosampsBiasBefore20060831[bias_key]
                    else:
                        raise Errors.TableKeyError()
                
                bias_level = static_bias
            
            ret_bias_level.update({(ext.extname(), ext.extver()): bias_level})
        
        if ret_bias_level == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_bias_level
    
    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):
        # Currently for GMOS data, the central wavelength is recorded in
        # nanometers
        input_units = "nanometers"
        
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
        
        # Determine the central wavelength keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_central_wavelength")
        
        # Get the value of the central wavelength keyword from the header of
        # the PHU
        raw_central_wavelength = dataset.phu_get_key_value(keyword)
        
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
        
        return ret_central_wavelength
    
    def detector_name(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_name = {}
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # Determine the name of the detector keyword (ccdname) from the
            # global keyword dictionary 
            keyword = self.get_descriptor_key("key_detector_name")
            
            # Get the value of the name of the detector keyword from the header
            # of each pixel data extension
            raw_detector_name = ext.get_key_value(keyword)
            
            if raw_detector_name is None:
                # It is possible that the data have been mosaiced, which means
                # that the ccdname keyword no longer exists in the pixel data
                # extensions. Instead, determine the detector ID keyword from
                # the global keyword dictionary
                keyword = self.get_descriptor_key("key_phu_detector_name")
                
                # Get the value of the detector ID keyword from the header of
                # the PHU
                raw_detector_name = dataset.phu_get_key_value(keyword)
                
                if raw_detector_name is None:
                    detector_name = None
                else:
                    # Use the detector name string as the value
                    detector_name = str(raw_detector_name)
            else:
                # Use the detector name string as the value
                detector_name = str(raw_detector_name)
            
            # Update the dictionary with the detector name value
            ret_detector_name.update({
                (ext.extname(), ext.extver()):detector_name})
        
        if ret_detector_name == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_detector_name
    
    def detector_rois_requested(self, dataset, **args):
        # This parses the DETROx GMOS headers and returns a list of ROIs in the
        # form [x1, x2, y1, y2]. These are in physical pixels - should be
        # irrespective of binning. These are 1-based, and inclusive, 2 pixels,
        # starting at pixel 2 would be [2, 3].
        rois=[]
        
        # Must be single digit ROI number
        for i in range(1,10):
            x1 = dataset.phu_get_key_value('DETRO'+str(i)+'X')
            xs = dataset.phu_get_key_value('DETRO'+str(i)+'XS')
            y1 = dataset.phu_get_key_value('DETRO'+str(i)+'Y')
            ys = dataset.phu_get_key_value('DETRO'+str(i)+'YS')
            if x1 is not None:
                # The headers are in the form of a start position and size
                # so make them into start and end pixels here.
                xs *= int(dataset.detector_x_bin())
                ys *= int(dataset.detector_y_bin())
                rois.append([x1, x1+xs-1, y1, y1+ys-1])
            else:
                break
        return rois
    
    def detector_roi_setting(self, dataset, **args):
        # Attempts to deduce the Name of the ROI, more or less as per
        # the options you can select in the OT.
        # Only considers the first ROI.
        gmosRoiSettings = Lookups.get_lookup_table("Gemini/GMOS/ROItable",
                                                   "gmosRoiSettings")
        
        roi_setting = "Undefined"
        rois = dataset.detector_rois_requested().as_list()
        if rois:
            roi = rois[0]
            
            # If we don't recognise it, it's "Custom"
            roi_setting = "Custom"
            
            for s in gmosRoiSettings.keys():
                if roi in gmosRoiSettings[s]:
                    roi_setting = s
        
        return roi_setting
    
    def detector_x_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_x_bin = {}
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # Determine the ccdsum keyword from the global keyword dictionary
            keyword = self.get_descriptor_key("key_ccdsum")
            
            # Get the value of the ccdsum keyword from the header of each pixel
            # data extension
            ccdsum = ext.get_key_value(keyword)
            
            if ccdsum is None:
                detector_x_bin = None
            else:
                # Use the binning of the x-axis integer as the value
                detector_x_bin, detector_y_bin = ccdsum.split()
            
            # Update the dictionary with the binning of the x-axis value
            ret_detector_x_bin.update({
                (ext.extname(), ext.extver()):detector_x_bin})
        
        if ret_detector_x_bin == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_y_bin = {}
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # Determine the ccdsum keyword from the global keyword dictionary
            keyword = self.get_descriptor_key("key_ccdsum")
            
            # Get the value of the ccdsum keyword from the header of each pixel
            # data extension
            ccdsum = ext.get_key_value(keyword)
            
            if ccdsum is None:
                detector_y_bin = None
            else:
                # Use the binning of the y-axis integer as the value
                detector_x_bin, detector_y_bin = ccdsum.split()
            
            # Update the dictionary with the binning of the y-axis value
            ret_detector_y_bin.update({
                (ext.extname(), ext.extver()):detector_y_bin})
        
        if ret_detector_y_bin == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_detector_y_bin
    
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
        
        return ret_disperser
    
    def dispersion(self, dataset, asMicrometers=False, asNanometers=False,
                   asAngstroms=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_dispersion = {}
        
        # Currently for GMOS data, the dispersion is recorded in meters (?)
        input_units = "meters"
        
        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # dispersion in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the dispersion in the default units of meters
            output_units = "meters"
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # Determine the dispersion keyword from the global keyword
            # dictionary
            keyword = self.get_descriptor_key("key_dispersion")
            
            # Get the value of the dispersion keyword from the header of each
            # pixel data extension
            raw_dispersion = ext.get_key_value(keyword)
            
            if raw_dispersion is None:
                dispersion = None
            else:
                # Use the utilities function convert_units to convert the
                # dispersion value from the input units to the output units
                dispersion = float(GemCalcUtil.convert_units(
                    input_units=input_units, input_value=float(raw_dispersion),
                    output_units=output_units))
            
            # Update the dictionary with the dispersion value
            ret_dispersion.update({(ext.extname(), ext.extver()):dispersion})
        
        if ret_dispersion == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_dispersion
    
    def dispersion_axis(self, dataset, **args):
        # The GMOS dispersion axis should always be 1
        ret_dispersion_axis = 1
        
        return ret_dispersion_axis
    
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
        
        return ret_exposure_time
    
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
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_gain = {}
        
        # If the data have been prepared, take the gain value directly from the
        # appropriate keyword. At some point, a check for the STDHDRSI header
        # keyword should be added, since the function that overwrites the gain
        # keyword also writes the STDHDRSI keyword
        if "PREPARED" in dataset.types:
            
            # Loop over the pixel data extensions in the dataset
            for ext in dataset:
                
                # Determine the gain keyword from the global keyword dictionary
                keyword = self.get_descriptor_key("key_gain")
                
                # Get the value of the gain keyword from the header of each
                # pixel data extension
                raw_gain = ext.get_key_value(keyword)
                
                if raw_gain is None:
                    gain = None
                else:
                    # Use the gain float as the value
                    gain = float(raw_gain)
                
                # Update the dictionary with the gain value
                ret_gain.update({(ext.extname(), ext.extver()):gain})
        else:
            # Get lookup table for GMOS gains by amp
            gmosampsGain, gmosampsGainBefore20060831 = \
                Lookups.get_lookup_table("Gemini/GMOS/GMOSAmpTables",
                                         "gmosampsGain",
                                         "gmosampsGainBefore20060831")
            
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
            
            # Get the UT date using the appropriate descriptor
            ut_date = str(dataset.ut_date())
            
            if ut_date is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
            old_ut_date = datetime(2006, 8, 31, 0, 0)
            
            # Get the gain setting and read speed setting values using the
            # appropriate descriptors.
            read_speed_setting_dv = dataset.read_speed_setting()
            gain_setting_dv = dataset.gain_setting()
            
            if read_speed_setting_dv is None or gain_setting_dv is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the
                # CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Loop over the pixel data extensions in the dataset
            for ext in dataset:
                
                # Determine the name of the detector amplifier keyword
                # (ampname) from the global keyword dictionary
                keyword = self.get_descriptor_key("key_ampname")
                
                # Get the value of the name of the detector amplifier keyword
                # from the header of each pixel data extension
                ampname = ext.get_key_value(keyword)
                
                if ampname is None:
                    gain = None
                else:
                    # Get the gain setting and read speed setting values using
                    # the dictionary created above
                    dict_key = (ext.extname(),ext.extver())
                    read_speed_setting = read_speed_setting_dv[dict_key]
                    gain_setting = gain_setting_dv[dict_key]
                    
                    gain_key = (read_speed_setting, gain_setting, ampname)
                    if obs_ut_date > old_ut_date:
                        if gain_key in gmosampsGain:
                            gain = gmosampsGain[gain_key]
                        else:
                            raise Errors.TableKeyError()
                    else:
                        if gain_key in gmosampsGainBefore20060831:
                            gain = gmosampsGainBefore20060831[gain_key]
                        else:
                            raise Errors.TableKeyError()
                
                # Return a dictionary with the gain float as the value
                ret_gain.update({(ext.extname(), ext.extver()):gain})
        
        if ret_gain == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_gain
    
    def gain_setting(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_gain_setting = {}
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # If the data have not been prepared, take the raw gain value
            # directly from the appropriate keyword.
            if "PREPARED" not in dataset.types:
                
                # Determine the gain keyword from the global keyword dictionary
                keyword = self.get_descriptor_key("key_gain")
                
                # Get the value of the gain keyword from the header of each
                # pixel data extension
                gain = ext.get_key_value(keyword)
                
                if gain is None:
                    gain_setting = None
                elif gain > 3.0:
                    gain_setting = "high"
                else:
                    gain_setting = "low"
            
            else:
                # If the data have been prepared, take the gain setting value
                # directly from the appropriate keyword in the header of each
                # pixel data extension
                gain_setting = ext.get_key_value(
                    self.get_descriptor_key("key_gain_setting"))
                
                if gain_setting is None:
                    # The dataset was not processed using gemini_python. Try to
                    # get the gain from the "GAINORIG" keyword in the header of
                    # the PHU
                    gain = ext.get_key_value("GAINORIG")
                    
                    if gain is None or gain == 1:
                        # If gain is equal to 1, it means that the very
                        # original gain was written to "GAINMULT". Try to get
                        # the gain from the "GAINMULT" keyword in the header of
                        # the PHU
                        gain = ext.get_key_value("GAINMULT")
                        
                        if gain is None:
                            # Resort to getting the gain using the appropriate
                            # descriptor ... this will use the updated gain
                            # value
                            gain = ext.gain()
                            
                            if gain is None:
                                gain_setting = None
                    
                    if gain > 3.0:
                        gain_setting = "high"
                    else:
                        gain_setting = "low"
            
            # Return a dictionary with the gain setting string as the value
            ret_gain_setting.update({
                (ext.extname(), ext.extver()):gain_setting})
        
        if ret_gain_setting == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_gain_setting
    
    def group_id(self, dataset, **args):
        # For GMOS data, the group id contains the detector_x_bin,
        # detector_y_bin and amp_read_area in addition to the observation id.
        # Get the observation id, the binning of the x-axis and y-axis and the
        # amp_read_area values using the appropriate descriptors
        observation_id = dataset.observation_id()
        detector_x_bin = dataset.detector_x_bin()
        detector_y_bin = dataset.detector_y_bin()
        
        # Return the amp_read_area as an ordered list
        amp_read_area = dataset.amp_read_area().as_list()
        
        if observation_id is None or detector_x_bin is None or \
           detector_x_bin is None or amp_read_area is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # For all data other than data with an AstroData type of GMOS_BIAS, the
        # group id contains the filter_name. Also, for data with an AstroData
        # type of GMOS_BIAS and GMOS_IMAGE_FLAT, the group id does not contain
        # the observation id. 
        if "GMOS_BIAS" in dataset.types:
            ret_group_id = "%s_%s_%s" % (detector_x_bin, detector_y_bin,
                                         amp_read_area)
        else:
            # Get the filter name using the appropriate descriptor
            filter_name = dataset.filter_name(pretty=True)
            
            if filter_name is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            if "GMOS_IMAGE_FLAT" in dataset.types:
                ret_group_id = "%s_%s_%s_%s" % (detector_x_bin, detector_y_bin,
                                                filter_name, amp_read_area)
            else:
                ret_group_id = "%s_%s_%s_%s_%s" % (observation_id,
                                                   detector_x_bin,
                                                   detector_y_bin, filter_name,
                                                   amp_read_area)
        
        return ret_group_id
    
    def nod_count(self, dataset, **args):
        # The number of nod and shuffle cycles can only be obtained from nod
        # and shuffle data
        if "GMOS_NODANDSHUFFLE" in dataset.types:
            
            # Determine the number of nod and shuffle cycles keyword from the
            # global keyword dictionary
            keyword = self.get_descriptor_key("key_nod_count")
            
            # Get the value of the number of nod and shuffle cycles keyword
            # from the header of the PHU
            nod_count = dataset.phu_get_key_value(keyword)
            
            if nod_count is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Return the nod count integer
            ret_nod_count = int(nod_count)
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_nod_count
    
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
        
        return ret_nod_pixels
    
    def nominal_photometric_zeropoint(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_nominal_photometric_zeropoint = {}
        
        table = Lookups.get_lookup_table("Gemini/GMOS/Nominal_Zeropoints",
                                         "nominal_zeropoints")
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # Get the values of the filter name and detector name using the
            # appropriate descriptors. Use as_pytype() to return the values as
            # the default python type, rather than an object.
            filter_name = ext.filter_name(pretty=True).as_pytype()
            detector_name = ext.detector_name().as_pytype()
            
            if filter_name is None or detector_name is None:
                nominal_photometric_zeropoint = None
            else:
                # Determine whether data are in ADU or electrons
                bunit = ext.get_key_value("BUNIT")
                
                # If bunit is "electron" or None, set the gain factor to 0.0
                gain_factor = 0.0
                
                if bunit == "adu":
                    gain_factor = 2.5*math.log10(float(ext.gain()))
                
                nominal_zeropoint_key = (detector_name, filter_name)
                
                if key in table:
                    nominal_zeropoint = (table[nominal_zeropoint_key] -
                                         gain_factor)
                else:
                    raise Errors.TableKeyError()
            
            # Update the dictionary with the nominal photometric zeropoint
            # value 
            ret_nominal_photometric_zeropoint.update({(
                ext.extname(), ext.extver()) : nominal_zeropoint})
        
        if ret_nominal_photometric_zeropoint == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_nominal_photometric_zeropoint
    
    def non_linear_level(self, dataset, **args):
        # Set the non linear level equal to the saturation level for GMOS
        ret_non_linear_level = dataset.saturation_level()
        
        if ret_non_linear_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        return ret_non_linear_level
    
    def overscan_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_overscan_section = {}
        
        # Loop over the pixel data extensions in the dataset
        for ext in dataset:
            
            # Determine the overscan section keyword from the global keyword
            # dictionary
            keyword = self.get_descriptor_key("key_overscan_section")
            
            # Get the value of the overscan section keyword from the header of
            # each pixel data extension
            raw_overscan_section = ext.get_key_value(keyword)
            
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
            ret_overscan_section.update({
                (ext.extname(), ext.extver()):overscan_section})
        
        if ret_overscan_section == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_overscan_section
    
    def pixel_scale(self, dataset, **args):
        # Get the pixel scale lookup table
        gmosPixelScales = Lookups.get_lookup_table(
            "Gemini/GMOS/GMOSPixelScale", "gmosPixelScales")
        
        # Get the values of the instrument and the binning of the y-axis using
        # the appropriate descriptors. Use as_pytype() to return the values as
        # the default python type, rather than an object.
        instrument = dataset.instrument().as_pytype()
        detector_y_bin = dataset.detector_y_bin()
        
        # Determine the detector type keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_detector_type")
        
        # Get the value for the detector type keyword from the header of the
        # PHU
        detector_type = dataset.phu_get_key_value(keyword)
        
        if instrument is None or detector_y_bin is None or \
           detector_type is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the unbinned pixel scale (in arcseconds per unbinned pixel) from
        # the lookup table 
        pixel_scale_key = (instrument, detector_type)
        
        if pixel_scale_key in gmosPixelScales:
            raw_pixel_scale = gmosPixelScales[pixel_scale_key]
        else:
            raise Errors.TableKeyError()
        
        # Return the binned pixel scale value
        ret_pixel_scale = float(detector_y_bin * raw_pixel_scale)
        
        return ret_pixel_scale
    
    def read_noise(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_read_noise = {}
        
        # If the data have been prepared, take the read noise value directly
        # from the appropriate keyword. At some point, a check for the STDHDRSI
        # header keyword should be added, since the function that overwrites
        # the read noise keyword also writes the STDHDRSI keyword
        if "PREPARED" in dataset.types:
            
            # Loop over the pixel data extensions in the dataset
            for ext in dataset:
                
                # Determine the read noise keyword from the global keyword
                # dictionary
                keyword = self.get_descriptor_key("key_read_noise")
                
                # Get the value of the read noise keyword from the header of
                # the PHU
                raw_read_noise = ext.get_key_value(keyword)
                
                if raw_read_noise is None:
                    read_noise = None
                else:
                    read_noise = float(raw_read_noise)
                
                # Update the dictionary with the read noise value
                ret_read_noise.update({
                    (ext.extname(), ext.extver()):read_noise})
        else:
            
            # Get the lookup table containing read noise numbers by amplifier
            gmosampsRdnoise, gmosampsRdnoiseBefore20060831 = \
                Lookups.get_lookup_table("Gemini/GMOS/GMOSAmpTables",
                                         "gmosampsRdnoise",
                                         "gmosampsRdnoiseBefore20060831")
            
            # Determine the amplifier integration time keyord (ampinteg) from
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
            
            # Get the UT date using the appropriate descriptor
            ut_date = str(dataset.ut_date())
            
            if ut_date is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
            old_ut_date = datetime(2006, 8, 31, 0, 0)
            
            # Loop over the pixel data extensions of the dataset
            for ext in dataset:
                
                # Determine the name of the detector amplifier keyword
                # (ampname) from the global keyword dictionary
                keyword = self.get_descriptor_key("key_ampname")
                
                # Get the name of the detector amplifier from the header of
                # each pixel data extension
                ampname = ext.get_key_value(keyword)
                
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors. Use as_pytype() to return the values
                # as the default python type, rather than an object
                read_speed_setting = dataset.read_speed_setting().as_pytype()
                gain_setting = dataset.gain_setting().as_pytype()
                
                if ampname is None or read_speed_setting is None or \
                   gain_setting is None:
                    read_noise = None
                else:
                    read_noise_key = (
                        read_speed_setting, gain_setting, ampname)
                    
                    if obs_ut_date > old_ut_date:
                        if read_noise_key in gmosampsRdnoise:
                            read_noise = gmosampsRdnoise[read_noise_key]
                        else:
                            raise Errors.TableKeyError()
                    else:
                        if read_noise_key in gmosampsRdnoiseBefore20060831:
                            read_noise = gmosampsRdnoiseBefore20060831[
                                read_noise_key]
                        else:
                            raise Errors.TableKeyError()
                
                # Update the dictionary with the read noise value
                ret_read_noise.update({
                    (ext.extname(), ext.extver()):read_noise})
        
        if ret_read_noise == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_read_noise
    
    def read_speed_setting(self, dataset, **args):
        # Determine the amplifier integration time keyword (ampinteg) from the
        # global keyword dictionary
        keyword = self.get_descriptor_key("key_ampinteg")
        
        # Get the amplifier integration time from the header of the PHU
        ampinteg = dataset.phu_get_key_value(keyword)
        
        if ampinteg is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if ampinteg == 1000:
            ret_read_speed_setting = "fast"
        else:
            ret_read_speed_setting = "slow"
        
        return ret_read_speed_setting
    
    def saturation_level(self, dataset, **args):
        # Get the lookup table containing saturation values by amplifier
        gmosThresholds = Lookups.get_lookup_table(
            "Gemini/GMOS/GMOSThresholdValues", "gmosThresholds")
        
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_saturation_level = {}
        
        # Determine the detector type keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_detector_type")
        
        # Get the value of the detector type keyword from the header of the PHU
        detector_type = dataset.phu_get_key_value(keyword)
        
        if detector_type is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Determine which kind of CCDs we're working with
        is_eev=False
        is_e2vDD=False
        is_hamamatsu=False
        if detector_type=="SDSU II CCD":
            is_eev = True
        elif detector_type=="SDSU II e2v DD CCD42-90":
            is_e2vDD = True
        elif detector_type=="S10892-01":
            is_hamamatsu = True
        
        # The hard limit for saturation is the controller digitization limit
        controller_limit = 65535
        
        # Check whether data has been bias- or dark-subtracted
        biasim = dataset.phu_get_key_value(
            self.get_descriptor_key("key_bias_image"))
        darkim = dataset.phu_get_key_value(
            self.get_descriptor_key("key_dark_image"))
        
        # Get the gain dictionary for the dataset
        gain_dv = dataset.gain()
        if gain_dv is None:
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        else:
            gain_dict = gain_dv.dict_val
        
        # Get the binning factor for non-eev detectors
        if not is_eev:
            xbin_dv = dataset.detector_x_bin()
            if xbin_dv is None:
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            else:
                xbin_dict = xbin_dv.dict_val
            
            ybin_dv = dataset.detector_y_bin()
            if ybin_dv is None:
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            else:
                ybin_dict = ybin_dv.dict_val
        
        # Loop over the science extensions in the dataset to
        # determine whether bias level is needed
        # Also store some useful information in dictionaries
        # to be retrieved in the next loop
        need_bias_level = False
        data_contains_bias_dict = {}
        bin_factor_dict = {}
        for ext in dataset["SCI"]:
            dict_key = (ext.extname(),ext.extver())
            
            # Check whether data has been overscan-subtracted
            overscan = ext.get_key_value(
                self.get_descriptor_key("key_overscan_value"))
            
            # Check whether data still contains bias
            if overscan is not None:
                # Data was overscan-subtracted
                data_contains_bias = False
            elif biasim is not None or darkim is not None:
                # Data was not overscan-corrected, but it was
                # bias- or dark-corrected.
                data_contains_bias = False
            else:
                # Data still contains bias level
                data_contains_bias = True
            data_contains_bias_dict[dict_key] = data_contains_bias
            
            # For the newer CCDs, get the x and y binning. If xbin*ybin
            # is greater than 2, then saturation will always be the
            # controller limit
            if not is_eev:
                xbin = xbin_dict[dict_key]
                ybin = ybin_dict[dict_key]
                bin_factor = xbin*ybin
                bin_factor_dict[dict_key] = bin_factor
            
            # Get the bias level for all extensions only if necessary
            # because the bias level calculation can take some time
            if (not data_contains_bias) or \
               (not is_eev and data_contains_bias and bin_factor<=2):
                need_bias_level = True
        
        if need_bias_level:
            bias_level_dv = dataset.bias_level()
            if bias_level_dv is None:
                if hasattr(ext, "exception_info"):
                    raise ext.exception_info
            else:
                bias_level_dict = bias_level_dv.dict_val
        
        # Loop over extensions to calculate saturation value
        for ext in dataset["SCI"]:
            dict_key = (ext.extname(),ext.extver())
            
            # Determine the name of the detector amplifier keyword (ampname)
            # from the global keyword dictionary
            keyword = self.get_descriptor_key("key_ampname")
            
            # Get the value of the name of the detector amplifier keyword from
            # the header of each pixel data extension 
            ampname = ext.get_key_value(keyword)
            
            if ampname is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                 if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            # Check units of data (ie. ADU vs. electrons)
            bunit = ext.get_key_value(
                self.get_descriptor_key("key_bunit"))
            
            # Get the gain for the extension
            gain = gain_dict[dict_key]
            
            # For the newer CCDs, get the bin factor
            if not is_eev:
                bin_factor = bin_factor_dict[dict_key]
            
            # Check whether data still contains bias
            data_contains_bias = data_contains_bias_dict[dict_key]
            
            # Get the bias level of the extension only if necessary
            if (not data_contains_bias) or \
               (not is_eev and data_contains_bias and bin_factor<=2):
                bias_level = bias_level_dict[dict_key]
            
            # Correct the controller limit for bias level and units
            processed_limit = controller_limit
            if not data_contains_bias:
                processed_limit -= bias_level
            if bunit=="electron" or bunit=="electrons":
                processed_limit *= gain
            
            if is_eev or bin_factor>2:
                # For old EEV CCDs, use the detector limit
                saturation = processed_limit
            else:
                # For new GMOS-N CCDs, look up the saturation limit from
                # the table, then correct for binning, units, and bias level
                
                # Get the base saturation value from the lookup table
                if ampname in gmosThresholds:
                    # saturation value is an integer with units ADU
                    saturation = gmosThresholds[ampname]
                else:
                    # This error condition will be hit for all mosaicked
                    # or tiled data
                    raise Errors.TableKeyError()
                
                # Correct the saturation level for binning
                saturation = saturation * bin_factor
                
                # The saturation level is reported in electrons; convert
                # it to ADU if needed
                if (bunit!="electron" and bunit!="electrons"):
                    saturation = saturation / gain
                
                # The saturation level does not contain the bias; add it
                # in if necessary
                if data_contains_bias:
                    saturation += bias_level
                
                # Check whether the value is now over the controller limit;
                # if so, set it to the hard limit
                if saturation > processed_limit:
                    saturation = processed_limit
            
            ret_saturation_level.update({dict_key: saturation})
        
        if ret_saturation_level == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        return ret_saturation_level
    
    def wavelength_band(self, dataset, **args):
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = Lookups.get_lookup_table(
                "Gemini/GMOS/GMOSFilterWavelength", "filter_wavelength")
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
        
        return ret_wavelength_band
