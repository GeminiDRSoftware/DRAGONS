import datetime, os, re
import dateutil.parser
import math
import pywcs

from astrodata.utils import Errors
from astrodata.interface.slices import pixel_exts
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_metadata_utils as gmu
from gempy.gemini.coordinate_utils import toicrs

import GemCalcUtil
from FITS_Descriptors import FITS_DescriptorCalc
from GEMINI_Keywords import GEMINI_KeyDict

from astrodata_Gemini.ADCONFIG_Gemini.lookups import NominalExtinction
from astrodata_Gemini.ADCONFIG_Gemini.lookups import WavelengthBand
# ------------------------------------------------------------------------------
class GEMINI_DescriptorCalc(FITS_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GEMINI_KeyDict
    nominal_extinction_table = None
    std_wavelength_band = None
    
    def __init__(self):
        self.std_wavelength_band = WavelengthBand.wavelength_band
        self.nominal_extinction_table =  NominalExtinction.nominal_extinction
        FITS_DescriptorCalc.__init__(self)
    
    def airmass(self, dataset, **args):
        # Determine the airmass keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_airmass")
        
        # Get the value of the airmass keyword from the header of the PHU
        airmass = dataset.phu_get_key_value(keyword)
        
        if airmass is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Validate the airmass value
        if airmass < 1.0:
            raise Errors.InvalidValueError()
        else:
            ret_airmass = float(airmass)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_airmass, name="airmass", ad=dataset)
        
        return ret_dv
    
    def amp_read_area(self, dataset, **args):
        # The amp_read_area descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific amp_read_area
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of "GMOS". For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def ao_seeing(self, dataset, **args):
        # For Altair observations, the AO-estimated seeing in arcseconds is
        # provided directly in the header. For GeMS observations, the 
        # estimated seeing must be calculated.
        
        # Determine the ao_seeing keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_ao_seeing")
        
        # Get the value of the ao_seeing keyword from the header of the PHU
        ao_seeing = dataset.phu_get_key_value(keyword)
       
        if not ao_seeing:
            # If there is no ao_seeing keyword, try the r_zero_val keyword
            keyword = self.get_descriptor_key("key_r_zero_val")
            r_zero_val = dataset.phu_get_key_value(keyword)
            
            if r_zero_val:
                # If the r_zero_val keyword (Fried's parameter) is present, 
                # a seeing estimate can be calculated (NOTE: Jo Thomas-Osip 
                # is providing a reference for this calculation. Until then, 
                # EJD checked using 
                # http://www.ctio.noao.edu/~atokovin/tutorial/part1/turb.html )

                # Seeing at 0.5 micron
                seeing_ref = (206265. * 0.98 * 0.5e-6) / (r_zero_val * 0.01)
                # Adjusting to wavelength of observation
                keyword = self.get_descriptor_key("key_wavelength")
                wavelength = dataset.phu_get_key_value(keyword)
                if wavelength:
                    ao_seeing = seeing_ref * (wavelength/5000.)**(-0.2)
                else:
                    raise Errors.ExistError()
            else:
                raise Errors.ExistError()
            
        return ao_seeing
        
    def array_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_array_section = {}
        
        # Determine the array section keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_array_section")
        
        # Get the value of the array section keyword from the header of each
        # pixel data extension as a dictionary where the key of the dictionary
        # is an ("*", EXTVER) tuple
        array_section_dict = gmu.get_key_value_dict(dataset, keyword)
        
        if array_section_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
            
        for ext_name_ver, raw_array_section in array_section_dict.iteritems():
            if raw_array_section is None:
                array_section = None
            elif pretty:
                # Use the array section string that uses 1-based indexing as
                # the value in the form [x1:x2,y1:y2]
                array_section = str(raw_array_section)
            else:
                # Use the array section list that uses 0-based, non-inclusive
                # indexing as the value in the form [x1, x2, y1, y2]
                array_section = gmu.sectionStrToIntList(raw_array_section)
            
            # Update the dictionary with the array section value
            ret_array_section.update({ext_name_ver:array_section})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_array_section, name="array_section",
                                 ad=dataset)
        return ret_dv

    def camera(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the camera name, stripping the ID and making pretty as appropriate
        """

        keyword = self.get_descriptor_key("key_camera")
        camera = dataset.phu_get_key_value(keyword)

        if camera is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if pretty:
            stripID = True
        if stripID:
            # Return the camera string with the component ID stripped
            ret_camera = gmu.removeComponentID(camera)
        else:
            # Return the decker string
            ret_camera = str(camera)

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_camera, name="camera", ad=dataset)

        return ret_dv

    def cass_rotator_pa(self, dataset, **args):
        # Determine the cassegrain rotator position angle keyword from the
        # global keyword dictionary
        keyword = self.get_descriptor_key("key_cass_rotator_pa")
        
        # Get the value of the cassegrain rotator position angle keyword from
        # the header of the PHU 
        cass_rotator_pa = dataset.phu_get_key_value(keyword)
        
        if cass_rotator_pa is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Validate the cassegrain rotator position angle value
        if cass_rotator_pa < -360.0 or cass_rotator_pa > 360.0:
            raise Errors.InvalidValueError()
        else:
            ret_cass_rotator_pa = float(cass_rotator_pa)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_cass_rotator_pa, name="cass_rotator_pa",
                                 ad=dataset)
        return ret_dv
    
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
    
    def coadds(self, dataset, **args):
        # Determine the number of coadds keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_coadds")
        
        # Get the value of the number of coadds keyword from the header of the
        # PHU
        coadds = dataset.phu_get_key_value(keyword)
        
        if coadds is None:
            # Return 1 as the default value for the number of coadds for Gemini
            # data
            ret_coadds = int(1)
        else:
            ret_coadds = int(coadds)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_coadds, name="coadds", ad=dataset)
        
        return ret_dv
    
    def data_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_data_section = {}
        
        # Determine the data section keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_data_section")
        
        # Get the value of the data section keyword from the header of each
        # pixel data extension as a dictionary where the key of the dictionary
        # is an ("*", EXTVER) tuple 
        data_section_dict = gmu.get_key_value_dict(dataset, keyword)
        
        if data_section_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        for ext_name_ver, raw_data_section in data_section_dict.iteritems():
            if raw_data_section is None:
                data_section = None
            elif pretty:
                # Use the data section string that uses 1-based indexing as the
                # value in the form [x1:x2,y1:y2]
                data_section = str(raw_data_section)
            else:
                # Use the data section list that uses 0-based, non-inclusive
                # indexing as the value in the form [x1, x2, y1, y2]
                data_section = gmu.sectionStrToIntList(raw_data_section)
            
            # Update the dictionary with the data section value
            ret_data_section.update({ext_name_ver:data_section})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_data_section, name="data_section",
                                 ad=dataset)
        return ret_dv

    def decker(self, dataset, stripID=False, pretty=False, **args):
        """
        In GNIRS, the decker is used to basically mask off the ends of the
        slit to create the short slits used in the cross dispersed modes.
        """
        # Determine the decker position keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_decker")
        
        # Get the value of the decker position keyword from the header of the
        # PHU 
        decker = dataset.phu_get_key_value(keyword)
        
        if decker is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if pretty:
            stripID = True
        if stripID:
            # Return the decker string with the component ID stripped
            ret_decker = gmu.removeComponentID(decker)
        else:
            # Return the decker string
            ret_decker = str(decker)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_decker, name="decker", ad=dataset)
        
        return ret_dv

    def detector_roi_setting(self, dataset, **args):
        # For instruments that do not support setting the ROI, return "Fixed"
        ret_detector_roi_setting = "Fixed"
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_roi_setting,
                                 name="detector_roi_setting", ad=dataset)
        return ret_dv
    
    def detector_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_section = {}
        
        # Determine the detector section keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_detector_section")
        
        # Get the value of the detector section keyword from the header of each
        # pixel data extension as a dictionary where the key of the dictionary
        # is an ("*", EXTVER) tuple
        detector_section_dict = gmu.get_key_value_dict(dataset, keyword)
        
        if detector_section_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ds_dict = detector_section_dict.iteritems()
        for ext_name_ver, raw_detector_section in ds_dict:
            if raw_detector_section is None:
                detector_section = None
            elif pretty:
                # Use the detector section string that uses 1-based indexing as
                # the value in the form [x1:x2,y1:y2] 
                detector_section = str(raw_detector_section)
            else:
                # Use the detector section list that uses 0-based,
                # non-inclusive indexing as the value in the form
                # [x1, x2, y1, y2] 
                detector_section = gmu.sectionStrToIntList(
                  raw_detector_section)
            
            # Update the dictionary with the detector section value
            ret_detector_section.update({ext_name_ver:detector_section})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_section, name="detector_section",
                                 ad=dataset)
        return ret_dv
    
    def detector_x_bin(self, dataset, **args):
        # Return the binning of the x-axis integer (set to 1 as default for
        # Gemini data)
        ret_detector_x_bin = int(1)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_x_bin, name="detector_x_bin",
                                 ad=dataset)
        return ret_dv
    
    def detector_y_bin(self, dataset, **args):
        # Return the binning of the y-axis integer (set to 1 as default for
        # Gemini data)
        ret_detector_y_bin = int(1)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_detector_y_bin, name="detector_y_bin",
                                 ad=dataset)
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
            stripID = True
        if stripID:
            # Return the disperser string with the component ID stripped
            ret_disperser = gmu.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_disperser, name="disperser", ad=dataset)
        
        return ret_dv
    
    def dispersion_axis(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_dispersion_axis = {}
        
        # The dispersion axis can only be obtained from data that does not
        # have an AstroData Type of IMAGE and that has been prepared (since
        # the dispersion axis keyword is written during the prepare step)
        if "IMAGE" not in dataset.types and "PREPARED" in dataset.types:
            
            # Determine the dispersion axis keyword from the global keyword
            # dictionary
            keyword = self.get_descriptor_key("key_dispersion_axis")
            
            # Get the value of the dispersion axis keyword from the header of
            # each pixel data extension as a dictionary where the key of the
            # dictionary is an ("*", EXTVER) tuple
            dispersion_axis_dict = gmu.get_key_value_dict(dataset, keyword)
            
            if dispersion_axis_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            da_dict = dispersion_axis_dict.iteritems()
            for ext_name_ver, raw_dispersion_axis in da_dict:
                if raw_dispersion_axis is None:
                    dispersion_axis = None
                else:
                    # Use the dispersion axis integer as the value
                    dispersion_axis = int(raw_dispersion_axis)
                
                # Update the dictionary with the dispersion axis value
                ret_dispersion_axis.update({ext_name_ver:dispersion_axis})
        else:
            raise Errors.DescriptorTypeError()
        
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

        # Make broken negative exposure times act like the header is missing
        if (exposure_time < 0):
            exposure_time = None
        
        if exposure_time is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # If the data have been prepared, take the (total) exposure time value
        # directly from the appropriate keyword
        if "PREPARED" in dataset.types:
            # Get the total exposure time value from the header of the PHU
            ret_exposure_time = float(exposure_time)
        else:
            # Get the number of coadds using the appropriate descriptor
            coadds = dataset.coadds()
            
            if coadds is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            ret_exposure_time = float(exposure_time * coadds)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_exposure_time, name="exposure_time",
                                 ad=dataset)
        return ret_dv
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Determine the two filter name keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_filter1")
        keyword2 = self.get_descriptor_key("key_filter2")
        
        # Get the value of the two filter name keywords from the header of the
        # PHU
        filter1 = dataset.phu_get_key_value(keyword1)
        filter2 = dataset.phu_get_key_value(keyword2)
        
        if filter1 is None or filter2 is None:
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
            # Remove any filters that have the value "open", "Open" or "Clear"
            if ("open" not in filter1 and "Open" not in filter1 and
                "Clear" not in filter1):
                filter.append(str(filter1))
            if ("open" not in filter2 and "Open" not in filter2 and
                "Clear" not in filter2):
                filter.append(str(filter2))
            if len(filter) == 0:
                filter.append("open")
            if "Block" in filter1 or "Block" in filter2:
                filter.append("blank")
            if "Dark" in filter1 or "Dark" in filter2:
                filter.append("blank")
            if "DK" in filter1 or "DK" in filter2:
                filter.append("dark")
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
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        
        # Determine the focal plane mask keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_focal_plane_mask")
        
        # Get the value of the focal plane mask value from the header of the
        # PHU
        focal_plane_mask = dataset.phu_get_key_value(keyword)
        
        if focal_plane_mask is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if stripID:
            # Return the focal plane mask string with the component ID stripped
            ret_focal_plane_mask = gmu.removeComponentID(focal_plane_mask)
        else:
            # Return the focal plane mask string
            ret_focal_plane_mask = str(focal_plane_mask)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_focal_plane_mask, name="focal_plane_mask",
                                 ad=dataset)
        return ret_dv
    
    def gain_setting(self, dataset, **args):
        # The gain_setting descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific gain_setting
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of "GMOS". For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def gcal_lamp(self, dataset, **args):
        # This tells you what GCAL is radiating. Generally, this is given by
        # the GCALLAMP header, *except* in the case of the IR lamp which is
        # never turned off (because it takes ages to stabilize when you turn it
        # back on) but is behind a shutter, the state of which is given by the
        # GCALSHUT header. Note that the only thing behind the shutter is the 
        # IR lamp, the other lamps are not shuttered but are simply turned
        # on and off on demand.

        # If the headers are not found, then GCAL is not in the beam.
        # phu_get_key_value will return None and that will be returned
        # to the called. This is desired behavour.

        # Various values have been seen in data to reflect no lamp on
        # These are all returned as "Off" by this descriptor

        # Lamps illuminated
        lamps = dataset.phu_get_key_value("GCALLAMP")

        # IR shutter state
        shut = dataset.phu_get_key_value("GCALSHUT")

        if shut == 'CLOSED':
            if lamps == 'IRhigh' or lamps == 'IRlow':
                lamps = ''

        if lamps in ['', 'No Value']:
            lamps = 'Off'

        return lamps

    def group_id(self, dataset, **args):
        # Get the observation id using the appropriate descriptor
        observation_id = dataset.observation_id()
        
        if observation_id is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the group_id string, which is equal to the observation_id for
        # Gemini data 
        ret_group_id = "%s" % (observation_id)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_group_id, name="group_id", ad=dataset)
        
        return ret_dv
    
    def is_ao(self, dataset, **args):
        # Returns True if an observation uses adaptive optics, and False
        # otherwise
        
        # Determine the ao_fold keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_ao_fold")
        
        # Get the value of the ao_seeing keyword from the header of the PHU
        ao_fold = dataset.phu_get_key_value(keyword)
        
        if ao_fold:
            is_ao = False
            if ao_fold == "IN":
                is_ao = True
        else:
            raise Errors.ExistError()
           
        return is_ao

    def local_time(self, dataset, **args):
        # Determine the local time keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_local_time")
        
        # Get the value of the local time keyword from the header of the PHU
        local_time = dataset.phu_get_key_value(keyword)
        
        if local_time is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Validate the local time value. The assumption is that the standard
        # mandates HH:MM:SS[.S]. We don't enforce the number of decimal places.
        # These are somewhat basic checks, it's not completely rigorous. Note
        # that seconds can be > 59 when leap seconds occurs
        if re.match("^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$", local_time):
            ret_local_time = dateutil.parser.parse(local_time).time()
        else:
            raise Errors.InvalidValueError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_local_time, name="local_time", ad=dataset)
        
        return ret_dv
    
    def mdf_row_id(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_mdf_row_id = {}
        
        # The MDF row ID can only be obtained from data that does not have an
        # AstroData Type of IMAGE and that has been cut (since the MDF row ID
        # keyword is written during the cut step). As there is no CUT type yet,
        # just check whether the dataset has been prepared.
        if "IMAGE" not in dataset.types and "PREPARED" in dataset.types:
            
            # Determine the MDF row ID keyword from the global keyword
            # dictionary
            keyword = self.get_descriptor_key("key_mdf_row_id")
            
            # Get the value of the MDF row ID keyword from the header of each
            # pixel data extension as a dictionary where the key of the
            # dictionary is an ("*", EXTVER) tuple
            mdf_row_id_dict = gmu.get_key_value_dict(dataset, keyword)
            
            if mdf_row_id_dict is None:
                # The get_key_value_dict() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
            for ext_name_ver, raw_mdf_row_id in mdf_row_id_dict.iteritems():
                if raw_mdf_row_id is None:
                    mdf_row_id = None
                else:
                    # Use the MDF row ID integer as the value
                    mdf_row_id = int(raw_mdf_row_id)
                
                # Update the dictionary with the MDF row ID value
                ret_mdf_row_id.update({ext_name_ver:mdf_row_id})
        else:
            raise Errors.DescriptorTypeError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_mdf_row_id, name="mdf_row_id", ad=dataset)
        
        return ret_dv
    
    def nod_count(self, dataset, **args):
        # The nod_count descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific nod_count
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of "GMOS". For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def nod_pixels(self, dataset, **args):
        # The nod_pixels descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific nod_pixels
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of "GMOS". For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def nominal_atmospheric_extinction(self, dataset, **args):
        # Get the telescope, filter and airmass using the appropriate
        # descriptors
        telescope = str(dataset.telescope())
        filter = str(dataset.filter_name(pretty=True))
        airmass = dataset.airmass()
        
        # Get the nominal extinction co-efficients from the lookup table for
        # the appropriate telescope (i.e., site) and filter
        table = self.nominal_extinction_table
        
        try:
            coeff = table[(telescope, filter)]
        except KeyError:
            coeff = 0.0
        
        ret_nominal_atmospheric_extinction = coeff * (airmass - 1.0)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nominal_atmospheric_extinction,
                                 name="nominal_atmospheric_extinction",
                                 ad=dataset)
        return ret_dv
    
    def overscan_section(self, dataset, **args):
        # The overscan_section descriptor is only specific to GMOS data. The
        # code below will be replaced with the GMOS specific overscan_section 
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of "GMOS". For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()

    def qa_state(self, dataset, **args):
        # Determine the keywords for whether the PI requirements were met
        # (rawpireq) and the raw Gemini Quality Assessment (rawgemqa) from the
        # global keyword dictionary
        keyword1 = self.get_descriptor_key("key_raw_pi_requirements_met")
        keyword2 = self.get_descriptor_key("key_raw_gemini_qa")
        
        # Get the value for whether the PI requirements were met and the value
        # for the raw Gemini Quality Assessment keywords from the header of the
        # PHU
        rawpireq = dataset.phu_get_key_value(keyword1)
        rawgemqa = dataset.phu_get_key_value(keyword2)
        
        if rawpireq is None or rawgemqa is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Calculate the derived QA state
        ret_qa_state = "%s:%s" % (rawpireq, rawgemqa)
        if rawpireq == "UNKNOWN" or rawgemqa == "UNKNOWN":
            ret_qa_state = "Undefined"
        if rawpireq.upper() == "YES" and rawgemqa.upper() == "USABLE":
            ret_qa_state = "Pass"
        if rawpireq.upper() == "NO" and rawgemqa.upper() == "USABLE":
            ret_qa_state = "Usable"
        if rawgemqa.upper() == "BAD":
            ret_qa_state = "Fail"
        if rawpireq.upper() == "CHECK" or rawgemqa.upper() == "CHECK":
            ret_qa_state = "CHECK"
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_qa_state, name="qa_state", ad=dataset)
        
        return ret_dv

    def raw_bg(self, dataset, **args):
        # Determine the raw background keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_raw_bg")
        
        # Get the value of the raw background keyword from the header of the
        # PHU
        raw_bg = dataset.phu_get_key_value(keyword)
        
        if raw_bg is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(raw_bg)
        
        if percentile is not None:
            ret_raw_bg = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_raw_bg, name="raw_bg", ad=dataset)
        
        return ret_dv
    
    def raw_cc(self, dataset, **args):
        # Determine the raw cloud cover keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_raw_cc")
        
        # Get the value of the raw cloud cover keyword from the header of the
        # PHU
        raw_cc = dataset.phu_get_key_value(keyword)
        
        if raw_cc is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(raw_cc)
        
        if percentile is not None:
            ret_raw_cc = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_raw_cc, name="raw_cc", ad=dataset)
        
        return ret_dv
    
    def raw_iq(self, dataset, **args):
        # Determine the raw image quality keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_raw_iq")
        
        # Get the value of the raw image quality keyword from the header of the
        # PHU
        raw_iq = dataset.phu_get_key_value(keyword)
        
        if raw_iq is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(raw_iq)
        
        if percentile is not None:
            ret_raw_iq = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_raw_iq, name="raw_iq", ad=dataset)
        
        return ret_dv
    
    def raw_wv(self, dataset, **args):
        # Determine the raw water vapour keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_raw_wv")
        
        # Get the value of the raw water vapour keyword from the header of the
        # PHU
        raw_wv = dataset.phu_get_key_value(keyword)
        
        if raw_wv is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(raw_wv)
        
        if percentile is not None:
            ret_raw_wv = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_raw_wv, name="raw_wv", ad=dataset)
        
        return ret_dv

    def read_speed_setting(self, dataset, **args):
        # The read_speed_setting descriptor is only specific to GMOS data. The
        # code below will be replaced with the GMOS specific read_speed_setting
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of "GMOS". For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def requested_bg(self, dataset, **args):
        # Determine the requested background keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_requested_bg")
        
        # Get the value of the requested background keyword from the header of
        # the PHU
        requested_bg = dataset.phu_get_key_value(keyword)
        
        if requested_bg is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(requested_bg)
        
        if percentile is not None:
            ret_requested_bg = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_requested_bg, name="requested_bg",
                                 ad=dataset)
        return ret_dv
    
    def requested_cc(self, dataset, **args):
        # Determine the requested cloud cover keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_requested_cc")
        
        # Get the value of the requested cloud cover keyword from the header of
        # the PHU
        requested_cc = dataset.phu_get_key_value(keyword)
        
        if requested_cc is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(requested_cc)
        
        if percentile is not None:
            ret_requested_cc = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_requested_cc, name="requested_cc",
                                 ad=dataset)
        return ret_dv
    
    def requested_iq(self, dataset, **args):
        # Determine the requested image quality keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_requested_iq")
        
        # Get the value of the requested image quality keyword from the header
        # of the PHU
        requested_iq = dataset.phu_get_key_value(keyword)
        
        if requested_iq is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(requested_iq)
        
        if percentile is not None:
            ret_requested_iq = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_requested_iq, name="requested_iq",
                                 ad=dataset)
        return ret_dv
    
    def requested_wv(self, dataset, **args):
        # Determine the requested water vapour keyword from the global keyword
        # dictionary 
        keyword = self.get_descriptor_key("key_requested_wv")
        
        # Get the value of the requested water vapour keyword from the header
        # of the PHU
        requested_wv = dataset.phu_get_key_value(keyword)
        
        if requested_wv is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return an integer percentile value (more useful for comparison)
        percentile = gmu.parse_percentile(requested_wv)
        
        if percentile is not None:
            ret_requested_wv = percentile
        else:
            raise Errors.InvalidValueError
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_requested_wv, name="requested_wv",
                                 ad=dataset)
        return ret_dv
    
    def target_ra(self, dataset, offset=False, pm=True, icrs=False, **args):
        # Return the target RA from the RA header
        # Optionally also apply RAOFFSET
        # Optionally also apply proper motion
        # Optioanlly convert to ICRS. This works even for APPT.

        ra = dataset.phu_get_key_value('RA')
        raoffset = dataset.phu_get_key_value('RAOFFSET')
        targ_raoffset = dataset.phu_get_key_value('RATRGOFF')
        pmra = dataset.phu_get_key_value('PMRA')
        epoch = dataset.phu_get_key_value('EPOCH')
        frame = dataset.phu_get_key_value('FRAME')
        equinox = dataset.phu_get_key_value('EQUINOX')
        if ra is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if offset:
            raoffset = 0 if raoffset is None else raoffset
            targ_raoffset = 0 if targ_raoffset is None else targ_raoffset
            raoffset /= 3600.0
            targ_raoffset /= 3600.0
            raoffset += targ_raoffset
            raoffset /= math.cos(math.radians(dataset.target_dec(offset=True).as_pytype()))
            ra += raoffset

        pmra = 0 if pmra is None else pmra
        if pm and pmra != 0:
            dt = dataset.ut_datetime().as_pytype()
            year = dt.year
            startyear = datetime.datetime(year, 1, 1, 0, 0, 0)
            # Handle leap year properly
            nextyear = datetime.datetime(year+1, 1, 1, 0, 0, 0)
            thisyear = nextyear - startyear
            sofar = dt - startyear
            fraction = sofar.total_seconds() / thisyear.total_seconds()
            obsepoch = year + fraction
            years = obsepoch - epoch
            pmra *= years
            pmra *= 15.0*math.cos(math.radians(dataset.target_dec(offset=True).as_pytype()))
            pmra /= 3600.0
            ra += pmra

        if icrs:
            ra, dec = toicrs(frame,
                             dataset.target_ra(offset=offset, pm=pm, icrs=False).as_pytype(),
                             dataset.target_dec(offset=offset, pm=pm, icrs=False).as_pytype(),
                             equinox=2000.0,
                             ut_datetime=dataset.ut_datetime().as_pytype())

        return DescriptorValue(ra, name="target_ra", ad=dataset)

    def target_dec(self, dataset, offset=False, pm=True, icrs=False, **args):
        # Return the target Dec from the DEC header
        # Optionally also apply DECOFFSET
        # Optionally also apply proper motion
        # Optioanlly convert to ICRS. This works even for APPT.

        dec = dataset.phu_get_key_value('DEC')
        decoffset = dataset.phu_get_key_value('DECOFFSE')
        targ_decoffset = dataset.phu_get_key_value('DECTRGOF')
        pmdec = dataset.phu_get_key_value('PMDEC')
        epoch = dataset.phu_get_key_value('EPOCH')
        frame = dataset.phu_get_key_value('FRAME')
        equinox = dataset.phu_get_key_value('EQUINOX')

        if dec is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if offset:
            decoffset = 0.0 if decoffset is None else decoffset
            targ_decoffset = 0.0 if targ_decoffset is None else targ_decoffset
            decoffset /= 3600.0
            targ_decoffset /= 3600.0
            dec += decoffset + targ_decoffset

        pmdec = 0 if pmdec is None else pmdec
        if pm and pmdec != 0:
            dt = dataset.ut_datetime().as_pytype()
            year = dt.year
            startyear = datetime.datetime(year, 1, 1, 0, 0, 0)
            # Handle leap year properly
            nextyear = datetime.datetime(year+1, 1, 1, 0, 0, 0)
            thisyear = nextyear - startyear
            sofar = dt - startyear
            fraction = sofar.total_seconds() / thisyear.total_seconds()
            obsepoch = year + fraction
            years = obsepoch - epoch
            pmdec *= years
            pmdec /= 3600.0
            dec += pmdec

        if icrs:
            ra, dec = toicrs(frame,
                             dataset.target_ra(offset=offset, pm=pm, icrs=False).as_pytype(),
                             dataset.target_dec(offset=offset, pm=pm, icrs=False).as_pytype(),
                             equinox=2000.0,
                             ut_datetime=dataset.ut_datetime().as_pytype())

        return DescriptorValue(dec, name="target_dec", ad=dataset)

    def ut_date(self, dataset, **args):
        # Call ut_datetime(strict=True, dateonly=True) to return a valid
        # ut_date, if possible.
        ret_ut_date = dataset.ut_datetime(strict=True, dateonly=True)
        
        if ret_ut_date is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_ut_date, name="ut_date", ad=dataset)
        
        return ret_dv
    
    def ut_datetime(self, dataset, strict=False, dateonly=False,
                    timeonly=False, **args):
        # First, we try and figure out the date, looping through several
        # header keywords that might tell us. DATE-OBS can also give us a full
        # date-time combination, so we check for this too.
        for kw in ["DATE-OBS", self.get_descriptor_key("key_ut_date"), "DATE",
                   "UTDATE"]:
            try:
                utdate_hdr = dataset.phu_get_key_value(kw).strip()
            except (KeyError, AttributeError):
                #print "Didn't get a utdate from keyword %s" % kw
                utdate_hdr = ""

            # Validate. The definition is taken from the FITS
            # standard document v3.0. Must be YYYY-MM-DD or
            # YYYY-MM-DDThh:mm:ss[.sss]. Here I also do some very basic checks
            # like ensuring the first digit of the month is 0 or 1, but I
            # don't do cleverer checks like 01<=M<=12. nb. seconds ss > 59 is
            # valid when leap seconds occur.

            # Did we get a full date-time string?
            if re.match("(\d\d\d\d-[01]\d-[0123]\d)(T)([012]\d:[012345]\d:\d\d.*\d*)", utdate_hdr):
                ut_datetime = dateutil.parser.parse(utdate_hdr)
                #print "Got a full date-time from %s: %s = %s" % (kw, utdate_hdr, ut_datetime)
                # If we got a full datetime, then just return it and we're done already!
                return ut_datetime

            # Did we get a date (only) string?
            match = re.match("\d\d\d\d-[01]\d-[0123]\d", utdate_hdr)
            if match:
                #print "got a date from %s: %s" % (kw, utdate_hdr)
                break
            else:
                #print "did not get a date from %s: %s" % (kw, utdate_hdr)
                pass

            # Did we get a *horrible* early niri style string DD/MM/YY[Y] - YYYY = 1900 + YY[Y]?
            match = re.match("(\d\d)/(\d\d)/(\d\d+)", utdate_hdr)
            if(match):
                #print "horrible niri format"
                d = int(match.groups()[0])
                m = int(match.groups()[1])
                y = 1900 + int(match.groups()[2])
                if((d > 0) and (d < 32) and (m > 0) and (m < 13) and (y>1990) and (y<2050)):
                    #print "got a date from horrible niri format"
                    utdate_hdr = "%d-%02d-%02d" % (y, m, d)
                    break
            else:
                utdate_hdr = ""


        # OK, at this point, utdate_hdr should contain either an empty string
        # or a valid date string, ie YYYY-MM-DD

        # If that's all we need, return it now
        if(utdate_hdr and dateonly):
            ut_datetime = dateutil.parser.parse(utdate_hdr+" 00:00:00")
            return ut_datetime.date()

        # Get and validate the ut time header, if present. We try several
        # header keywords that might contain a ut time.
        for kw in [self.get_descriptor_key("key_ut_time"), "UT", "TIME-OBS", "STARTUT", "UTSTART"]:
            try:
                uttime_hdr = dataset.phu_get_key_value(kw).strip()
            except (KeyError, AttributeError):
                #print "Didn't get a uttime from keyword %s" % kw
                uttime_hdr = ""
            # The standard mandates HH:MM:SS[.S...] 
            # OK, we allow single digits to cope with crap data
            # These are somewhat basic checks, it's not completely rigorous
            # Note that seconds can be > 59 when leap seconds occurs
            if re.match("^([012]?\d)(:)([012345]?\d)(:)(\d\d?\.?\d*)$", uttime_hdr):
                #print "Got UT time from keyword %s: %s" % (kw, uttime_hdr)
                break
            else:
                #print "Could not parse a UT time from keyword %s: %s" % (kw, uttime_hdr)
                uttime_hdr = ""
        # OK, at this point, uttime_hdr should contain either an empty string
        # or a valid UT time string, ie HH:MM:SS[.S...]
          
        # If that's all we need, parse it and return it now
        if(uttime_hdr and timeonly):
            ut_datetime = dateutil.parser.parse("2000-01-01 "+uttime_hdr)
            return ut_datetime.time()

        # OK, if we got both a date and a time, then we can go ahead
        # and stick them together then parse them into a datetime object
        if(utdate_hdr and uttime_hdr):
            datetime_str = "%sT%s" % (utdate_hdr, uttime_hdr)
            #print "Got both a utdate and a uttime and made: %s" % datetime_str
            ut_datetime = dateutil.parser.parse(datetime_str)
            return ut_datetime

        # OK, we didn't get good ut_date and ut_time headers, this is non-compliant
        # data, probably taken in engineering mode or something
 
        # Maybe there's an MJD_OBS header we can use
        try:
            #print "Trying to use MJD_OBS"
            mjd = dataset.phu_get_key_value("MJD_OBS")
            if(mjd > 1):
                # MJD zero is 1858-11-17T00:00:00.000 UTC
                mjdzero = datetime.datetime(1858, 11, 17, 0, 0, 0, 0, None)
                mjddelta = datetime.timedelta(mjd)
                ut_datetime = mjdzero + mjddelta
                #print "determined ut_datetime from MJD_OBS"
                if(dateonly):
                    return ut_datetime.date()
                if(timeonly):
                    return ut_datetime.time()
                return ut_datetime
        except KeyError:
            pass

        # Maybe there's an OBSSTART header we can use
        try:
            #print "Trying to use OBSSTART"
            obsstart = dataset.phu_get_key_value("OBSSTART").strip()
            if(obsstart):
                # Sometimes has a Z on the end which gives the datetime a tzinfo = tzutc
                ut_datetime = dateutil.parser.parse(obsstart).replace(tzinfo=None)
                #print "Did it by OBSSTART"
                if(dateonly):
                    return ut_datetime.date()
                if(timeonly):
                    return ut_datetime.time()
                return ut_datetime
        except (KeyError, AttributeError):
            pass

        # OK, now we're getting a desperate. If we're in strict mode, we give up now
        if(strict):
            return None

        # If we didn't get a utdate, can we parse it from the framename header if there is one, or the filename?
        if(not utdate_hdr):
            #print "Desperately trying FRMNAME, filename etc"
            try:
                # Loop over the pixel data extensions in the dataset
                for ext in dataset[pixel_exts]:
                    frmname = ext.get_key_value("FRMNAME")
            except (KeyError, ValueError, IndexError):
                frmname = ""
            for string in [frmname, os.path.basename(dataset.filename)]:
                try:
                    #print "... Trying to Parse: %s" % string
                    year = string[1:5]
                    y = int(year)
                    month = string[5:7]
                    m = int(month)
                    day = string[7:9]
                    d = int(day)
                    if(( y > 1999) and (m < 13) and (d < 32)):
                        utdate_hdr = "%s-%s-%s" % (year, month, day)
                        #print "Guessed utdate from %s: %s" % (string, utdate_hdr)
                        break
                except (KeyError, ValueError, IndexError):
                    pass

        # If we didn't get a uttime, assume midnight
        if(not uttime_hdr):
            #print "Assuming midnight. Bleah."
            uttime_hdr = "00:00:00"


        # OK, if all we wanted was a date, and we've got it, return it now
        if(utdate_hdr and dateonly):
            ut_datetime = dateutil.parser.parse(utdate_hdr+" 00:00:00")
            return ut_datetime.date()

        # OK, if all we wanted was a time, and we've got it, return it now
        if(uttime_hdr and timeonly):
            ut_datetime = dateutil.parser.parse("2000-01-01 "+uttime_hdr)
            return ut_datetime.time()

        # If we've got a utdate and a uttime, return it
        if(utdate_hdr and uttime_hdr):
            datetime_str = "%sT%s" % (utdate_hdr, uttime_hdr)
            #print "Got both a utdate and a uttime and made: %s" % datetime_str
            ut_datetime = dateutil.parser.parse(datetime_str)
            return ut_datetime

        # Well, if we get here, we really have no idea
        return None
    
    def ut_time(self, dataset, **args):
        # Call ut_datetime(strict=True, timeonly=True) to return a valid
        # ut_time, if possible.
        ret_ut_time = dataset.ut_datetime(strict=True, timeonly=True)
        
        if ret_ut_time is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_ut_time, name="ut_time", ad=dataset)
        
        return ret_dv
    
    def wavefront_sensor(self, dataset, **args):
        # Determine the AOWFS, OIWFS, PWFS1 and PWFS2 probe states (aowfs,
        # oiwfs, pwfs1 and pwfs2, respectively) keywords from the global
        # keyword dictionary
        keyword1 = self.get_descriptor_key("key_aowfs")
        keyword2 = self.get_descriptor_key("key_oiwfs")
        keyword3 = self.get_descriptor_key("key_pwfs1")
        keyword4 = self.get_descriptor_key("key_pwfs2")
        
        # Get the values of the AOWFS, OIWFS, PWFS1 and PWFS2 probe states
        # keywords from the header of the PHU
        aowfs = dataset.phu_get_key_value(keyword1)
        oiwfs = dataset.phu_get_key_value(keyword2)
        pwfs1 = dataset.phu_get_key_value(keyword3)
        pwfs2 = dataset.phu_get_key_value(keyword4)
        
        if aowfs is None or oiwfs is None or pwfs1 is None or pwfs2 is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # If any of the probes are guiding, add them to the list
        wavefront_sensors = []
        if aowfs == "guiding":
            wavefront_sensors.append("AOWFS")
        if oiwfs == "guiding":
            wavefront_sensors.append("OIWFS")
        if pwfs1 == "guiding":
            wavefront_sensors.append("PWFS1")
        if pwfs2 == "guiding":
            wavefront_sensors.append("PWFS2")
        
        if len(wavefront_sensors) == 0:
            # If no probes are guiding, raise an exception
            raise Errors.CalcError()
        else:
            # Return a unique, sorted, wavefront sensor identifier string with
            # an ampersand separating each wavefront sensor name
            wavefront_sensors.sort
            ret_wavefront_sensor = str("&".join(wavefront_sensors))
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_wavefront_sensor, name="wavefront_sensor",
                                 ad=dataset)
        return ret_dv
    
    def wavelength_band(self, dataset, **args):
        # The wavelength band can only be obtained from data that has does not
        # have an AstroData Type of IMAGE (since there is no Gemini-level
        # lookup table to convert the filter name to a more standard band)
        if "IMAGE" not in dataset.types:
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
        else:
            raise Errors.DescriptorTypeError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_wavelength_band, name="wavelength_band",
                                 ad=dataset)
        return ret_dv
    
    def wavelength_reference_pixel(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_wavelength_reference_pixel = {}
        
        # Determine the reference pixel of the central wavelength keyword from
        # the global keyword dictionary 
        keyword = self.get_descriptor_key("key_wavelength_reference_pixel")
        
        # Get the value of the reference pixel of the central wavelength
        # keyword from the header of each pixel data extension as a dictionary
        # where the key of the dictionary is an ("*", EXTVER) tuple
        wavelength_reference_pixel_dict = gmu.get_key_value_dict(
          dataset, keyword)
        
        if wavelength_reference_pixel_dict is None:
            # The get_key_value_dict() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        wrp_dict = wavelength_reference_pixel_dict.iteritems()
        for ext_name_ver, raw_wavelength_reference_pixel in wrp_dict:
            if raw_wavelength_reference_pixel is None:
                wavelength_reference_pixel = None
            else:
                # Use the reference pixel of the central wavelength float as
                # the value
                wavelength_reference_pixel = float(
                  raw_wavelength_reference_pixel)
            
            ret_wavelength_reference_pixel.update(
              {ext_name_ver:wavelength_reference_pixel})
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_wavelength_reference_pixel,
                                 name="wavelength_reference_pixel", ad=dataset)
        return ret_dv
    
    def wcs_ra(self, dataset, **args):
        # This generic version finds the mid point of the data array
        # in ['SCI', 1] and uses that.
        # The WCS is taken from ['SCI', 1] if it exists, otherwise
        # we look in the PHU
        ext = dataset['SCI', 1]
        if ext.get_key_value('CTYPE1'):
            wcs = pywcs.WCS(ext.header)
        else:
            wcs = pywcs.WCS(dataset.phu.header)

        naxis1 = ext.get_key_value('NAXIS1')
        naxis2 = ext.get_key_value('NAXIS1')
        (x, y) = (0.5 * naxis1, 0.5 * naxis2)
        result = wcs.wcs_pix2sky([[x,y]], 1)
        ra = result[0][0]

        return DescriptorValue(ra, name="wcs_ra", ad=dataset)


    def wcs_dec(self, dataset, **args):
        # This generic version finds the mid point of the data array
        # in ['SCI', 1] and uses that.
        # The WCS is taken from ['SCI', 1] if it exists, otherwise
        # we look in the PHU
        ext = dataset['SCI', 1]
        if ext.get_key_value('CTYPE1'):
            wcs = pywcs.WCS(ext.header)
        else:
            wcs = pywcs.WCS(dataset.phu.header)
        naxis1 = ext.get_key_value('NAXIS1')
        naxis2 = ext.get_key_value('NAXIS1')
        (x, y) = (0.5 * naxis1, 0.5 * naxis2)
        result = wcs.wcs_pix2sky([[x,y]], 1)
        dec = result[0][1]

        return DescriptorValue(dec, name="wcs_dec", ad=dataset)

    # In the Generic GEMINI case, we trust the WCS (yeah... I know)
    # Specific instrument cases can over rise this of course
    # and the caller can still call wcs_ra/dec or target_ra/dec as required
    def ra(self, dataset, **args):
        return dataset.wcs_ra()

    def dec(self, dataset, **args):
        return dataset.wcs_dec()

    def well_depth_setting(self, dataset, **args):
        # The well_depth_setting descriptor is only specific to GNIRS and NIRI
        # data. The code below will be replaced with the GNIRS or NIRI specific
        # well_depth_setting descriptor function located in the instrument
        # specific descriptor files. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
