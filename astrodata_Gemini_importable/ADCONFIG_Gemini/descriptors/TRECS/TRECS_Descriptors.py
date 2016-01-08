
from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.Descriptors import DescriptorValue

import GemCalcUtil

from TRECS_Keywords import TRECS_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class TRECS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = TRECS_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):
        # For TRECS data, the central wavelength is recorded in
        # micrometers - it's actually hardwired by disperser below.
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
            # return the central wavelength in the default units of meters
            output_units = "meters"
        
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
        
        if disperser == "LowRes-10":
            central_wavelength = 10.5
        elif disperser == "LowRes-20":
            central_wavelength = 20.0
        
        # Sometimes the header value takes the form "HighRes-10 + 230" or
        # similar
        elif disperser.startswith("HighRes-10"):
            
            # There is a HRCENWL keyword that contains the central wavelength
            # only if we are using the HighRes-10 grating
            central_wavelength = dataset.phu_get_key_value(
                self.get_descriptor_key("key_central_wavelength"))
        
        elif disperser == "Mirror":
            raise Errors.DescriptorTypeError()
        else:
            raise Errors.CalcError()
        
        # Use the utilities function convert_units to convert the central
        # wavelength value from the input units to the output units
        ret_central_wavelength = GemCalcUtil.convert_units(
            input_units=input_units, input_value=central_wavelength,
            output_units=output_units)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_central_wavelength,
                                 name="central_wavelength", ad=dataset)
        return ret_dv
    
    def dispersion(self, dataset, asMicrometers=False, asNanometers=False,
                   asAngstroms=False, **args):
        # Currently for TRECS data, the dispersion is recorded in meters (?)
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
        
        # Determine the dispersion keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_dispersion")
        
        # Get the value of the dispersion keyword from the header of the PHU
        raw_dispersion = dataset.phu_get_key_value(keyword)
        
        if raw_dispersion is None:
            # The get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if disperser == "LowRes-10":
            dispersion = 0.022
        elif disperser == "LowRes-20":
            dispersion = 0.033
        elif disperser == "Mirror":
            raise Errors.DescriptorTypeError()
        else:
            raise Errors.CalcError()
        
        # Use the utilities function convert_units to convert the dispersion
        # value from the input units to the output units
        ret_dispersion = GemCalcUtil.convert_units(
            input_units=input_units, input_value=dispersion,
            output_units=output_units)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_dispersion, name="dispersion", ad=dataset)
        
        return ret_dv
    
    def gain(self, dataset, **args):
        # Determine the bias value keyword (biaslevel) from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biaslevel = dataset.phu_get_key_value(keyword)
        
        if biaslevel is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if biaslevel == "2":
            ret_gain = 214.0
        elif biaslevel == "1":
            ret_gain = 718.0
        else:
            Errors.CalcError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain, name="gain", ad=dataset)
        
        return ret_dv
    
    def pixel_scale(self, dataset, **args):
        # Return the pixel scale float
        ret_pixel_scale = float(0.089)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    def read_mode(self, dataset, **args):
        # For TRECS data, raise an exception if the read_mode descriptor
        # called, since it is not relevant for TRECS data.
        raise Errors.ExistError()
