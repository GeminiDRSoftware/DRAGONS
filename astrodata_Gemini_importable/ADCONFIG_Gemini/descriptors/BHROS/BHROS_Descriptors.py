# new BHROS descriptors from trunk @Rev4997

from astrodata.utils import Errors
from astrodata.interface import Descriptors
from astrodata.interface.Descriptors import DescriptorValue

from GEMINI_Descriptors import GEMINI_DescriptorCalc
import GemCalcUtil

class BHROS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
    
    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):

        # For this approach, the central wavelength is recorded in
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
            # return the central wavelength in the default units of meters.
            output_units = "meters"

        keyword = 'WAVELENG'

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

    def disperser(self, dataset, stripID=False, pretty=False, **args):

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue('bHROS', name="disperser", ad=dataset)

        return ret_dv

