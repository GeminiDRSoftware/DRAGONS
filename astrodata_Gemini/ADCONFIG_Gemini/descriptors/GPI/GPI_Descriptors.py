from GPI_Keywords import GPI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

from astrodata.interface.Descriptors import DescriptorValue

class GPI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GPI_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)

    # For GPI, the WCS is completely bogus, so for ra and dec we
    # reference the target_ra/dec. If the software gets fixed, this
    # should call the wcs_ra/dec functions if the ut_datetime is after
    # the date of the fix
    def ra(self, dataset, **args):
        return target_ra(offset=True, icrs=True)

    def dec(self, dataset, **args):
        return target_dec(offset=True, icrs=True)

    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Determine the two filter name keywords from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_filter")

        # Get the value of the two filter name keywords from the header of the
        # PHU
        filtername = dataset.phu_get_key_value(keyword)

        if filtername is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if stripID or pretty:
            # Strip the component ID from the two filter name values
            filtername = filtername[:filtername.find('_G')]

        if pretty:
            filtername = filtername.replace('IFSFILT_', '')

        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(filtername, name="filter_name",
                                 ad=dataset)
        return ret_dv

    def exposure_time(self, dataset, **args):
        # Determine the exposure time keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_exposure_time")

        # Get the value of the exposure time keyword from the header of the PHU
        exposure_time = dataset[1].get_key_value(keyword)

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


