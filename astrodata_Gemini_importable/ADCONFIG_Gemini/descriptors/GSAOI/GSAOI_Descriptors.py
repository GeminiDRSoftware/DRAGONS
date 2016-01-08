from GSAOI_Keywords import GSAOI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc
from astrodata.interface.Descriptors import DescriptorValue

class GSAOI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GSAOI_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)

    def wcs_ra(self, dataset, **args):
        # Return the RA derived from the WCS
        # This GMOS version simply returns the reference value for each coordinate
        # from the WCS. This is simplistic, but very robust and is good for GSAOI
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
        # from the WCS. This is simplistic, but very robust and is good for GSAOI
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

