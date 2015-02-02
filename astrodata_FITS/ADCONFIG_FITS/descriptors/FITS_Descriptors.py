from astrodata.interface.Descriptors import Calculator
from FITS_Keywords import FITS_KeyDict

class FITS_DescriptorCalc(Calculator):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = FITS_KeyDict
