from GSAOI_Keywords import GSAOI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class GSAOI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GSAOI_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)
