import re

from astrodata.Calculator import Calculator
from StandardGenericKeyDict import stdkeyDictGeneric

class Generic_DescriptorCalc(Calculator):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = stdkeyDictGeneric
