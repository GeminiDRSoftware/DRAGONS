import re

from astrodata.Calculator import Calculator
from StandardGenericKeyDict import stdkeyDictGeneric

class Generic_DescriptorCalc(Calculator):
    # Saving the dict used to update the specific key 
    #   dictionary for header lookup
    _update_stdkey_dict = stdkeyDictGeneric
