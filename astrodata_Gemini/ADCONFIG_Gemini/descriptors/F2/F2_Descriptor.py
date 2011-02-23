from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardF2KeyDict import stdkeyDictF2
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class F2_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictF2)
    
    def __init__(self):
        pass

