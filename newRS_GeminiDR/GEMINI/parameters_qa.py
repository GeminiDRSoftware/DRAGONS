# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersQA(ParametersCORE):
    measureBG = {
        "suffix"            : "_bgMeasured",
        "remove_bias"       : True,
        "separate_ext"      : False,
    }
    measureCC = {
        "suffix"            : "_ccMeasured",
    }
    measureCCAndAstrometry = {
        "suffix"            : "_ccAndAstrometryMeasured",
        "correct_wcs"       : False,
    }
    measureIQ = {
        "suffix"            : "_iqMeasured",
        "display"           : False,
        "remove_bias"       : False,
        "separate_ext"      : False,
    }
