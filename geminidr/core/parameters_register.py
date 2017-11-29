# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersRegister(ParametersBASE):
    matchWCSToReference = {
        "suffix"            : "_wcsCorrected",
        "method"            : "sources",
        "fallback"          : None,
        "use_wcs"           : True,
        "first_pass"        : 5.0,
        "min_sources"       : 3,
        "cull_sources"      : False,
        "rotate"            : False,
        "scale"             : False,
    }
    determineAstrometricSolution = {
        "suffix"            : "_astrometryCorrected",
        "full_wcs"          : None,
        "initial"           : 5.0,
        "final"             : 1.0,
    }
