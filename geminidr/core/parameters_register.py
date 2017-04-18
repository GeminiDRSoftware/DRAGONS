# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersRegister(ParametersBASE):
    correctWCSToReferenceFrame = {
        "suffix"            : "_wcsCorrected",
        "method"            : "sources",
        "fallback"          : None,
        "use_wcs"           : True,
        "first_pass"        : 10.0,
        "min_sources"       : 3,
        "cull_sources"      : False,
        "rotate"            : False,
        "scale"             : False,
    }
    # This primitive only sets the filename if you
    # ask it to correct the Astrometry
    determineAstrometricSolution = {
        "suffix"            : "_astrometryCorrected",
        "full_wcs"          : None,
    }
