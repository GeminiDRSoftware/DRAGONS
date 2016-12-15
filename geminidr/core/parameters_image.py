# This parameter file contains the parameters related to the primitives located
# in the primitives_image.py file, in alphabetical order.

from geminidr.core.parameters_register import ParametersRegister
from geminidr.core.parameters_resample import ParametersResample

class ParametersImage(ParametersRegister, ParametersResample):
    fringeCorrect = {
    }
    makeFringe = {
        "subtract_median_image" : None,
    }
    makeFringeFrame = {
        "suffix"                : "_fringe",
        "operation"             : "median",
        "reject_method"         : "avsigclip",
        "subtract_median_image" : True,
    }
    scaleByIntensity = {
        "suffix"                : "_scaled",
    }
    scaleFringeToScience = {
        "suffix"                : "_fringeScaled",
        "science"               : None,
        "stats_scale"           : False,
    }
    subtractFringe = {
        "suffix"                : "_fringeSubtracted",
        "fringe"                : None,
    }
