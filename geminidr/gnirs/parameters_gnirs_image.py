# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_image.py file, in alphabetical order.
from .parameters_gnirs import ParametersGNIRS
from ..core.parameters_image import ParametersImage
from ..core.parameters_photometry import ParametersPhotometry

class ParametersGNIRSImage(ParametersGNIRS, ParametersImage, ParametersPhotometry):
    correctWCSToReferenceFrame = {
        "suffix"                : "_wcsCorrected",
        "method"                : "header",
        "fallback"              : None,
        "use_wcs"               : False,
        "first_pass"            : 2.0,
        "min_sources"           : 1,
        "cull_sources"          : True,
        "rotate"                : True,
        "scale"                 : False,
    }
    detectSources = {
        "suffix"                : "_sourcesDetected",
        "mask"                  : False,
        "replace_flags"         : 249,
        "set_saturation"        : False,
        "detect_minarea"        : 40,
        "detect_thresh"         : 3.0,
        "analysis_thresh"       : 1.5,
        "deblend_mincont"       : 0.001,
        "phot_min_radius"       : 3.5,
        "back_size"             : 32,
        "back_filtersize"       : 3,
    }
    determineAstrometricSolution = {
        "suffix"            : "_astrometryCorrected",
        "full_wcs"          : None,
        "initial"           : 15.0,
        "final"             : 1.0,
    }
    stackSkyFrames = {
        "suffix"            : "_skyStacked",
        "mask"              : True,
        "nhigh"             : 1,
        "nlow"              : 0,
        "operation"         : "median",
        "reject_method"     : "avsigclip",
    }
    standardizeStructure = {
        "suffix"                : "_structureStandardized",
        "attach_mdf"            : False,
        "mdf"                   : None,
    }