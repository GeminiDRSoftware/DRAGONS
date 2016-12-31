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