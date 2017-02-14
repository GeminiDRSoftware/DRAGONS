# This parameter file contains the parameters related to the primitives located
# in the primitives_f2.py file, in alphabetical order.
from .parameters_f2 import ParametersF2
from ..core.parameters_image import ParametersImage
from ..core.parameters_photometry import ParametersPhotometry

class ParametersF2Image(ParametersF2, ParametersImage, ParametersPhotometry):
    addDQ = {
        "suffix"            : "_dqAdded",
        "bpm"               : None,
        "illum_mask"        : True,
    }
    associateSky = {
        "suffix"            : "_skyAssociated",
        "distance"          : 1.,
        "time"              : 600.,
        "max_skies"         : None,
        "use_all"           : False,
    }
    makeLampFlat = {}
    stackSkyFrames = {
        "suffix"            : "_skyStacked",
        "mask"              : True,
        "nhigh"             : 1,
        "nlow"              : 0,
        "operation"         : "median",
        "reject_method"     : "minmax",
    }