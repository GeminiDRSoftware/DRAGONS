# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi_image.py file, in alphabetical order.
from .parameters_gsaoi import ParametersGSAOI
from ..core.parameters_image import ParametersImage
from ..core.parameters_photometry import ParametersPhotometry

class ParametersGSAOIImage(ParametersGSAOI, ParametersImage, ParametersPhotometry):
    detectSources = {
        "suffix"                : "_sourcesDetected",
        "mask"                  : False,
        "replace_flags"         : 249,
        "set_saturation"        : False,
        "detect_minarea"        : 20,
        "detect_thresh"         : 5.0,
        "analysis_thresh"       : 5.0,
        "deblend_mincont"       : 0.005,
        "phot_min_radius"       : 1.0,
        "back_size"             : 128,
        "back_filtersize"       : 5,
    }
