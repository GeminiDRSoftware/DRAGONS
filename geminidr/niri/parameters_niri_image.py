# This parameter file contains the parameters related to the primitives located
# in the primitives_niri_image.py file, in alphabetical order.
from .parameters_niri import ParametersNIRI
from ..core.parameters_image import ParametersImage
from ..core.parameters_photometry import ParametersPhotometry

class ParametersNIRIImage(ParametersNIRI, ParametersImage, ParametersPhotometry):
    detectSources = {
        "suffix"                : "_sourcesDetected",
        "mask"                  : False,
        "replace_flags"         : 249,
        "set_saturation"        : False,
        "detect_minarea"        : 40,
        "detect_thresh"         : 1.5,
        "analysis_thresh"       : 1.5,
        "deblend_mincont"       : 0.005,
        "phot_min_radius"       : 3.5,
        "back_size"             : 32,
        "back_filtersize"       : 3,
    }
