# This parameter file contains the parameters related to the primitives located
# in the primitives_photometry.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersPhotometry(ParametersBASE):
    addReferenceCatalog = {
        "suffix"                : "_refcatAdded",
        "radius"                : 0.067,
        "source"                : "gmos",
    }
    detectSources = {
        "suffix"                : "_sourcesDetected",
        "mask"                  : False,
        "replace_flags"         : 249,
        "set_saturation"        : False,
        "detect_minarea"        : 8,
        "detect_thresh"         : 2.0,
        "analysis_thresh"       : 2.0,
        "deblend_mincont"       : 0.005,
        "phot_min_radius"       : 3.5,
        "back_size"             : 32,
        "back_filtersize"       : 8,
    }
