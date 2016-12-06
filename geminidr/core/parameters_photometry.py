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
        "max_sources"           : 50,
        "replace_flags"         : 249,
        "set_saturation"        : False,
    }

