# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersPhotometry(ParametersCORE):
    addReferenceCatalog = {
        "suffix"    : "_refcatAdded",
        "radius"    : 0.067,
        "source"    : "gmos",
    }
    detectSources = {
        "suffix"            : "_sourcesDetected",
        "centroid_function" : "moffat",
        "fwhm"              : None,
        "mask"              : False,
        "max_sources"       : 50,
        "method"            : "sextractor",
        "sigma"             : None,
        "threshold"         : 3.0,
        "set_saturation"    : False,
    }
    measureCCAndAstrometry = {
        "suffix"            : "_ccAndAstrometryMeasured",
        "correct_wcs"       : False,
    }
