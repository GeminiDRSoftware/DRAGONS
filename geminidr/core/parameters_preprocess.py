# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersPreprocess(ParametersBASE):
    ADUToElectrons = {
        "suffix": "_ADUToElectrons"
    }
    associateSky = {
        "suffix"   : "_skyAssociated",
        "time"     : 600.,
        "distance" : 3.,
        "use_all"  : False,
    }
    correctBackgroundToReferenceImage = {
        "suffix"            : "_backgroundCorrected",
        "remove_zero_level" : False,
    }
    # No default type defined; the flat parameter could be a string
    # or an AstroData object
    divideByFlat = {
        "suffix"            : "_flatCorrected",
        "flat"              : None,
    }
    nonlinearityCorrect = {
        "suffix"            : "_nonlinearityCorrected",
    }
    normalizeFlat = {
        "suffix"            : "_normalized",
    }
    separateSky = {
        "suffix"            : "_skySeparated",
        "ref_obj"           : "",
        "ref_sky"           : "",
        "frac_FOV"          : 0.9,
    }
    subtractDark = {
        "suffix"            : "_darkCorrected",
        "dark"              : None,
    }
    subtractSky = {
        "suffix"            : "_skyCorrected",
    }
    thresholdFlatfield = {
        "suffix"            : "_thresholdFlatfielded",
        "upper"             : 10.0,
        "lower"             : 0.01,
    }
