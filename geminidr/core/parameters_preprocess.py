# This parameter file contains the parameters related to the primitives located
# in the primitives_preprocess.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersPreprocess(ParametersBASE):
    addObjectMaskToDQ = {
        "suffix"            : "_objectMaskAdded",
    }
    ADUToElectrons = {
        "suffix"            : "_ADUToElectrons",
    }
    applyDQPlane = {
        "suffix"            : "_dqPlaneApplied",
        "replace_flags"     : 255,
        "replace_value"     : "median",
    }
    associateSky = {
        "suffix"            : "_skyAssociated",
        "time"              : 600.,
        "distance"          : 3.,
        "max_skies"         : None,
        "use_all"           : False,
    }
    correctBackgroundToReference = {
        "suffix"            : "_backgroundCorrected",
        "remove_background" : False,
    }
    darkCorrect = {
        "suffix"            : '_darkCorrected',
        "dark"              : None,
        "do_dark"           : True,
    }
    dilateObjectMask = {
        "suffix"            : '_objmaskDilated',
        "dilation"          : 1,
        "repeat"            : False,
    }
    flatCorrect = {
        "suffix"            : '_flatCorrected',
        "flat"              : None,
        "do_flat"           : True,
    }
    makeSky = {
        "max_skies"         : None,
    }
    nonlinearityCorrect = {
        "suffix"            : "_nonlinearityCorrected",
    }
    normalizeFlat = {
        "suffix"            : "_normalized",
        "scale"             : "median",
        "separate_ext"      : False,
    }
    separateSky = {
        "suffix"            : "_skySeparated",
        "ref_obj"           : "",
        "ref_sky"           : "",
        "frac_FOV"          : 0.9,
    }
    skyCorrect = {
        "apply_dq"          : True,
        "dilation"          : 2,
        "hsigma"            : 3.0,
        "lsigma"            : 3.0,
        "mask_objects"      : True,
        "mclip"             : True,
        "operation"         : "median",
        "reject_method"     : "sigclip",
        "reset_sky"         : False,
        "scale"             : True,
        "zero"              : False,
    }
    subtractSky = {
        "suffix"            : "_skyCorrected",
        "reset_sky"         : False,
        "scale"             : True,
        "sky"               : None,
        "zero"              : False,
    }
    subtractSkyBackground = {
        "suffix"            : "_skyBackgroundSubtracted",
    }
    thresholdFlatfield = {
        "suffix"            : "_thresholdFlatfielded",
        "upper"             : 10.0,
        "lower"             : 0.01,
    }
