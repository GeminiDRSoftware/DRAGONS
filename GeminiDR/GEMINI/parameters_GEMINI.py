# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersGemini(ParametersCORE):
    addDQ = {
        "suffix"    : "_dqAdded",
        "bpm"       : None,
        "illum_mask": False,
    }
    addMDF = {
        "suffix"    : "_mdfAdded",
        "mdf"       : None,
    }
    mosaicADdetectors = {
        "suffix"    : "_mosaicAD",
        "tile"      : False,
        # If True, transforms the DQ image bit_plane by bit_plane.
        "dq_planes" : False,
    }
    addObjectMaskToDQ = {
        "suffix"    : "_objectMaskAdded"
    }
    addReferenceCatalog = {
        "suffix"    : "_refcatAdded",
        "radius"    : 0.067,
        "source"    : "gmos",
    }
    addToList = {
        "purpose": None,
    }
    addVAR = {
        "suffix"       : "_varAdded",
        "read_noise"   : False,
        "poisson_noise": False,
    }
    ADUToElectrons = {
        "suffix": "_ADUToElectrons"
    }
    alignAndStack = {
        "check_if_stack": False,
    }
    alignToReferenceFrame = {
        "suffix"       : "_align",
        "interpolator" : "linear",
        "trim_data"    : False,
    }
    applyDQPlane = {
        "suffix"        : "_dqPlaneApplied",
        "replace_flags" : 255,
        "replace_value" : "median",
    }
    associateSky = {
        "suffix"   : "_skyAssociated",
        "time"     : 600.,
        "distance" : 3.,
        "use_all"  : False,
    }
    contextReport = {
        "report_history"    : False,
        "internal_dict"     : False,
        "context_vars"      : False,
        "report_inputs"     : False,
        "report_parameters" : False,
        "showall"           : False,
    }
    correctBackgroundToReferenceImage = {
        "suffix"            : "_backgroundCorrected",
        "remove_zero_level" : False,
    }

    # This primitive only sets the filename if you
    # ask it to correct the Astrometry
    determineAstrometricSolution = {
        "suffix"            : "_astrometryCorrected",
    }
    correctWCSToReferenceFrame = {
        "suffix"            : "_wcsCorrected",
        "method"            : "sources",
        "fallback"          : None,
        "use_wcs"           : True,
        "first_pass"        : 10.0,
        "min_sources"       : 3,
        "cull_sources"      : False,
        "rotate"            : False,
        "scale"             : False,
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
    display = { 
        "extname"           : "SCI",
        "frame"             : 1,
        "ignore"            : False,
        "remove_bias"       : False,
        "threshold"         : "auto",
        "tile"              : False,
        "zscale"            : True,
    }
    thresholdFlatfield = {
        "suffix"            : "_thresholdFlatfielded",
        "upper"             : 10.0,
        "lower"             : 0.01,
    }
    # No default type defined; the flat parameter could be a string
    # or an AstroData object
    divideByFlat = {
        "suffix"            : "_flatCorrected",
        "flat"              : None,
    }
    getCalibration = {
        "source"            : "all",
        "caltype"           : None,
    }
    getList = {
        "purpose"           : None,
    }
    markAsPrepared = {
        "suffix"            : "_prepared",
    }
    measureBG = {
        "suffix"            : "_bgMeasured",
        "remove_bias"       : True,
        "separate_ext"      : False,
    }
    measureCC = {
        "suffix"            : "_ccMeasured",
    }
    measureCCAndAstrometry = {
        "suffix"            : "_ccAndAstrometryMeasured",
        "correct_wcs"       : False,
    }
    measureIQ = {
        "suffix"            : "_iqMeasured",
        "display"           : False,
        "remove_bias"       : False,
        "separate_ext"      : False,
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
    showList = {
        "purpose"           : None,
    }

    # BEGIN showParameters -- No idea what to do with this.
    showParameters = {
        "test":{
            "default"       : True,
            "type"          : "bool",
            "recipeOverride": True,
            "userOverride"  : False, #True,
            "uiLevel"       : "debug",
            "tags"          : ["test", "iraf"],
        },
        "otherTest":{
            "default"       : False,
            "userOverride"  : True,
        },
        "otherTest2":{
            "userOverride"  : True,
            "tags"          :["test", "wcslib"],
        },
    }
    # END showParameters

    sleep = {
        "duration"          : 5.0,
    }
    stackFrames = {
        "suffix"            : "_stack",
        "mask"              : True,
        "nhigh"             : 1,
        "nlow"              : 1,
        "operation"         : "average",
        "reject_method"     : "avsigclip",
    }
    stackSkyFrames = {
        "suffix"            : "_skyStacked",
        "mask"              : True,
        "nhigh"             : 1,
        "nlow"              : 1,
        "operation"         : "median",
        "reject_method"     : "avsigclip",
    }
    standardizeGeminiHeaders = {
        "suffix"            : "_geminiHeadersStandardized",
    }
    storeProcessedArc = {
        "suffix"            : "_arc",
    }
    storeProcessedBias = {
        "suffix"            : "_bias",
    }
    storeProcessedDark = {
        "suffix"            : "_dark",
    } 
    storeProcessedFlat = {
        "suffix"            : "_flat",
    }

    # No default type defined, since the dark parameter could be a string
    # or an AstroData object
    subtractDark = {
        "suffix"            : "_darkCorrected",
        "dark"              : None,
    }
    subtractSky = {
        "suffix"            : "_skyCorrected",
    }
    updateWCS = {
        "suffix"            : "_wcsUpdated",
    }
    writeOutputs = {
        "suffix"            : None,
        "strip"             : False,
        "prefix"            : None,
        "outfilename"       : None,
    }
    traceFootprints = {
        "function"          : "polynomial",
        "order"             : 2,
        "trace_threshold"   : 1,                    # enhance_edges function.
        "suffix"            : "_tracefootprints",
    }
    cutFootprints = {
        "suffix"            : "_cutfootprints",
    }

    # No default type defined, since the arc parameter could be a string or
    # an AstroData object
    attachWavelengthSolution = {
        "suffix"            : "_wavelengthSolution",
        "arc"               : None,
    }
