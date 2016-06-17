# This parameter file contains the parameters related to the primitives located
# in the primitives_IMAGE.py file, in alphabetical order.

from parameters_GMOS import ParametersGMOS

class ParametersIMAGE(ParametersGMOS):
    detectSources= {
        "set_saturation"    : False,
    }
    makeFringe = {
        "subtract_median_image" : False,
    }
    makeFringeFrame = {
        "suffix"            : "_fringe",
        "operation"         : "median",
        "reject_method"     : "avsigclip",
        "subtract_median_image" : True,
    }
    normalizeFlat ={
        "suffix"            : "_normalized",
    }
    scaleByIntensity ={
        "suffix"            : "_scaled",
    }
    # No default type for science parameter could be a
    # string or an AstroData object
    scaleFringeToScience ={
        "suffix"            : "_fringeScaled",
        "science"           : None,
        "stats_scale"       : False,
    }
    showParameters ={
        "suffix"            : "_scafaasled",
    }
    stackFlats ={
        "suffix"            : "_stack",
        "mask"              : True
        "operation"         : "median",
        "reject_method"     : "minmax"
    }
    # The standardizeStructure primitive is actually located in the
    # primtives_GMOS.py file, but the attach_mdf parameter should be set to False
    # as default for data with an AstroData Type of IMAGE.
    # No default type for mdf parameter; could be a string or
    # an AstroData object
    standardizeStructure = {
        "suffix"            : "_structureStandardized",
        "attach_mdf"        : False,
        "mdf"               : None,
    }
    storeProcessedFringe = {
        "suffix"            : "_fringe",
    }
    # No default type defined, since the fringe parameter could be a string
    # or an AstroData object
    subtractFringe = {
        "suffix"            : "_fringeSubtracted",
        "fringe"            : None,
    }
