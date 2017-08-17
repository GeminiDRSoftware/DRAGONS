# This parameter file contains the parameters related to the primitives located
# in the primitives_standardize.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersStandardize(ParametersBASE):
    addDQ = {
        "suffix"            : "_dqAdded",
        "bpm"               : None,
        "illum_mask"        : False,
    }
    addIllumMaskToDQ = {
        "mask"              : None,
        "suffix"            : "_illumMaskAdded",
    }
    addMDF = {
        "suffix"            : "_mdfAdded",
        "mdf"               : None,
    }
    addVAR = {
        "suffix"            : "_varAdded",
        "read_noise"        : False,
        "poisson_noise"     : False,
    }
    prepare = {
        "suffix"            : "_prepared",
    }
    standardizeInstrumentHeaders = {
        "suffix"            : "_instrumentHeadersStandardized",
    }
    standardizeObservatoryHeaders = {
        "suffix"            : "_observatoryHeadersStandardized",
    }
    standardizeStructure = {
        "suffix"            : "_structureStandardized",
        "attach_mdf"        : True,
        "mdf"               : None,
    }
    validateData = {
        "suffix"            : "_dataValidated",
        "num_exts"          : None,
        "repair"            : False,
    }
