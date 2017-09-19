# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersCCD(ParametersBASE):
    biasCorrect = {
        "suffix"            : "_biasCorrected",
        "bias"              : None,
    }
    overscanCorrect = {
        "suffix"            : "_overscanCorrected",
        "niterate"          : 2,
        "high_reject"       : 3.0,
        "low_reject"        : 3.0,
        "function"          : "spline",
        "nbiascontam"       : None,
        "order"             : None,
    }
    subtractBias = {
        "suffix"            : "_biasCorrected",
        "bias"              : None,
    }
    subtractOverscan = {
        "suffix"            : "_overscanSubtracted",
        "niterate"          : 2,
        "high_reject"       : 3.0,
        "low_reject"        : 3.0,
        "function"          : "spline",
        "nbiascontam"       : None,
        "order"             : None,
    }
    trimOverscan = {
        "suffix"            : "_overscanTrimmed",
    }
