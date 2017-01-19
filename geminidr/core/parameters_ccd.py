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
        "overscan_section"  : None,
    }
    subtractBias = {
        "suffix"            : "_biasCorrected",
        "bias"              : None,
    }
    subtractOverscan = {
        "suffix"            : "_overscanSubtracted",
        "overscan_section"  : None,
    }
    trimOverscan = {
        "suffix"            : "_overscanTrimmed",
    }
