# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersStack(ParametersBASE):
    alignAndStack = {
    }
    stackFlats = {
        "suffix"            : "_stack",
        "apply_dq"          : True,
        "nhigh"             : 1,
        "nlow"              : 1,
        "operation"         : "median",
        "reject_method"     : "minmax",
    }
    stackFramesOld = {
        "suffix"            : "_stack",
        "apply_dq"          : True,
        "nhigh"             : 1,
        "nlow"              : 1,
        "operation"         : "average",
        "reject_method"     : "avsigclip",
        "remove_background" : False,
        "zero"              : True,
    }
    stackFrames = {
        "suffix"            : "_stack",
        "apply_dq"          : True,
        "operation"         : "mean",
        "reject_method"     : "varclip",
        "remove_background" : False,
        "scale"             : False,
        "separate_ext"      : True,
        "statsec"           : None,
        "zero"              : False,
    }
    stackSkyFrames = {
        "suffix"            : "_skyStacked",
        "apply_dq"          : True,
        "dilation"          : 2,
        "hsigma"            : 3.0,
        "lsigma"            : 3.0,
        "mask_objects"      : True,
        "mclip"             : True,
        "operation"         : "median",
        "reject_method"     : "sigclip",
        "scale"             : True,
        "zero"              : False,
    }