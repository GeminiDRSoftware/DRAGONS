# This parameter file contains the parameters related to the primitives located
# in the primitives_visualze.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersVisualize(ParametersBASE):
    display = {
        "extname"           : "SCI",
        "frame"             : 1,
        "ignore"            : False,
        "remove_bias"       : False,
        "threshold"         : "auto",
        "tile"              : False,
        "zscale"            : True,
    }
    mosaicDetectors = {
        "suffix"            : "_mosaicked",
        "tile"              : False,
        "interpolate_gaps"  : False,
        "interpolator"      : "linear",

    }
    tileArrays = {
        "suffix"            : "_tiled",
        "tile_all"          : False,
    }