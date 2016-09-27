# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersDisplay(ParametersCORE):
    display = { 
        "extname"           : "SCI",
        "frame"             : 1,
        "ignore"            : False,
        "remove_bias"       : False,
        "threshold"         : "auto",
        "tile"              : False,
        "zscale"            : True,
    }
