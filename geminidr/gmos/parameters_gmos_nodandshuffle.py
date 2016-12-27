# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_nodandshuffle.py file, in alphabetical order.

from .parameters_gmos import ParametersGMOS

class ParametersGMOSNodAndShuffle(ParametersGMOS):
    skyCorrectNodAndShuffle = {
        "suffix"                : "_skyCorrected",
    }
