# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_longslit.py file, in alphabetical order.

from .parameters_gmos_spect import ParametersGMOSSpect
from .parameters_gmos_nodandshuffle import ParametersGMOSNodAndShuffle

class ParametersGMOSLongslit(ParametersGMOSSpect, ParametersGMOSNodAndShuffle):
    pass