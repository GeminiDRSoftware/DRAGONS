# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_nodandshuffle.py file, in alphabetical order.
from gempy.library import config

class skyCorrectNodAndShuffleConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyCorrected", optional=True)
