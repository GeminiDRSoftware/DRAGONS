# This parameter file contains the parameters related to the primitives located
# in the primitives_qa.py file, in alphabetical order.
from gempy.library import config

class measureBGConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bgMeasured")
    remove_bias = config.Field("Remove bias level?", bool, False)
    separate_ext = config.Field("Supply measurement for each extension?", bool, False)

class measureCCConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_ccMeasured")

class measureIQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_iqMeasured")
    display = config.Field("Display image with sources marked?", bool, False)
    remove_bias = config.Field("Remove bias level before displaying?", bool, False)
    separate_ext = config.Field("Supply measurement for each extension?", bool, False)
