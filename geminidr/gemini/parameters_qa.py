# This parameter file contains the parameters related to the primitives located
# in the primitives_qa.py file, in alphabetical order.
from gempy.library import config

class measureBGConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bgMeasured", optional=True)
    separate_ext = config.Field("Supply measurement for each extension?", bool, False)

class measureCCConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_ccMeasured", optional=True)

class measureIQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_iqMeasured", optional=True)
    display = config.Field("Display image with sources marked?", bool, False)
    separate_ext = config.Field("Supply measurement for each extension?", bool, False)
