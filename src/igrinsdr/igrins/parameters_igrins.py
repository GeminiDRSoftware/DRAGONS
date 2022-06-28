# This parameter file contains the parameters related to the primitives
# define in the primitives_igrins.py file

from gempy.library import config

class somePrimitiveConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_suffix")
    param1 = config.Field("Param1", str, "default")
    param2 = config.Field("do param2?", bool, False)

class someStuffConfig(config.Config):
    suffix = config.Field("Output suffix", str, "_somestuff")
