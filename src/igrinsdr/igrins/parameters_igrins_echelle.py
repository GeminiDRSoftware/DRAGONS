# This parameter file contains the parameters related to the primitives
# define in the primitives_igrins_echelle.py file

from gempy.library import config

class myNewPrimitiveConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_suffix")
    param1 = config.Field("Param1", str, "default")
    param2 = config.Field("do param2?", bool, False)
