# This parameter file contains the parameters related to the primitives
# define in the primitives_igrins.py file

from gempy.library import config

class selectFrameConfig(config.Config):
    frmtype = config.Field("frametype to filter", str)

class streamPatternCorrectedConfig(config.Config):
    # rpc_mode = config.Field("RP Correction mode", str, "guard")
    rpc_mode = config.Field("method to correct the pattern", str)
    suffix = config.Field("Readout pattern corrected", str, "_rpc")

class estimateNoiseConfig(config.Config):
    pass

class selectStreamConfig(config.Config):
    stream_name = config.Field("stream name for the output", str)

class addNoiseTableConfig(config.Config):
    pass

class setSuffixConfig(config.Config):
    suffix = config.Field("output suffix", str)

class somePrimitiveConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_suffix")
    param1 = config.Field("Param1", str, "default")
    param2 = config.Field("do param2?", bool, False)

class someStuffConfig(config.Config):
    suffix = config.Field("Output suffix", str, "_somestuff")
