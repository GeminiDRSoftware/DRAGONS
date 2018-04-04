# This parameter file contains the parameters related to the primitives located
# in the primitives_nearIR.py file, in alphabetical order.
from gempy.library import config
from . import parameters_standardize

class addLatencyToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_latencyAdded")
    non_linear = config.Field("Flag non-linear pixels?", bool, False)
    time = config.RangeField("Persistence time (seconds)", float, 120., min=0.)

class addDQConfig(addLatencyToDQConfig, parameters_standardize.addDQConfig):
    pass

class makeBPMConfig(config.Config):
    pass

class makeLampFlatConfig(config.Config):
    pass

class lampOnLampOffConfig(config.Config):
    pass

class removeFirstFrameConfig(config.Config):
    pass

class separateFlatsDarksConfig(config.Config):
    pass

class stackDarksConfig(config.Config):
    pass

class thermalEmissionCorrectConfig(config.Config):
    pass
