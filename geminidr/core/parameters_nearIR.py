# This parameter file contains the parameters related to the primitives located
# in the primitives_nearIR.py file, in alphabetical order.
from gempy.library import config
from . import parameters_stack, parameters_standardize

class addLatencyToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_latencyAdded", optional=True)
    non_linear = config.Field("Flag non-linear pixels?", bool, False)
    time = config.RangeField("Persistence time (seconds)", float, 120., min=0.)

class addDQConfig(parameters_standardize.addDQConfig, addLatencyToDQConfig):
    latency = config.Field("Apply latency for saturated pixels?", bool, False)

class makeBPMConfig(config.Config):
    pass

class makeLampFlatConfig(parameters_stack.core_stacking_config):
    def setDefaults(self):
        self.reject_method = "none"
        self.separate_ext = False

class removeFirstFrameConfig(config.Config):
    pass

class separateFlatsDarksConfig(config.Config):
    pass

class stackDarksConfig(parameters_stack.core_stacking_config):
    pass
