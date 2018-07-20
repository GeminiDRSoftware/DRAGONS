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
    dark_lo_thresh = config.Field("Low rejection threshold for dark (ADU)", float, -20.)
    dark_hi_thresh = config.Field("High rejection threshold for dark (ADU)", float, 100.)
    flat_lo_thresh = config.RangeField("Low rejection threshold for normalized flat", float, 0.8, max=1.0)
    flat_hi_thresh = config.RangeField("High rejection threshold for normalized flat", float, 1.25, min=1.0)

    def validate(self):
        config.Config.validate(self)
        if self.dark_lo_thresh >= self.dark_hi_thresh:
            raise ValueError("dark_hi_thresh must be greater than dark_lo_thresh")

class makeLampFlatConfig(parameters_stack.core_stacking_config):
    pass

class removeFirstFrameConfig(config.Config):
    pass

class separateFlatsDarksConfig(config.Config):
    pass

class stackDarksConfig(parameters_stack.core_stacking_config):
    pass
