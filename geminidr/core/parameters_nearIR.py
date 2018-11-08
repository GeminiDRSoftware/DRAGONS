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
    override_thresh = config.ChoiceField("Apply user-specified thresholds, overriding any default calculation?", bool, { True : 'Must be True where no default algorithm is implemented' }, default=True, optional=False)
    dark_lo_thresh = config.Field("Low rejection threshold for dark (ADU)", float, None, optional=True)
    dark_hi_thresh = config.Field("High rejection threshold for dark (ADU)", float, None, optional=True)
    flat_lo_thresh = config.RangeField("Low rejection threshold for normalized flat", float, None, max=1.0, optional=True)
    flat_hi_thresh = config.RangeField("High rejection threshold for normalized flat", float, None, min=1.0, optional=True)

    def validate(self):
        config.Config.validate(self)
        if self.dark_lo_thresh is not None and \
           self.dark_hi_thresh is not None and \
           self.dark_lo_thresh >= self.dark_hi_thresh:
            raise ValueError("dark_hi_thresh must be greater than dark_lo_thresh")

class makeLampFlatConfig(parameters_stack.core_stacking_config):
    pass

class removeFirstFrameConfig(config.Config):
    pass

class separateFlatsDarksConfig(config.Config):
    pass

class stackDarksConfig(parameters_stack.core_stacking_config):
    pass
