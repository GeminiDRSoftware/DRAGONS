# This parameter file contains the parameters related to the primitives located
# in the primitives_nearIR.py file, in alphabetical order.
from gempy.library import config, astrotools as at
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
    suffix = config.Field("Filename suffix", str, "_lampstack", optional=True)

class removeFirstFrameConfig(config.Config):
    remove_first = config.Field("Remove first frame?", bool, True)
    remove_files = config.ListField("List of files to remove", str, None, optional=True)

class removePatternNoiseConfig(config.Config):

    def validate_regions(value, multiple=True):
        ranges = at.parse_user_regions(value, dtype=int, allow_step=False)
        return multiple or len(ranges) == 1

    suffix = config.Field("Filename suffix", str, "_patternNoiseRemoved", optional=True)
    must_reduce_rms = config.Field("Require reduction in RMS to apply pattern removal?", bool, True)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    pattern_x_size = config.RangeField("Pattern x size (pixels)", int, 16, min=4)
    pattern_y_size = config.RangeField("Pattern y size (pixels)", int, 4, min=4)
    subtract_background = config.Field("Subtract median from each pattern box?", bool, True)
    region = config.Field("Rows to remove pattern in, e.g. 'y1:y2,y3:y4' etc.",
                          str, '0:1024', optional=True, check=validate_regions)

class separateFlatsDarksConfig(config.Config):
    pass
