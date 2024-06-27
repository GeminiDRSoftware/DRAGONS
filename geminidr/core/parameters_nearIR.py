# This parameter file contains the parameters related to the primitives located
# in the primitives_nearIR.py file, in alphabetical order.
from gempy.library import config
from . import parameters_stack, parameters_standardize

def powerof2(value):
    """pattern boxes must be a power of 2"""
    return value > 0 and (value & (value - 1) == 0)


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


class cleanReadoutConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_readoutCleaned", optional=True)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    pattern_x_size = config.Field("Pattern x size (pixels)", int, 16, check=powerof2)
    pattern_y_size = config.Field("Pattern y size (pixels)", int, 4, check=powerof2)
    debug_subtract_background = config.Field("Subtract median from each pattern box?", bool, True)
    level_bias_offset = config.Field("Level the bias offset across (sub-)quads accompanying pattern noise?", bool, True)
    smoothing_extent = config.RangeField("Width (in pix) of the region at a given quad interface to be smoothed over", int, 5, min=1)
    intraquad_smooth = config.RangeField("Height (in pix) of the region at a bias jump to be smoothed over", int, 50, min=1)
    sg_win_size = config.RangeField("Smoothing window size (pixels) for Savitzky-Golay filter", int, 25, min=3)
    simple_thres = config.RangeField("Pattern edge detection threshold", float, 0.6, min=0.0)
    pat_strength_thres = config.RangeField("Pattern strength threshold", float, 15.0, min=0.0)
    clean = config.ChoiceField("Cleaning behavior", str,
                               allowed={"default": "perform pattern removal if pattern in strong enough",
                                        "force": "force pattern removal",
                                        "skip": "skip primitive"},
                               default="default", optional=False)
    debug_canny_sigma = config.RangeField("Standard deviation for smoothing of Canny edge-finding", float, 3, min=1)


class cleanFFTReadoutConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_readoutFFTCleaned", optional=True)
    win_size = config.RangeField("Window size for Fourier amplitude spectrum thresholding", int, 9, min=3)
    periodicity = config.Field("Periodicity of the pattern noise (pixels)", int, 8)
    sigma_fact = config.RangeField("Sigma factor used for the Fourier amplitude threshold", float, 4., min=0)
    pat_thres = config.RangeField("Standardized pattern strength threshold", float, -0.3, max=1.0)
    lquad = config.Field("Level the bias offset across (sub-)quads accompanying pattern noise?", bool, True)
    l2clean = config.Field("Clean Fourier artifacts?", bool, True)
    l2thres = config.RangeField("Sigma factor to be used in thresholding for l2clean", float, 4., min=0)
    smoothing_extent = config.RangeField("Width (in pix) of the region at a given quad interface to be smoothed over", int, 5, min=1)
    pad_rows = config.Field("Number of dummy rows to append to the top quads of the image", int, 0)
    clean = config.ChoiceField("Cleaning behavior", str,
                               allowed={"default": "perform pattern removal if pattern in strong enough",
                                        "skip": "skip primitive"},
                               default="skip", optional=False)
    


class separateFlatsDarksConfig(config.Config):
    pass
