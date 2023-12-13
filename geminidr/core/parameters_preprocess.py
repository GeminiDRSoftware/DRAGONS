# This parameter file contains the parameters related to the primitives located
# in the primitives_preprocess.py file, in alphabetical order.
from gempy.library import config
from . import parameters_stack
from astrodata import AstroData
from geminidr.core import parameters_generic


def replace_valueCheck(value):
    """validate applyDQPlane.replace_value"""
    return (value in ('mean', 'median') or isinstance(value, float))


class addObjectMaskToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_objectMaskAdded", optional=True)


class ADUToElectronsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_ADUToElectrons", optional=True)


class applyDQPlaneConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dqPlaneApplied", optional=True)
    replace_flags = config.RangeField("DQ bitmask (uint16) for pixels to replace.", int, 65529, min=0)
    replace_value = config.Field("Replacement value [mean|median|<value>]",
                                 (str, float), "median", check=replace_valueCheck)
    inner = config.RangeField("Inner filter radius (pixels)", float, None, min=1, optional=True)
    outer = config.RangeField("Outer filter radius (pixels)", float, None, min=1, optional=True)
    max_iters = config.RangeField("Maximum number of iterations", int, 1, min=1)

    def validate(self):
        config.Config.validate(self)
        if self.inner is not None and self.outer is not None and self.outer < self.inner:
            raise ValueError("Outer radius must be larger than inner radius")


class associateSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyAssociated", optional=True)
    time = config.RangeField("Maximum association time (seconds)", float, 600., min=0)
    distance = config.RangeField("Minimum association distance (arcsec)", float, 3., min=0)
    min_skies = config.RangeField("Minimum number of sky frames to associate",
                             int, 3, min=0, optional=True)
    max_skies = config.RangeField("Maximum number of sky frames to associate",
                             int, None, min=1, optional=True)
    use_all = config.Field("Use all frames as skies?", bool, False)
    sky = config.ListField("Sky-only frames stored on disk", (AstroData, str),
                           None, optional=True, single=True)


class correctBackgroundToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_backgroundCorrected", optional=True)
    separate_ext = config.Field("Treat each extension separately?", bool, True)
    remove_background = config.Field("Remove background level?", bool, False)


class darkCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_darkCorrected", optional=True)
    dark = config.ListField("Dark frame", (str, AstroData), None, optional=True, single=True)


class dilateObjectMaskConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_objmaskDilated", optional=True)
    dilation = config.RangeField("Dilation radius (pixels)", float, 1., min=0)
    repeat = config.Field("Allow dilation of already-dilated image?", bool, False)


class fixPixelsConfig(config.Config):
    suffix = config.Field("Filename suffix", dtype=str,
                          default="_pixelsFixed", optional=True)
    regions = config.Field('Regions to fix, e.g. "450,521; 430:437,513:533"',
                           dtype=str, optional=True)
    regions_file = config.Field("Path to a file containing the regions to fix",
                                dtype=str, optional=True)
    axis = config.Field("Interpolation axis. 1 is x-axis, 2 is y-axis, 3 is z-axis. If left undefined, use the narrowest region dimension.",
                        dtype=int, optional=True)
    use_local_median = config.Field("Use a local median filter for single pixels?",
                                    dtype=bool, default=False, optional=True)
    debug = config.Field("Display regions?", dtype=bool, default=False)


class flatCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_flatCorrected", optional=True)
    flat = config.ListField("Flatfield frame", (str, AstroData), None, optional=True, single=True)

class nonlinearityCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_nonlinearityCorrected", optional=True)


class normalizeFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    scale = config.ChoiceField("Statistic for scaling", str,
                               allowed = {"mean": "Scale by mean",
                                          "median": "Scale by median"},
                               default="median")
    separate_ext = config.Field("Scale extensions separately?", bool, False)


class separateSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skySeparated", optional=True)
    ref_obj = config.Field("Manually-assigned object files", str, None, optional=True)
    ref_sky = config.Field("Manually-assigned sky files", str, None, optional=True)
    frac_FOV = config.RangeField("Field of view scaling for coaddition", float, 0.9, min=0.5, max=1)


class makeSkyConfig(associateSkyConfig, separateSkyConfig):
    pass


class scaleByExposureTimeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_exposureTimeScaled", optional=True)
    time = config.RangeField("Output exposure time", float, None, min=0.1, optional=True)


class scaleCountsToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_countsScaled", optional=True)
    tolerance = config.RangeField("Tolerance for scaling compared to exposure time",
                                  float, 0, min=0, max=1, inclusiveMax=True)
    use_common = config.Field("Use only sources common to all frames?",
                              bool, True)
    radius = config.RangeField("Matching radius (arcseconds)", float, 0.5, min=0.1)


class subtractSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skySubtracted", optional=True)
    reset_sky = config.Field("Replace lost sky counts?", bool, False)
    scale_sky = config.Field("Scale sky frame to science frame?", bool, True)
    offset_sky = config.Field("Apply offset to sky frame to match science frame?", bool, False)
    sky = config.ListField("Sky frame to subtract", (str, AstroData), None, optional=True, single=True)
    save_sky = config.Field("Save sky frame to disk?", bool, False)


class skyCorrectConfig(parameters_stack.stackSkyFramesConfig, subtractSkyConfig):
    def setDefaults(self):
        self.suffix = "_skyCorrected"


class subtractSkyBackgroundConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyBackgroundSubtracted", optional=True)


class thresholdFlatfieldConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_thresholdFlatfielded", optional=True)
    upper = config.RangeField("Upper limit of valid pixels", float, 10., min=1)
    lower = config.RangeField("Lower limit of valid pixels", float, 0.01, max=1)
