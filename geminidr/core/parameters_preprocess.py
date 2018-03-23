# This parameter file contains the parameters related to the primitives located
# in the primitives_preprocess.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class addObjectMaskToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_objectMaskAdded")

class ADUToElectronsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_ADUToElectrons")

class applyDQPlaneConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dqPlaneApplied")
    replace_flags = config.RangeField("DQ bitmask for pixels to replace", int, 255, min=0)
    replace_value = config.Field("Replacement value", (str, float), "median")

class associateSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyassociated")
    time = config.RangeField("Maximum association time (seconds)", float, 600., min=0)
    distance = config.RangeField("Minimum association distance (arcsec)", float, 3., min=0)
    max_skies = config.RangeField("Maximum number of sky frames to associate",
                             int, None, min=1, optional=True)
    use_all = config.Field("Use all frames as skies?", bool, False)

class correctBackgroundToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_backgroundCorrected")
    remove_background = config.Field("Remove background level?", bool, False)

class darkCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_darkCorrected")
    dark = config.Field("Dark frame", (str, AstroData), None, optional=True)

class dilateObjectMaskConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_objmaskDilated")
    dilation = config.RangeField("Dilation radius (pixels)", float, 1., min=0)
    repeat = config.Field("Allow dilation of already-dilated image?", bool, False)

class flatCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_flatCorrected")
    flat = config.Field("Flatfield frame", (str, AstroData), None, optional=True)

class nonlinearityCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_nonlinearityCorrect")

class normalizeFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_normalized")
    scale = config.ChoiceField("Statistic for scaling", str,
                               allowed = {"mean": "Scale by mean",
                                          "median": "Scale by median"},
                               default="median")
    separate_ext = config.Field("Scale extensions separately?", bool, False)

class separateSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skySeparated")
    ref_obj = config.Field("Manually-assigned object files", str, "")
    ref_sky = config.Field("Manually-assigned sky files", str, "")
    frac_FOV = config.RangeField("Field of view scaling for coaddition", float, 0.9, min=0.5, max=1)

class makeSkyConfig(associateSkyConfig, separateSkyConfig):
    pass

class subtractSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skySubtracted")
    reset_sky = config.Field("Replace lost sky counts?", bool, False)
    scale = config.Field("Scale sky frame to science frame?", bool, True)
    zero = config.Field("Apply offset to sky frame to match science frame?", bool, False)
    sky = config.ListField("Sky frame to subtract", (str, AstroData), None, optional=True, single=True)

class skyCorrectConfig(subtractSkyConfig):
    suffix = config.Field("Filename suffix", str, "_skyCorrected")
    dilation = config.RangeField("Object mask dilation radius (pixels)", float, 2., min=0)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    mask_objects = config.Field("Use OBJMASK when making sky?", bool, True)
    mclip = config.Field("Use median for sigma-clipping?", bool, True)
    operation = config.Field("Averaging operation", str, "median")
    reject_method = config.Field("Rejection method", str, "sigclip")

class subtractSkyBackgroundConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyBackgroundSubtracted")

class thresholdFlatfieldConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_thresholdFlatfielded")
    upper = config.RangeField("Upper limit of valid pixels", float, 10., min=1)
    lower = config.RangeField("Lower limit of valid pixels", float, 0.01, max=1)
