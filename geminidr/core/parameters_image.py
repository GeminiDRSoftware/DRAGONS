# This parameter file contains the parameters related to the primitives located
# in the primitives_image.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData
from geminidr.core import parameters_stack, parameters_photometry
from geminidr.core import parameters_generic


class fringeCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_fringeCorrected", optional=True)
    fringe = config.ListField("Fringe frame to subtract", (str, AstroData),
                          None, optional=True, single=True)
    scale = config.Field("Scale fringe frame?", bool, None, optional=True)
    scale_factor = config.ListField("Scale factor for fringe frame", float, None,
                                    optional=True, single=True)


class makeFringeFrameConfig(parameters_stack.core_stacking_config, parameters_photometry.detectSourcesConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, True)
    dilation = config.RangeField("Object dilation radius (pixels)", float, 2., min=0)
    debug_distance = config.RangeField("Minimum association distance (arcsec)",
                                       float, 5., min=0)
    def setDefaults(self):
        self.suffix = "_fringe"
        self.reject_method = "varclip"


class makeFringeForQAConfig(makeFringeFrameConfig):
    pass


class resampleToCommonFrameConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_align", optional=True)
    interpolant = config.ChoiceField("Type of interpolant", str,
                                     allowed={"nearest": "Nearest neighbour",
                                              "linear": "Linear interpolation",
                                              "poly3": "Cubic polynomial interpolation",
                                              "poly5": "Quintic polynomial interpolation",
                                              "spline3": "Cubic spline interpolation",
                                              "spline5": "Quintic spline interpolation"},
                                     default="poly3", optional=False)
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
    clean_data = config.Field("Clean bad pixels before interpolation?", bool, False)
    conserve = config.Field("Conserve image flux?", bool, True)
    force_affine = config.Field("Force affine transformation for speed?", bool, True)
    reference = config.Field("Name of reference image (optional)", (str, AstroData), None, optional=True)
    dq_threshold = config.RangeField("Fraction from DQ-flagged pixel to count as 'bad'",
                                     float, 0.001, min=0.)


class scaleByIntensityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scaled", optional=True)
    section = config.Field("Statistics section", str, None, optional=True)
    scaling = config.ChoiceField("Statistic for scaling", str,
                                 allowed={"mean": "Scale by mean",
                                            "median": "Scale by median"},
                                 default="mean")
    separate_ext = config.Field("Scale extensions separately?", bool, False)


class transferObjectMaskConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_objmaskTransferred", optional=True)
    source = config.Field("Filename of/Stream containing stacked image", str, None)
    interpolant = config.ChoiceField("Type of interpolant", str,
                                     allowed={"nearest": "Nearest neighbour",
                                              "linear": "Linear interpolation",
                                              "poly3": "Cubic polynomial interpolation",
                                              "poly5": "Quintic polynomial interpolation",
                                              "spline3": "Cubic spline interpolation",
                                              "spline5": "Quintic spline interpolation"},
                                     default="poly3", optional=False)
    threshold = config.RangeField("Threshold for flagging pixels", float, 0.01, min=0., max=1.)
    dilation = config.RangeField("Dilation radius (pixels)", float, 1.5, min=0)


class flagCosmicRaysByStackingConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_CRMasked", optional=True)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 7., min=0)
    dilation = config.RangeField("CR dilation radius (pixels)", float, 1., min=0)
