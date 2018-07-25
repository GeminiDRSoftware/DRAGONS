# This parameter file contains the parameters related to the primitives located
# in the primitives_image.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData
from geminidr.core import parameters_stack, parameters_photometry

class fringeCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringeCorrected", optional=True)
    fringe = config.ListField("Fringe frame to subtract", (str, AstroData),
                          None, optional=True, single=True)
    scale = config.Field("Scale fringe frame?", bool, False)
    scale_factor = config.ListField("Scale factor for fringe frame", float, 1.,
                                    optional=True, single=True)

class makeFringeConfig(parameters_stack.core_stacking_config, parameters_photometry.detectSourcesConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, True)
    dilation = config.RangeField("Object dilation radius (pixels)", float, 2., min=0)
    def setDefaults(self):
        self.suffix = "_fringe"
        self.operation = "median"
        self.reject_method = "minmax"
        self.nhigh = 1

class makeFringeFrameConfig(parameters_stack.core_stacking_config, parameters_photometry.detectSourcesConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, True)
    dilation = config.RangeField("Object dilation radius (pixels)", float, 2., min=0)
    def setDefaults(self):
        self.suffix = "_fringe"

class scaleByIntensityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scaled", optional=True)
    section = config.Field("Statistics section", str, None, optional=True)
    scaling = config.ChoiceField("Statistic for scaling", str,
                                 allowed = {"mean": "Scale by mean",
                                            "median": "Scale by median"},
                                 default = "mean")
    separate_ext = config.Field("Scale extensions separately?", bool, False)
