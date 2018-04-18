# This parameter file contains the parameters related to the primitives located
# in the primitives_image.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class makeFringeConfig(config.Config):
    pass

class makeFringeFrameConfig(config.Config):
    pass

class scaleByIntensityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scaled", optional=True)
    section = config.Field("Statistics section", str, None, optional=True)
    scaling = config.ChoiceField("Statistic for scaling", str,
                                 allowed = {"mean": "Scale by mean",
                                            "median": "Scale by median"},
                                 default = "mean")
    separate_ext = config.Field("Scale extensions separately?", bool, False)

class scaleFringeToScienceConfig(config.Config):
    pass

class subtractFringeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringeSubtracted", optional=True)
    fringe = config.ListField("Fringe frame to subtract", (str, AstroData),
                          None, optional=True, single=True)

class fringeCorrectConfig(subtractFringeConfig):
    def setDefaults(self):
        self.suffix = '_fringeCorrected'
