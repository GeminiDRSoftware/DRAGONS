# This parameter file contains the parameters related to the primitives located
# in the primitives_image.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class fringeCorrectConfig(config.Config):
    pass

class makeFringeConfig(config.Config):
    pass

class makeFringeFrameConfig(config.Config):
    pass

class ScaleByIntensityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scaled")
    section = config.Field("Statistics section", str, None, optional=True)
    scaling = config.ChoiceField("Statistic for scaling", str,
                                 allowed = {"mean": "Scale by mean",
                                            "median": "Scale by median"},
                                 default = "mean")
    separate_ext = config.Field("Scale extensions separately?", bool, False)

class scaleFringeToScienceConfig(config.Config):
    pass

class subtractFringeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringeSubtracted")
    fringe = config.Field("Fringe frame to subtract", (str, AstroData),
                          None, optional=True)
