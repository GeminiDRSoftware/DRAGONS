# This parameter file contains the parameters related to the primitives located
# in the primitives_photometry.py file, in alphabetical order.
from gempy.library import config

class addReferenceCatalogConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_refcatAdded", optional=True)
    radius = config.RangeField("Search radius (degrees)", float, 0.067, min=0.)
    source = config.ChoiceField("Name of catalog to search", str,
                                allowed = {"gmos": "Gemini optical catalog",
                                           "2mass": "2MASS Infrared catalog",
                                           "sdss9": "SDSS DR9 optical catalog",
                                           "ukidss9": "UKIDSS DR9 infrared catalog"},
                                default = "gmos")

class detectSourcesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_sourcesDetected", optional=True)
    mask = config.Field("Replace DQ-flagged pixels with median of image?", bool, False)
    replace_flags = config.RangeField("DQ bitmask for flagging if mask=True", int, 249, min=0)
    set_saturation = config.Field("Inform SExtractor of saturation level?", bool, False)
    detect_minarea = config.RangeField("Minimum object detection area (pixels)", int, 8, min=1)
    detect_thresh = config.RangeField("Detection threshold (standard deviations)", float, 2., min=0.1)
    analysis_thresh = config.RangeField("Analysis threshold (standard deviations)", float, 2., min=0.1)
    deblend_mincont = config.RangeField("Minimum deblending contrast", float, 0.005, min=0.)
    phot_min_radius = config.RangeField("Minimum radius for photometry (pixels)", float, 3.5, min=1.0)
    back_size = config.RangeField("Background mesh size (pixels)", int, 32, min=1)
    back_filtersize = config.RangeField("Filtering scale for background", int, 8, min=1)
