# This parameter file contains the parameters related to the primitives located
# in the primitives_visualize.py file, in alphabetical order.
from gempy.library import config

def thresholdCheck(value):
    return (isinstance(value, float) or value == 'auto')

class displayConfig(config.Config):
    extname = config.Field("EXTNAME to display", str, "SCI")
    frame = config.RangeField("Starting frame for display", int, 1, min=1)
    ignore = config.Field("Turn off display?", bool, False)
    overlay = config.ListField("Overlays for display", tuple, None, optional=True)
    remove_bias = config.Field("Remove estimated bias level before displaying?", bool, False)
    threshold = config.Field("Threshold level for indicating saturation",
                             (str, float), "auto", optional=True, check=thresholdCheck)
    tile = config.Field("Tile multiple extensions into single display frame?", bool, True)
    zscale = config.Field("Use zscale algorithm?", bool, True)

class mosaicDetectorsConfig(config.Config):
    pass

class mosaicADdetectorsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mosaicked")
    tile = config.Field("Tile rather than mosaic?", bool, False)
    doimg = config.Field("Mosaic only SCI extensions?", bool, False)
    interpolator = config.Field("Type of interpolation", str, "linear")

class tileArraysConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_tiled")
    tile_all = config.Field("Tile to single extension?", bool, False)
