# This parameter file contains the parameters related to the primitives located
# in the primitives_visualize.py file, in alphabetical order.
import numbers

from gempy.library import config

def threshold_check(value):
    return (isinstance(value, float) or value == 'auto')


class displayConfig(config.Config):
    extname = config.Field("EXTNAME to display", str, "SCI")
    frame = config.RangeField("Starting frame for display", int, 1, min=1)
    ignore = config.Field("Turn off display?", bool, False)
    debug_overlay = config.Field("Overlays for display", (tuple, str), None, optional=True)
    threshold = config.Field("Threshold level for indicating non-linearity and saturation. 'None' to turn off.",
                             (str, float), "auto", optional=True, check=threshold_check)
    tile = config.Field("Tile multiple extensions into single display frame?", bool, True)
    zscale = config.Field("Use zscale algorithm?", bool, True)


class inspectConfig(displayConfig):
    pause = config.RangeField("Pause between the display, in seconds", int, 2, min=0)


class mosaicDetectorsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mosaic")
    sci_only = config.Field("Mosaic only SCI extensions?", bool, False)
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)


class tileArraysConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_tiled", optional=True)
    sci_only = config.Field("Tile only SCI extensions?", bool, False)
    tile_all = config.Field("Tile to single extension?", bool, False)


class plotSpectraForQAConfig(config.Config):
    url = config.Field(
        doc="URL address to the ADCC server.",
        dtype=str,
        default="http://localhost:8777/spec_report")
