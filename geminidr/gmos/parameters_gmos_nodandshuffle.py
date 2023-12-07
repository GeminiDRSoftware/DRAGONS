# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_nodandshuffle.py file, in alphabetical order.
from gempy.library import config, astrotools as at


def validate_section(value):
    ranges = at.parse_user_regions(value, dtype=int, allow_step=False)
    return len(ranges) == 1


class combineNodAndShuffleBeamsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_beamCombined", optional=True)
    align_sources = config.Field("Try to align beams using slit profile?", bool, False)
    region = config.Field("Pixel section for measuring the spatial profile",
                          str, None, optional=True, check=validate_section)
    tolerance = config.RangeField("Maximum distance from the header offset "
                                  "for the correlation method (arcsec)",
                                  float, 0.5, min=0., optional=True)
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)
    subsample = config.RangeField("Subsampling", int, 1, min=1)
    dq_threshold = config.RangeField("Fraction from DQ-flagged pixel to count as 'bad'",
                                     float, 0.001, min=0.)


class skyCorrectNodAndShuffleConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyCorrected", optional=True)
