# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.
from gempy.library import config

class matchWCSToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wcsCorrected", optional=True)
    method = config.ChoiceField("Alignment method", str,
                                allowed={"offsets": "Use telescope offsets",
                                         "sources": "Match sources in images"},
                                default="sources")
    fallback = config.ChoiceField("Fallback method", str,
                                  allowed={"offsets": "Use telescope offsets"},
                                  default="offsets", optional=True)
    first_pass = config.RangeField("Search radius for source matching (arcseconds)",
                              float, 5., min=0)
    min_sources = config.RangeField("Minimum number of sources required to use source matching",
                               int, 3, min=1)
    cull_sources = config.Field("Use only point sources for alignment?", bool, False)
    rotate = config.Field("Allow rotation for alignment?", bool, False)
    scale = config.Field("Allow magnification for alignment?", bool, False)

class determineAstrometricSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_astrometryCorrected", optional=True)
    initial = config.RangeField("Search radius for cross-correlation (arcseconds)", float, 5., min=1)
    final = config.RangeField("Search radius for object matching (arcseconds)", float, 1., min=0)
    # None => False if 'qa' in mode else True
    full_wcs = config.Field("Recompute positions using full WCS rather than offsets?",
                            bool, None, optional=True)
