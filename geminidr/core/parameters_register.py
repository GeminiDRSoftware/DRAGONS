# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.
from gempy.library import config

class adjustWCSToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wcsCorrected", optional=True)
    method = config.ChoiceField("Alignment method", str,
                                allowed={"offsets": "Use telescope offsets",
                                         "sources": "Match sources in images"},
                                default="sources")
    fallback = config.ChoiceField("Fallback method", str,
                                  allowed={"offsets": "Use telescope offsets"},
                                  default="offsets", optional=True)
    first_pass = config.RangeField("Search radius for image alignment (arcseconds)",
                              float, 5., min=0, inclusiveMin=False)
    match_radius = config.RangeField("Search radius for source matching (arcseconds)",
                                     float, 0.5, min=0, inclusiveMin=False)
    min_sources = config.RangeField("Minimum number of sources required to use source matching",
                               int, 3, min=1)
    cull_sources = config.Field("Use only point sources for alignment?", bool, False)
    rotate = config.Field("Allow adjustment to rotation?", bool, False)
    scale = config.Field("Allow adjustment to magnification?", bool, False)
    debug_ignore_wcs = config.Field("Ignore WCS information when aligning?", bool, False)

class determineAstrometricSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_astrometryCorrected", optional=True)
    initial = config.RangeField("Search radius for cross-correlation (arcseconds)", float, 5., min=1)
    final = config.RangeField("Search radius for object matching (arcseconds)", float, 1., min=0)
    rotate = config.Field("Allow rotation for alignment?", bool, False)
    scale = config.Field("Allow magnification for alignment?", bool, False)
