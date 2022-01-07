"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'ARC'].
These are GMOS longslit arc-lamp calibrations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'ARC'}

from ..sq.recipes_ARC_LS_SPECT import makeProcessedArc

_default = makeProcessedArc
