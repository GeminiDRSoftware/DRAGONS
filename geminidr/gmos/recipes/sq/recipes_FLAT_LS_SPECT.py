"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS', 'FLAT'])

from ..ql.recipes_FLAT_LS_SPECT import makeProcessedFlat

_default = makeProcessedFlat
