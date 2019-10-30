"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS'])

from ..ql.recipes_LS_SPECT import reduce, reduceStandard

_default = reduce
