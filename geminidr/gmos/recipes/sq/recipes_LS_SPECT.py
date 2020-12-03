"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduceScience".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS'}
blocked_tags = {'NODANDSHUFFLE'}

from ..ql.recipes_LS_SPECT import reduceScience, reduceStandard

_default = reduceScience
