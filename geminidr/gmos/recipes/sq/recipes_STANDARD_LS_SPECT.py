"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduceStandard".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'STANDARD'}

from ..ql.recipes_STANDARD_LS_SPECT import reduceScience, reduceStandard

_default = reduceStandard
