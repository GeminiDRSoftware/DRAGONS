"""
Recipes available to data with tags ['GHOST', 'SPECT'].
Default is "reduce".
"""
recipe_tags = set(['GHOST', 'SPECT'])

from ..sq.recipes_SPECT import reduceScience, reduceStandard

_default = reduceScience
