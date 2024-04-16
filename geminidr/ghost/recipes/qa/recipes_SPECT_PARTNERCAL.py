"""
Recipes available to data with tags ['GHOST', 'SPECT', 'PARTNER_CAL'].
Default is "reduce".
"""
recipe_tags = set(['GHOST', 'SPECT', 'PARTNER_CAL'])

from ..sq.recipes_SPECT_PARTNERCAL import reduceScience, reduceStandard

_default = reduceStandard
