"""
Recipes available to data with tags ['GHOST', 'CAL', 'ARC'].
Default is "makeProcessedArc".
"""
recipe_tags = set(['GHOST', 'CAL', 'ARC'])

from ..sq.recipes_ARC import makeProcessedArc

_default = makeProcessedArc
