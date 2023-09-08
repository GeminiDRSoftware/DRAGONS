"""
Recipes available to data with tags: GHOST, BUNDLE, RAW, UNPREPARED
Default is "makeProcessedBundle".
"""
recipe_tags = set(['GHOST', 'BUNDLE', 'RAW', 'UNPREPARED'])

from ..sq.recipes_BUNDLE import makeProcessedBundle

_default = makeProcessedBundle
