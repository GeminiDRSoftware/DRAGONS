"""
Recipes available to data with tags ['F2', 'CAL', 'IMAGE', 'FLAT'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'F2', 'CAL', 'IMAGE', 'FLAT'}

from geminidr.f2.recipes.sq.recipes_FLAT_IMAGE import makeProcessedFlat


_default = makeProcessedFlat
