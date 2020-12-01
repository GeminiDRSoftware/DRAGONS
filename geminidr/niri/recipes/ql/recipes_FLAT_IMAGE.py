"""
Recipes available to data with tags ['NIRI', 'CAL', 'IMAGE', 'FLAT'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'NIRI', 'CAL', 'IMAGE', 'FLAT'}

from geminidr.niri.recipes.sq.recipes_FLAT_IMAGE import makeProcessedFlat


_default = makeProcessedFlat
