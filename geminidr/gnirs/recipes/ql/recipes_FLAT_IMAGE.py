"""
Recipes available to data with tags ['GNIRS', 'CAL', 'IMAGE', 'FLAT'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'CAL', 'IMAGE', 'FLAT'}

from geminidr.gnirs.recipes.sq.recipes_FLAT_IMAGE import makeProcessedFlat


_default = makeProcessedFlat
