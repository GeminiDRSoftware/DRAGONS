"""
Recipes available to data with tags ['GSAOI', 'CAL', 'IMAGE', 'FLAT'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'GSAOI', 'CAL', 'IMAGE', 'FLAT'}

from geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE import makeProcessedFlat


_default = makeProcessedFlat
