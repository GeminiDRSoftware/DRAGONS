"""
Recipes available to data with tags ['GSAOI', 'CAL', 'DARK'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'GSAOI', 'CAL', 'DARK'}

from geminidr.gsaoi.recipes.sq.recipes_DARK import makeProcessedDark


_default = makeProcessedDark
