"""
Recipes available to data with tags ['GNIRS', 'CAL', 'DARK'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'CAL', 'DARK'}

from geminidr.gnirs.recipes.sq.recipes_DARK import makeProcessedDark


_default = makeProcessedDark
