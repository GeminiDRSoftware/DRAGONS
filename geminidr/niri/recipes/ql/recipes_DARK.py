"""
Recipes available to data with tags ['NIRI', 'CAL', 'DARK'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'NIRI', 'CAL', 'DARK'}

from geminidr.niri.recipes.sq.recipes_DARK import makeProcessedDark


_default = makeProcessedDark
