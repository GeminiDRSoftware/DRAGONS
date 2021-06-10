"""
Recipes available to data with tags ['F2', 'CAL', 'DARK'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'F2', 'CAL', 'DARK'}

from geminidr.f2.recipes.sq.recipes_DARK import makeProcessedDark


_default = makeProcessedDark
