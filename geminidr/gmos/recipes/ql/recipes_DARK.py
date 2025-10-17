"""
Recipes available to data with tags ['GMOS', 'CAL', 'DARK'].
Default is "makeProcessedDark".
"""
recipe_tags = {'GMOS', 'CAL', 'DARK'}

from geminidr.gmos.recipes.sq.recipes_DARK import makeProcessedDark
from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible

_default = makeProcessedDark
