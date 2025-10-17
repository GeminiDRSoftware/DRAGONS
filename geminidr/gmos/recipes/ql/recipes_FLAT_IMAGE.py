"""
Recipes available to data with tags ['GMOS', 'CAL', 'IMAGE', 'FLAT'].
Default is "makeProcessedFlat".
"""
recipe_tags = {'GMOS', 'CAL', 'IMAGE', 'FLAT'}

from geminidr.gmos.recipes.sq.recipes_FLAT_IMAGE import makeProcessedFlat
from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible

_default = makeProcessedFlat
