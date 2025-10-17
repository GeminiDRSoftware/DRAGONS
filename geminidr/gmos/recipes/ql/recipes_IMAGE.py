"""
Recipes available to data with tags ['GMOS', 'IMAGE'].
Default is "makeProcessedBias".
"""
recipe_tags = {'GMOS', 'IMAGE'}
blocked_tags = {'THRUSLIT'}

from geminidr.gmos.recipes.sq.recipes_IMAGE import reduce
from geminidr.gmos.recipes.sq.recipes_IMAGE import makeProcessedFringe
from geminidr.gmos.recipes.sq.recipes_IMAGE import alignAndStack
from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible

_default = reduce
