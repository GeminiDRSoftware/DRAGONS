"""
Recipes available to data with tags ['GNIRS', IMAGE'].
Default is "reduce".
"""
recipe_tags = {'F2', 'IMAGE'}

from geminidr.f2.recipes.sq.recipes_IMAGE import reduce
from geminidr.f2.recipes.sq.recipes_IMAGE import alignAndStack
from geminidr.f2.recipes.sq.recipes_IMAGE import makeIRAFCompatible


_default = reduce
