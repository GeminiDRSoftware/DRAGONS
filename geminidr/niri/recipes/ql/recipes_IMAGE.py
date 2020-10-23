"""
Recipes available to data with tags ['NIRI', IMAGE'].
Default is "reduce".
"""
recipe_tags = {'NIRI', 'IMAGE'}

from geminidr.niri.recipes.sq.recipes_IMAGE import reduce
from geminidr.niri.recipes.sq.recipes_IMAGE import alignAndStack


_default = reduce
