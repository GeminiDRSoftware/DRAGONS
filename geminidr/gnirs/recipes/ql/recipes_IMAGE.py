"""
Recipes available to data with tags ['GNIRS', IMAGE'].
Default is "reduce".
"""
recipe_tags = {'GNIRS', 'IMAGE'}

from geminidr.gnirs.recipes.sq.recipes_IMAGE import reduce
from geminidr.gnirs.recipes.sq.recipes_IMAGE import alignAndStack


_default = reduce
