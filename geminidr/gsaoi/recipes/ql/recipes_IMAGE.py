"""
Recipes available to data with tags ['GSAOI', IMAGE'].
Default is "reduce_nostack".
"""
recipe_tags = {'GSAOI', 'IMAGE'}

from geminidr.gsaoi.recipes.sq.recipes_IMAGE import reduce_nostack


_default = reduce_nostack
