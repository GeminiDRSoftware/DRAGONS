"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'NODANDSHUFFLE'].

The default recipe is "reduce".

"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'NODANDSHUFFLE'}


from ..sq.recipes_NS_LS_SPECT import reduce


_default = reduce
