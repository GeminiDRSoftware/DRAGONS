"""
Recipes available to data with tags ['F2', 'IMAGE', 'CAL', 'FLAT']
Default is "makeProcessedFlat".
"""
recipe_tags = set(['F2', 'IMAGE', 'CAL', 'FLAT'])

default = makeProcessedFlat

# TODO: This recipe needs serious fixing to be made meaningful to the user.
def makeProcessedFlat(p):
    """
    This recipe calls a selection primitive, since K-band F2 flats only have
    lamp-off frames, and so need to be treated differently.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.selectFlatRecipe()
    return

