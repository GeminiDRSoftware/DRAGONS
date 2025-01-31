"""
Recipes available to data with tags ['F2', 'IMAGE', 'CAL', 'FLAT']
Default is "makeProcessedFlat".
"""
recipe_tags = {'F2', 'IMAGE', 'CAL', 'FLAT'}

# TODO: This recipe needs serious fixing to be made meaningful to the user.
def makeProcessedFlat(p):
    """
    This recipe calls a selection primitive, since K-band F2 flats only have
    lamp-off frames, and so need to be treated differently.

    Parameters
    ----------
    p : PrimitivesF2 object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    #p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.addToList(purpose='forFlat')
    p.getList(purpose='forFlat')
    p.makeLampFlat()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()
    return

_default = makeProcessedFlat
