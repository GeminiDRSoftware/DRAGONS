"""
Recipes available to data with tags ['GMOS', 'IMAGE', 'CAL', 'FLAT']
Default is "makeProcessedFlat".
"""
recipe_tags = set(['GMOS', 'IMAGE', 'CAL', 'FLAT'])

def makeProcessedFlat(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input flat images into a single stacked and normalized
    flat image.  This output processed flat is stored on disk using
    storeProcessedFlat and has a name equal to the name of the first input
    flat image with "_flat.fits" appended.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.display()
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.addToList(purpose="forStack")
    p.getList(purpose="forStack")
    p.stackFlats()
    p.normalizeFlat()
    p.storeProcessedFlat()
    return

_default = makeProcessedFlat
