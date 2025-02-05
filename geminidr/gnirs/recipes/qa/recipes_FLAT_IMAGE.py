"""
Recipes available to data with tags ['GNIRS', 'IMAGE', 'CAL', 'FLAT']
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'IMAGE', 'CAL', 'FLAT'}

def makeProcessedFlat(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input flat images into a single stacked and normalized flat image.
    This output processed flat is stored on disk using storeProcessedFlat and
    has a name equal to the name of the first input flat image with "_flat.fits"
    appended.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ(add_illum_mask=False)
    #p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.addToList(purpose='forFlat')
    p.getList(purpose='forFlat')
    p.makeLampFlat()
    p.addIllumMaskToDQ()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()
    return

_default = makeProcessedFlat
