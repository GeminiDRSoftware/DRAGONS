"""
Recipes available to data with tags 'GMOS', 'CAL', 'DARK'].
Default is "makeProcessedDark".
"""

recipe_tags = set(['NIRI', 'CAL', 'DARK'])

def makeProcessedDark(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input dark images into a single stacked dark image.
    This output processed dark is stored on disk using storeProcessedDark
    and has a name equal to the name of the first input dark image with
    "_dark.fits" appended.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ(illum_mask=False)
    p.addVAR(read_noise=True)
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames()
    p.storeProcessedDark()
    return

default = makeProcessedDark
