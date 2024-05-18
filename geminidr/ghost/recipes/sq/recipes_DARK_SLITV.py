"""
Recipes available to data with tags ``['GHOST', 'SLITV', 'CAL', 'DARK']``.
Default is ``makeProcessedSlitDark``.
"""
recipe_tags = set(['GHOST', 'SLITV', 'CAL', 'DARK'])

def makeProcessedSlitDark(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input dark images into a single stacked dark image. This output
    processed dark is stored on disk using storeProcessedDark and has a name
    equal to the name of the first input dark image with "_dark.fits" appended.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames(operation='median')
    p.storeProcessedDark()
    return

_default = makeProcessedSlitDark
