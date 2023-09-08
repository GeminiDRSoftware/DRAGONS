"""
Recipes available to data with tags ``['GHOST', 'SLITV', 'CAL', 'FLAT']``.
Default is ``makeProcessedSlitFlat``.
"""
recipe_tags = set(['GHOST', 'SLITV', 'CAL', 'FLAT'])

def makeProcessedSlitFlat(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input flat images into a single stacked flat image. This output
    processed flat is stored on disk using storeProcessedFlat and has a name
    equal to the name of the first input flat image with "_flat.fits" appended.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """

    p.prepare(attach_mdf=False)
    p.addDQ()
    p.addVAR(read_noise=True)
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.stackFrames(operation='median')
    p.storeProcessedSlitFlat()
    return

_default = makeProcessedSlitFlat
