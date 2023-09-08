"""
Recipes available to data with tags ['GHOST', 'CAL', 'SLITV', 'ARC'].
Default is "makeProcessedSlitArc".
"""
recipe_tags = set(['GHOST', 'CAL', 'SLITV', 'ARC',])

from .recipes_SLITV import makeProcessedSlit

def makeProcessedSlitArc(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input arc images into a single stacked arc image. This output
    processed arc is stored on disk using storeProcessedArc and has a name
    equal to the name of the first input arc image with "_arc.fits" appended.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.biasCorrect()
    p.addVAR(poisson_noise=True)
    p.ADUToElectrons()
    p.darkCorrect()
    p.CRCorrect()
    p.addToList(purpose="forStack")
    p.getList(purpose="forStack")
    p.stackFrames(operation='median')
    p.storeProcessedArc()
    return

makeProcessedSlitArc = makeProcessedSlit

_default = makeProcessedSlitArc
