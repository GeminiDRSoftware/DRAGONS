"""
Recipes available to data with tags ['GMOS', 'CAL', 'DARK'].
Default is "makeProcessedDark".
"""
recipe_tags = {'GMOS', 'CAL', 'DARK'}

from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible


def makeProcessedDark(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input dark images into a single stacked dark image. This output
    processed bias is stored on disk using storeProcessedDark and has a name
    equal to the name of the first input bias image with "_dark.fits" appended.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.addToList(purpose="forStack")
    p.getList(purpose="forStack")
    # Force "varclip" due to large number of CRs
    p.stackFrames(zero=False, scale=False, reject_method="varclip")
    p.storeProcessedDark()
    return


_default = makeProcessedDark
