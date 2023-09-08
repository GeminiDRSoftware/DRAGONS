"""
Recipes available to data with tags ['GHOST', 'CAL', 'DARK'].
Default is "makeProcessedDark".
"""
recipe_tags = set(['GHOST', 'CAL', 'DARK'])

def makeProcessedDark(p):
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
    p.overscanCorrect()
    #p.tileArrays()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.addToList(purpose="forStack")
    p.getList(purpose="forStack")
    p.stackFrames(operation='median',reject_method='sigclip', mclip=True,
                  # snoise=0.02,   # No longer a valid option
                  lsigma=7., hsigma=7.)
    p.clipSigmaBPM(sigma=3.0, iters=10)
    p.storeProcessedDark()
    return

_default = makeProcessedDark
