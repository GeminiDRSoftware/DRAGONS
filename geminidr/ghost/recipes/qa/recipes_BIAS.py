"""
Recipes available to data with tags ['GHOST', 'CAL', 'BIAS'].
Default is "makeProcessedBias".
"""
recipe_tags = set(['GHOST', 'CAL', 'BIAS'])

def makeProcessedBias(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input bias images into a single stacked bias image. This output
    processed bias is stored on disk using storeProcessedBias and has a name
    equal to the name of the first input bias image with "_bias.fits" appended.

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
    p.addToList(purpose="forStack")
    p.getList(purpose="forStack")
    p.stackFrames(operation="median")
    p.storeProcessedBias()
    return

_default = makeProcessedBias
