"""
Recipes available to data with tags ``['GHOST', 'CAL', 'BIAS']``.
Default is ``makeProcessedBias``.
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
    p.stackFrames(operation="median")
    p.storeProcessedBias()
    return

def checkBiasOSCO(p):
    """
    This recipe checks bias frames by processing them as regular bias frames
    (notably including overscan correction), then recording some pixel
    statistics.
    :param p:
    :return:
    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.recordPixelStats(prefix='OSCO')
    p.writeOutputs(strip=True, suffix='_checkBiasOSCO')
    return

_default = makeProcessedBias
