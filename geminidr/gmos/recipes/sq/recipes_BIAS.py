"""
Recipes available to data with tags ['GMOS', 'CAL', 'BIAS'].
Default is "makeProcessedBias".
"""
recipe_tags = {'GMOS', 'CAL', 'BIAS'}

from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible

def makeProcessedBias(p):
    """
     This recipe performs the standardization and corrections needed to convert
     the raw input bias images into a single stacked bias image. This output
     processed bias is stored on disk using storeProcessedBias and has a name
     equal to the name of the first input bias image with "_bias.fits" appended.

     Parameters
     ----------
     p : PrimitivesBASE object
         A primitive set matching the recipe_tags.
     """

    p.prepare()
    p.addDQ(add_illum_mask=False)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.stackBiases()
    p.makeIRAFCompatible()
    p.storeProcessedBias()
    return

def checkBias(p):
    """
    This recipe checks bias frames by processing them as regular bias frames
    (notably including overscan correction), then recording some pixel
    statistics. We then additionally subtract a "known good" processed bias
    from them and compute statistics on the residual.
    :param p:
    :return:
    """
    p.prepare()
    p.addDQ(add_illum_mask=False)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.stats(prefix='OSCO')
    p.biasCorrect()
    p.stats(prefix="BICO")
    p.writeOutputs(strip=True, suffix='_checkBias')
    return


_default = makeProcessedBias
