"""
Recipes available to data with tags ['GMOS', 'CAL', 'BIAS'].
Default is "makeProcessedBias".
"""
recipe_tags = {'GMOS', 'CAL', 'BIAS'}

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
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.addToList(purpose="forStack")
    p.getList(purpose="forStack")
    p.stackBiases()
    p.storeProcessedBias()
    return

def checkBias1(p):
    """
    This recipe checks bias frames by processing them as regular bias frames
    (notably including overscan correction), then recording some pixel
    statistics.
    :param p:
    :return:
    """
    p.prepare()
    p.addDQ(add_illum_mask=False)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.recordPixelStats(prefix='OSCO')
    p.writeOutputs(strip=True, suffix='_checkBias1')
    return

def checkBias2(p):
    """
    This recipe checks bias frames by processing them as regular bias frames
    (notably including overscan correction), then subtracting a processed bias
    and recording some pixel statistics.
    :param p:
    :return:
    """
    p.prepare()
    p.addDQ(add_illum_mask=False)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect(do_cal="force")
    p.recordPixelStats(prefix="BICO")
    p.writeOutputs(strip=True, suffix='_checkBias2')
    return

_default = makeProcessedBias
