"""
Recipes available to data with tags ['GNIRS', 'IMAGE']
Default is "reduce".
"""
recipe_tags = {'GNIRS', 'IMAGE'}

def reduce(p):
    """
    This recipe process GNIRS keyhole imaging data and stack them in
    a single stacked output image is produced.
    It will attempt flat correction if a processed calibration
    is available.  Sky subtraction is done when possible.  QA metrics are
    measured.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.nonlinearityCorrect()
    p.darkCorrect()
    p.flatCorrect()
    p.applyDQPlane()
    p.separateSky()
    p.associateSky()
    p.skyCorrect(mask_objects=False)
    #p.cleanReadout()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.writeOutputs()
    p.storeProcessedScience(suffix="_image")
    return

_default = reduce


def alignAndStack(p):
    """
    This recipe stack already preprocessed data.

    Parameters
    ----------
    p : PrimitivesBASEE object
        A primitive set matching the recipe_tags.
    """

    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    return
