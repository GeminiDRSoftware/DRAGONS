"""
Recipes available to data with tags ['GNIRS', IMAGE'].
Default is "reduce".
"""
recipe_tags = {'F2', 'IMAGE'}

#from geminidr.f2.recipes.sq.recipes_IMAGE import reduce
from geminidr.f2.recipes.sq.recipes_IMAGE import alignAndStack
from geminidr.f2.recipes.sq.recipes_IMAGE import makeIRAFCompatible

def reduce(p):
    """
    This recipe process F2 data up to and including alignment and stacking.
    A single stacked output image is produced.
    It will attempt to do dark and flat correction if a processed calibration
    is available.

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
    p.flushPixels()
    p.separateSky()
    p.associateSky(max_skies=9)
    p.skyCorrect(mask_objects=False)
    p.flushPixels()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.writeOutputs()
    p.storeProcessedScience(suffix="_image")
    return

_default = reduce
