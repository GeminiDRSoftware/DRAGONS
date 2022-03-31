"""
Recipes available to data with tags ['GSAOI', 'IMAGE']
Default is "reduce".
"""
recipe_tags = {'GSAOI', 'IMAGE'}

def reduce(p):
    """
    This recipe will fully reduce GSAOI data, including alignment and
    stacking.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.flatCorrect()
    p.flushPixels()
    p.separateSky()
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.writeOutputs()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.storeProcessedScience()
    return

def reduce_nostack(p):
    """
    This recipe reduce GSAOI up to but NOT including alignment and stacking.
    It will attempt to do flat correction and sky subtraction.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.flatCorrect()
    p.flushPixels()
    p.separateSky()
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.storeProcessedScience()
    return

def alignAndStack(p):
    """
    This recipe continues the reduction of GSAOI imaging data that have been
    processed by the reduce_nostack() recipe. It aligns and stacks the images.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.storeProcessedScience()
    return


# The nostack version is used because stacking of GSAOI is time consuming.
_default = reduce
