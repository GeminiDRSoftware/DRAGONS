"""
Recipes available to data with tags ['GMOS', 'IMAGE'].
Default is "reduce_nostack".
"""
recipe_tags = set(['GMOS', 'IMAGE'])

def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image.
    QA metrics are being calculated at different point during the reduction.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.detectSources()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCC()
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.detectSources()
    p.determineAstrometricSolution()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCC()
    p.addToList(purpose='forStack')
    p.getList(purpose='forStack')
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.detectSources()
    p.determineAstrometricSolution()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCC()
    p.writeOutputs()
    return


def reduce_nostack(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into an image ready to be stacked.
    QA metrics are being calculated at different point during the reduction.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureCC()
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.writeOutputs()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.determineAstrometricSolution()
    p.measureCC()
    p.addToList(purpose='forStack')
    p.writeOutputs()
    return

# we have to use the nostack version for qap because stacking is too slow.
# KL: is this still true with gemini_python 2.0?
# KRA: unknown yet.
default = reduce_nostack

def stack(p):
    """
    This recipe stacks images already reduced up to stacking.  It will
    collect data marked "forStack", for example the output of
    reduce_nostack.  The product is a stack of the aligned inputs with
    suffix "_stack".  QA metrics are measured.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.getList(purpose='forStack')
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.correctBackgroundToReference()
    p.stackFrames()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.determineAstrometricSolution()
    p.measureCC()
    p.writeOutputs()
    return

def makeProcessedFringe(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input fringe images into a single stacked fringe
    image. This output processed fringe is stored on disk using
    storeProcessedFringe and has a name equal to the name of the first
    input fringe image with "_fringe.fits" appended.

    Fringe frames are normally generated with normal science data.  There
    isn't a keyword identifying raw frames as fringe frames.  Therefore
    we cannot put this recipe in a set specific to a fringe tag.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.addToList(purpose="forFringe")
    p.getList(purpose="forFringe")
    p.makeFringeFrame()
    p.storeProcessedFringe()
    p.writeOutputs()
    return
