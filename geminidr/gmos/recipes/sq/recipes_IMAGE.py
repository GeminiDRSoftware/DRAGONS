"""
Recipes available to data with tags ['GMOS', 'IMAGE'].
Default is "reduce".
"""
recipe_tags = {'GMOS', 'IMAGE'}
blocked_tags = {'THRUSLIT'}

from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible


def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image.

    Parameters
    ----------
    p : GMOSImage object
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
    p.fringeCorrect()
    p.QECorrect()
    p.mosaicDetectors()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.flagCosmicRaysByStacking()
    p.scaleCountsToReference()
    p.stackFrames(zero=True)
    p.storeProcessedScience(suffix="_image")
    return


_default = reduce


def reduceSeparateCCDsCentral(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image. To deal
    with different color terms on the different CCDs, the images are
    split by CCD midway through the recipe and subsequently reduced
    separately. The relative WCS is determined from the central CCD
    (CCD2) and then applied to CCDs 1 and 3.

    Parameters
    ----------
    p : GMOSImage object
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
    p.fringeCorrect()
    p.tileArrays(tile_all=False)
    p.sliceIntoStreams(root_stream_name="ccd")
    p.detectSources(stream="ccd2")
    p.adjustWCSToReference(stream="ccd2")
    p.applyWCSAdjustment(stream="main", reference_stream="ccd2")
    p.clearAllStreams()
    p.detectSources()
    p.scaleCountsToReference()
    p.sliceIntoStreams(root_stream_name="ccd")
    for ccd in (1, 2, 3):
        p.resampleToCommonFrame(instream=f"ccd{ccd}")
        p.detectSources()
        p.flagCosmicRaysByStacking()
        p.stackFrames(zero=True)
        p.appendStream(stream="all", from_stream="main", copy=False)
    p.mergeInputs(instream="all")
    p.storeProcessedScience(suffix="_image")


def reduceSeparateCCDs(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image. To deal
    with different color terms on the different CCDs, the images are
    split by CCD midway through the recipe and subsequently reduced
    separately. The relative WCS is determined from mosaicked versions
    of the images and then applied to each of the CCDs separately.

    Parameters
    ----------
    p : GMOSImage object
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
    p.fringeCorrect()
    p.tileArrays(tile_all=False)
    p.mosaicDetectors(outstream="mosaic")
    p.detectSources(stream="mosaic")
    p.adjustWCSToReference(stream="mosaic")
    p.applyWCSAdjustment(reference_stream="mosaic")
    p.clearStream(stream="mosaic")
    p.detectSources()
    p.scaleCountsToReference()
    p.sliceIntoStreams(root_stream_name="ccd", copy=False)
    p.clearStream(stream="main")
    for ccd in (1, 2, 3):
        p.resampleToCommonFrame(instream=f"ccd{ccd}")
        p.detectSources()
        p.flagCosmicRaysByStacking()
        p.stackFrames(zero=True)
        p.appendStream(stream="all", from_stream="main", copy=False)
    p.mergeInputs(instream="all")
    p.storeProcessedScience(suffix="_image")


def makeProcessedFringe(p):
    """
    This recipe creates a fringe frame from the inputs files. The output
    is stored on disk using storeProcessedFringe and has a name equal
    to the name of the first input bias image with "_fringe.fits" appended.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    #p.addIllumMaskToDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.makeFringeFrame()
    p.storeProcessedFringe()
    return


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
    p.stackFrames(zero=True)
    return

# def makeIRAFCompatible(p):
#     """
#     Add header keywords needed to run some Gemini IRAF tasks.  This is needed
#     only if the reduced file will be used as input to Gemini IRAF tasks.
#
#     Parameters
#     ----------
#     p : PrimitivesBASEE object
#         A primitive set matching the recipe_tags.
#     """
#
#     p.makeIRAFCompatible()
#     p.writeOutputs()
#     return
