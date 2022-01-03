"""
Recipes available to data with tags ['GMOS', 'IMAGE'].
Default is "reduce".
"""
recipe_tags = {'GMOS', 'IMAGE'}

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
    p.mosaicDetectors()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.flagCosmicRaysByStacking()
    p.scaleByExposureTime()
    p.stackFrames(zero=True)
    p.storeProcessedScience()
    return


_default = reduce


def separateCCDs(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image. To deal
    with different color terms on the different CCDs, the images are
    split by CCD midway through the recipe and subsequently reduced
    separately. The relative WCS is determined from CCD2 and then applied
    to CCDs 1 and 3.

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
    p.sliceIntoStreams(clear=True)
    p.detectSources(stream="ext2")
    p.adjustWCSToReference(stream="ext2")
    p.applyWCSAdjustment(stream="ext1", reference_stream="ext2")
    p.applyWCSAdjustment(stream="ext3", reference_stream="ext2")
    for index in (1, 2, 3):
        p.resampleToCommonFrame(instream=f"ext{index}")
        p.detectSources()
        p.flagCosmicRaysByStacking()
        p.scaleByExposureTime()
        p.stackFrames(zero=True, suffix=f"_ccd{index}_stack")
        p.storeProcessedScience()


def separateCCDsMosaicking(p):
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
    p.sliceIntoStreams(clear=True)
    for index in (1, 2, 3):
        p.resampleToCommonFrame(instream=f"ext{index}")
        p.detectSources()
        p.flagCosmicRaysByStacking()
        p.scaleByExposureTime()
        p.stackFrames(zero=True, suffix=f"_ccd{index}_stack")
        p.storeProcessedScience()


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
    p.scaleByExposureTime()
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

