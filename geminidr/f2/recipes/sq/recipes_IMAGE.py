"""
Recipes available to data with tags ['F2', 'IMAGE'].
Default is "reduce".
"""
recipe_tags = {'F2', 'IMAGE'}


def reduce(p):
    """
    This recipe process F2 data up to and including alignment and stacking.
    A single stacked output image is produced.
    It will attempt to do dark and flat correction if a processed calibration
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
    p.flushPixels()
    p.separateSky()
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.flushPixels()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleByExposureTime()
    p.stackFrames()
    p.writeOutputs()
    p.storeProcessedScience()
    return


def makeSkyFlat(p):
    """
    This recipe makes a flatfield image from a series of dithered sky images.

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
    # Make a "fastsky" by combining frames
    p.stackFrames(operation='median', scale=True, outstream='fastsky')
    p.normalizeFlat(stream='fastsky')
    p.thresholdFlatfield(stream='fastsky')
    # Flatfield with the fastsky and find objects
    p.flatCorrect(flat=p.streams['fastsky'][0], outstream='flattened')
    p.detectSources(detect_minarea=20, stream='flattened')
    p.dilateObjectMask(dilation=10, stream='flattened')
    p.addObjectMaskToDQ(stream='flattened')
    p.writeOutputs(stream='flattened')
    p.transferAttribute(source='flattened', attribute='mask')
    p.clearStream(stream='flattened')
    p.scaleByIntensity()
    p.stackFrames(operation='average', reject_method="minmax", nlow=0, nhigh=1)
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()
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
    p.scaleByExposureTime()
    p.stackFrames()
    return


def makeIRAFCompatible(p):
    """
    Add header keywords needed to run some Gemini IRAF tasks.  This is needed
    only if the reduced file will be used as input to Gemini IRAF tasks.

    Parameters
    ----------
    p : PrimitivesBASEE object
        A primitive set matching the recipe_tags.
    """

    p.makeIRAFCompatible()
    p.writeOutputs()
    return