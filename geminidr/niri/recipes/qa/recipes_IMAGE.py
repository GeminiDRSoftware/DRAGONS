"""
Recipes available to data with tags ['NIRI', 'IMAGE'].
Default is "reduce".
"""
recipe_tags = {'NIRI', 'IMAGE'}

def reduce(p):
    """
    This recipe process NIRI data up to and including alignment and stacking.
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
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureCC()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.separateSky()
    p.associateSky()
    p.skyCorrect(mask_objects=False)
    p.detectSources()
    p.measureIQ(display=True)
    p.determineAstrometricSolution()
    p.measureCC()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.detectSources()
    p.measureIQ(display=True)
    p.determineAstrometricSolution()
    p.measureCC()
    p.writeOutputs()
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
    p.scaleByIntensity(outstream='fastsky')
    p.stackFrames(operation='median', stream='fastsky')
    p.normalizeFlat(stream='fastsky')
    p.thresholdFlatfield(stream='fastsky')
    # Flatfield with the fastsky and find objects
    p.flatCorrect(flat=p.streams['fastsky'][0], outstream='flattened')
    p.detectSources(detect_minarea=20, stream='flattened')
    p.dilateObjectMask(dilation=10, stream='flattened')
    p.addObjectMaskToDQ(stream='flattened')
    p.writeOutputs(stream='flattened')
    p.transferAttribute(source='flattened', attribute='mask')
    p.scaleByIntensity()
    p.stackFrames(operation='mean', nlow=0, nhigh=1)
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()
    return

_default = reduce
