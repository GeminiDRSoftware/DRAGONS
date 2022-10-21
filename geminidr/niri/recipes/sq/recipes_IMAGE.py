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
    p.removeFirstFrame()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.nonlinearityCorrect()
    p.darkCorrect()
    p.flatCorrect()
    p.separateSky()
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.cleanReadout()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.storeProcessedScience(suffix="_image")
    return


def ultradeep(p):
    """
    This recipe process F2 data to produce a single final stacked image.
    It will attempt to do dark and flat correction if a processed calibration
    is available.
    It conducts an additional pass over and above the standard recipe, where
    objects are found in the full stack and then masked out in the individual
    inputs, to improve the quality of the sky subtraction. It is designed for
    deep on-source-dithered sequences.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.removeFirstFrame()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.nonlinearityCorrect()
    p.darkCorrect()
    p.flatCorrect()
    # A shortcut way to copy all the AD object to a new stream
    # since this doesn't modify the AD objects
    p.flushPixels(outstream='flat_corrected')
    p.separateSky()
    assert len(p.streams['main']) == len(p.streams['sky']), \
        "Sequence includes sky-only frames"
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.detectSources()
    p.adjustWCSToReference()
    # Transfer correct WCS to inputs
    p.transferAttribute(stream='flat_corrected', source='main', attribute='wcs')
    p.flushPixels()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.detectSources()
    p.writeOutputs()  # effectively the standard recipe output
    p.transferObjectMask(stream='flat_corrected', source='main')
    p.clearStream(stream='main')
    p.dilateObjectMask(instream='flat_corrected', outstream='main', dilation=2)
    p.clearStream(stream='flat_corrected')  # no longer needed
    p.separateSky()
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.flushPixels()
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.writeOutputs()
    p.storeProcessedScience(suffix="_image")


def ultradeep_part1(p):
    """
    This recipe simply performs the standard reduction steps to remove
    instrumental signatures from the inputs. It's intended to be run as
    a first step for ultradeep (three-pass) imaging reduction, to
    produce intermediate reduction products that do not need to be
    recreated if there is an issue with the initial reduction.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.removeFirstFrame()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.nonlinearityCorrect()
    p.darkCorrect()
    p.flatCorrect()


def ultradeep_part2(p):
    """
    This recipe takes _flatCorrected images from part 1 as input and
    continues the reduction to produce a stacked image. It then
    identifies sources in the stack and transfers the OBJMASK back to the
    individual input images, saving those to disk, ready for part 3.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.copyInputs(outstream='flat_corrected')
    p.separateSky()
    assert len(p.streams['main']) == len(p.streams['sky']), \
        "Sequence must not contain sky-only frames"
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.detectSources()
    p.adjustWCSToReference()
    # Transfer correct WCS to inputs
    p.transferAttribute(stream='flat_corrected', source='main', attribute='wcs')
    p.flushPixels()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.detectSources()
    p.writeOutputs()  # effectively the standard recipe output
    p.transferObjectMask(instream='flat_corrected', outstream='main', source='main')


def ultradeep_part3(p):
    """
    This recipe takes flat-corrected images with OBJMASKs as inputs and
    produces a final stack. It should take the _objmaskTransferred outputs
    from part 2.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.separateSky()
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.writeOutputs()
    p.storeProcessedScience(suffix="_image")


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
    p.removeFirstFrame()
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
    p.detectSources(stream='flattened')
    p.dilateObjectMask(dilation=10, stream='flattened')
    p.addObjectMaskToDQ(stream='flattened')
    #p.writeOutputs(stream='flattened')
    p.transferAttribute(source='flattened', attribute='mask')
    p.stackFrames(operation='mean', scale=True, reject_method="minmax", nlow=0, nhigh=1)
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat(force=True)
    return

_default = reduce


def alignAndStack(p):
    """
    This recipe stack already preprocessed data.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    return
