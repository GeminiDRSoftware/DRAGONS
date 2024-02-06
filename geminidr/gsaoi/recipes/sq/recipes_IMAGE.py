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
    p.detectSources()
    p.writeOutputs()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.storeProcessedScience(suffix="_image")
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
    p.detectSources()
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
    p.transferObjectMask(stream='flat_corrected', source='main')
    p.clearStream(stream='main')
    p.dilateObjectMask(instream='flat_corrected', outstream='main', dilation=2)


def ultradeep_part3(p):
    """
    This recipe takes flat-corrected images with OBJMASKs as inputs and
    produces a final stack. It should take the _objectMaskDilated outputs
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


# The nostack version is used because stacking of GSAOI is time consuming.
_default = reduce
