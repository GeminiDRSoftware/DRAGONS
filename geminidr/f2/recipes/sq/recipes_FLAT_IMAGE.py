"""
Recipes available to data with tags ['F2', 'IMAGE', 'CAL', 'FLAT']
Default is "makeProcessedFlat".
"""
recipe_tags = {'F2', 'IMAGE', 'CAL', 'FLAT'}


def makeProcessedFlat(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input flat images into a single stacked and normalized flat image.
    This output processed flat is stored on disk using storeProcessedFlat and
    has a name equal to the name of the first input flat image with "_flat.fits"
    appended.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.makeLampFlat()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()
    return


def makeProcessedBPM(p):
    """
    This recipe requires flats and *short* darks, not darks that match
    the exptime of the flats.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.ADUToElectrons()  # needed to allow subtracting processed dark in K band
    p.selectFromInputs(tags="DARK", outstream="darks")
    p.selectFromInputs(tags="FLAT")
    p.stackFrames(stream="darks")
    p.makeLampFlat()
    p.normalizeFlat()
    p.makeBPM()
    #p.storeBPM()
    return


_default = makeProcessedFlat
