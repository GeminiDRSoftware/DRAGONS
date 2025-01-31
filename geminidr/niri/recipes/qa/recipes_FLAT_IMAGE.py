"""
Recipes available to data with tags ['NIRI', 'IMAGE', 'CAL', 'FLAT']
Default is "makeProcessedFlat".
"""
recipe_tags = {'NIRI', 'IMAGE', 'CAL', 'FLAT'}

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
    p.addToList(purpose='forFlat')
    p.getList(purpose='forFlat')
    p.makeLampFlat()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()
    return

    # TODO: Figure out where the makeProcessedBPM recipe belongs
    # TODO: Figure out how the makeProcessedBPM needs to be written

    # def makeProcessedBPM(p):
    #     """
    #     This recipe requires flats and *short* darks, not darks that match
    #     the exptime of the flats.
    #
    #     Maybe we can add an option to getProcessedDark to set the exptime.
    #     Maybe we can start with the short raw darsk and do something to
    #     getProcessedFlat, but then we need some mess to select the filter.
    #
    #     There isn't a clear solution at this time.  So the recipe is fully
    #     commented out for now.
    #     """
    #
    #     p.prepare()
    #     p.ADUToElectrons()
    #     p.separateFlatsDarks()
    #     p.stackDarks()
    #     p.makeLampFlat(stream=flats)
    #     p.normalizeFlat()
    #     p.makeBPM()
    #     p.storeBPM()
    #     return

_default = makeProcessedFlat
