def reduce(p):
    """
    This recipe process NIRI data up to and including alignment and stacking.
    A single stacked output image is produced.
    It will attempt to do dark and flat correction if a processed calibration
    is available.  Sky subtraction is done when possible.  QA metrics are
    measured.

    Parameters
    ----------
    p : PrimitivesCORE object
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
    p.measureCCAndAstrometry()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.makeSky()
    p.skyCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureCCAndAstrometry()
    p.alignAndStack()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureCCAndAstrometry()
    return

