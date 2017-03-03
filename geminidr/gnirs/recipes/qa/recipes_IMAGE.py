"""
Recipes available to data with tags ['GNIRS', 'IMAGE']
Default is "reduce".
"""
recipe_tags = set(['GNIRS', 'IMAGE'])

def reduce(p):
    """
    This recipe process GNIRS keyhole imaging data and stack them in
    a single stacked output image is produced.
    It will attempt flat correction if a processed calibration
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
    p.flatCorrect()
    p.applyDQPlane()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.makeSky()
    p.skyCorrect()
    p.alignAndStack()
    p.applyDQPlane()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureCCAndAstrometry()
    p.writeOutputs()
    return

default = reduce
