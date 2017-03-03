"""
Recipes available to data with tags ['F2', 'IMAGE']
Default is "reduce_nostack".
"""
recipe_tags = set(['F2', 'IMAGE'])

def reduce(p):
    """
    This recipe process F2 data up to and including alignment and stacking.
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
    p.darkCorrect()
    p.flatCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.subtractSkyBackground()
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
    p.writeOutputs()
    return


def reduce_nostack(p):
    """
    This recipe process F2 data up to but not including alignment and stacking.
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
    p.darkCorrect()
    p.flatCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.subtractSkyBackground()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.makeSky()
    p.skyCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureCCAndAstrometry()
    p.writeOutputs()
    return

# The nostack version is used because stacking was too slow for QAP.
# KL: Is this still true with gemini_python 2.0?
default = reduce_nostack
