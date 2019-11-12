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
    p : PrimitivesBASE object
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
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureCC()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.determineAstrometricSolution()
    p.measureCC()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.detectSources()
    p.determineAstrometricSolution()
    p.measureIQ(display=True)
    p.measureCC()
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
    p : PrimitivesBASE object
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
    p.writeOutputs()
    return

# The nostack version is used because stacking was too slow for QAP.
# KL: Is this still true with gemini_python 2.0?
_default = reduce_nostack
