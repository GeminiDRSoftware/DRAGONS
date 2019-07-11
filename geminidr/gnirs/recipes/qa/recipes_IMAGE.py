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
    p : PrimitivesBASE object
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
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureCC()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.separateSky()
    p.associateSky()
    p.skyCorrect(mask_objects=False)
    p.detectSources()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.applyDQPlane()
    p.detectSources()
    p.measureIQ(display=True)
    p.determineAstrometricSolution()
    p.measureCC()
    p.writeOutputs()
    return

_default = reduce
