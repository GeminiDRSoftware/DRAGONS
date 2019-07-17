"""
Recipes available to data with tags ['GSAOI', 'IMAGE']
Default is "reduce_nostack".
"""
recipe_tags = set(['GSAOI', 'IMAGE'])

def reduce_nostack(p):
    """
    This recipe reduce GSAOI up to but NOT including alignment and stacking.
    It will attempt to do flat correction if a processed calibration is
    available.  Sky subtraction is done when possible.  QA metrics are
    measured.

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
    p.detectSources(detect_thresh=5., analysis_thresh=5., back_size=128)
    p.measureIQ(display=True)
    p.measureBG()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureCC()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky', max_frames=9)
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.detectSources(detect_thresh=5., analysis_thresh=5., back_size=128)
    p.measureIQ(display=True)
    p.determineAstrometricSolution()
    p.measureCC()
    p.writeOutputs()
    return

# The nostack version is used because stacking of GSAOI is time consuming.
_default = reduce_nostack
