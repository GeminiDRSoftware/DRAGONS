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
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.flatCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky', max_frames=8)
    p.makeSky()
    p.skyCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureCCAndAstrometry()
    p.writeOutputs(p)
    return

# The nostack version is used because stacking of GSAOI is time consuming.
default = reduce_nostack
