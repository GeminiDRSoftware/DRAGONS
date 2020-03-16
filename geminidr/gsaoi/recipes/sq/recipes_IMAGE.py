"""
Recipes available to data with tags ['GSAOI', 'IMAGE']
Default is "reduce_nostack".
"""
recipe_tags = set(['GSAOI', 'IMAGE'])

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
    p.writeOutputs()
    p.storeScience()
    return

# The nostack version is used because stacking of GSAOI is time consuming.
_default = reduce_nostack
