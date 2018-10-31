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
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.flatCorrect()
    p.flushPixels()
    #p.detectSources()
    #p.measureIQ(display=True)
    #p.writeOutputs()
    #p.measureBG()
    #p.addReferenceCatalog()
    #p.determineAstrometricSolution()
    #p.measureCC()
    #p.addToList(purpose='forSky')
    #p.getList(purpose='forSky', max_frames=9)
    p.separateSky()
    p.associateSky(stream='sky')
    p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')
    p.detectSources(stream='skysub')
    p.transferAttribute(stream='sky', source='skysub', attribute='OBJMASK')
    p.clearStream(stream='skysub')
    p.associateSky()
    p.skyCorrect(mask_objects=True)
    #p.measureIQ(display=True)
    #p.determineAstrometricSolution()
    #p.measureCC()
    p.writeOutputs()
    return

# The nostack version is used because stacking of GSAOI is time consuming.
default = reduce_nostack
