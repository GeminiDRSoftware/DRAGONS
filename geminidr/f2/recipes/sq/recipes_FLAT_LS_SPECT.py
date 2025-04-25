"""
Recipes available to data with tags ['F2', 'SPECT', LS', 'FLAT''].
These are F2 longslit observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'F2', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    """
    Create F2 longslit flat field.
    Inputs are:
       * raw lamp-on flats + raw darks.
       * raw lamp-on flats -> then will look in caldb for matching procdark.
    """
    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.makeLampFlat()
    p.determineSlitEdges()
    p.maskBeyondSlit()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.makeIRAFCompatible()
    p.storeProcessedFlat()

_default = makeProcessedFlat
