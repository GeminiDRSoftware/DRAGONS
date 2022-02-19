"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'LS', 'FLAT'].
These are GNIRS longslit observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFlats()
    p.normalizeFlat()
    p.thresholdFlatfield()
    # p.makeIRAFCompatible()
    p.storeProcessedFlat()

_default = makeProcessedFlat
