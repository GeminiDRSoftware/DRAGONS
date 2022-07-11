"""
Recipes available to data with tags ['NIRI', 'SPECT', 'LS', 'FLAT'].
These are GNIRS longslit observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'NIRI', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.darkCorrect()
    p.stackFlats()
    p.determineSlitEdges()
    p.maskBeyondSlit()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.makeIRAFCompatible()
    p.storeProcessedFlat()

_default = makeProcessedFlat
