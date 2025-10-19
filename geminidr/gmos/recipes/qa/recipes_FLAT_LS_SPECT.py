"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.addToList(purpose='forStack')
    p.getList(purpose='forStack')
    p.stackFrames()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()

_default = makeProcessedFlat
