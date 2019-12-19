"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS', 'FLAT'])


def makeProcessedFlat(p):
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.makeIRAFCompatible()
    p.storeProcessedFlat()


_default = makeProcessedFlat
