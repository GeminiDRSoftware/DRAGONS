"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS', 'FLAT'])

def reduce(p):
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    #p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    #p.stackFlats()
    p.normalizeFlat()
    #p.makeIRAFCompatible()
    #p.storeProcessedFlat()
    p.writeOutputs()

_default = reduce
