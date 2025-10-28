"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'SLITILLUM'].
These are GMOS longslit observations.

Default is "makeProcessedSlitIllum".

"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'SLITILLUM'}


def makeProcessedSlitIllum(p):
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames()
    p.makeSlitIllum()
    p.storeProcessedSlitIllum()


_default = makeProcessedSlitIllum
