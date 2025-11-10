"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'FLAT'}

from geminidr.gmos.recipes.sq.recipes_common import makeIRAFCompatible

def makeProcessedFlatNoStack(p):
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()


_default = makeProcessedFlatNoStack


def makeProcessedFlatStack(p):
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()


def makeProcessedSlitIllum(p):
    p.prepare(require_wcs=False)
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames()
    p.makeSlitIllum()
    p.storeProcessedSlitIllum()
