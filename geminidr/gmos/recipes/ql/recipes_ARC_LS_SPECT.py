"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'ARC'].
These are GMOS longslit arc-lamp calibrations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'ARC'}

from geminidr.gmos.recipes.ql.recipes_common import makeIRAFCompatible

def makeProcessedArc(p):
    p.prepare()
    p.maskFaultyAmp(instrument='GMOS-S', bad_amps=5, valid_from='20220128')
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion()
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
