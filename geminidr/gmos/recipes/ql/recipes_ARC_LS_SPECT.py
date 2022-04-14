"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'ARC'].
These are GMOS longslit arc-lamp calibrations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'ARC'}

from geminidr.gmos.recipes.ql.recipes_common import makeIRAFCompatible

def makeProcessedArc(p):
    p.prepare()
    p.addDQ()
    p.maskAmp5()
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
