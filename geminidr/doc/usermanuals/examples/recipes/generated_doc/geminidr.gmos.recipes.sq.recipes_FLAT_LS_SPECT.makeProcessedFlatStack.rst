makeProcessedFlatStack
======================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_FLAT_LS_SPECT
| **Astrodata Tags**: {'LS', 'GMOS', 'FLAT', 'SPECT'}
::

    def makeProcessedFlatStack(p):
        p.prepare()
        p.addDQ()
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

