makeProcessedSlitIllum
======================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_FLAT_LS_SPECT
| **Astrodata Tags**: {'LS', 'GMOS', 'FLAT', 'SPECT'}
::

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
        p.makeIRAFCompatible()
        p.storeProcessedSlitIllum()
