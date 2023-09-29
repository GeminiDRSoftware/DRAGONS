makeProcessedArc
================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_ARC_LS_SPECT
| **Astrodata Tags**: {'ARC', 'GMOS', 'LS', 'SPECT'}
::

    def makeProcessedArc(p):
        p.prepare()
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

