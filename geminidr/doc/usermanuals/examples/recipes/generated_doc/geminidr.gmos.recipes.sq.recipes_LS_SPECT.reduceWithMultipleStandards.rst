reduceWithMultipleStandards
===========================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_LS_SPECT
| **Astrodata Tags**: {'LS', 'GMOS', 'SPECT'}

todo: add docstring

::

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`


::

    def reduceWithMultipleStandards(p):
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.attachWavelengthSolution()
        p.flatCorrect()
        p.QECorrect()
        p.flagCosmicRays()
        p.distortionCorrect()
        p.findApertures()
        p.skyCorrectFromSlit()
        p.adjustWCSToReference()
        p.fluxCalibrate()
        p.resampleToCommonFrame(conserve=True)  # default force_linear=True, ie. linearized.
        p.scaleCountsToReference()
        p.stackFrames()
        p.findApertures()
        p.traceApertures()
        p.storeProcessedScience(suffix="_2D")
        p.extractSpectra()
        p.storeProcessedScience(suffix="_1D")
