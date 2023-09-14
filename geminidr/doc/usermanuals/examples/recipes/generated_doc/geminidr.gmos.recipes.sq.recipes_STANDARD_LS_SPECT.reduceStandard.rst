reduceStandard
==============

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_STANDARD_LS_SPECT
| **Recipe Imported From**: geminidr.gmos.recipes.sq.recipes_LS_SPECT
| **Astrodata Tags**: {'LS', 'GMOS', 'STANDARD', 'SPECT'}

todo: add docstring

::

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`


::

    def reduceStandard(p):
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
        p.distortionCorrect()
        p.findApertures(max_apertures=1)
        p.skyCorrectFromSlit()
        p.traceApertures()
        p.extractSpectra()
        p.resampleToCommonFrame(conserve=True)  # default force_linear=True, ie. linearized.
        p.scaleCountsToReference()
        p.stackFrames()
        p.calculateSensitivity()
        p.storeProcessedStandard()
        p.writeOutputs()

