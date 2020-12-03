"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduceScience".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS'}
blocked_tags = {'NODANDSHUFFLE'}


def reduceScience(p):
    """
    todo: add docstring

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.QECorrect()
    p.distortionCorrect()
    p.findSourceApertures()
    p.skyCorrectFromSlit()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.findSourceApertures()
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extract1DSpectra()
    p.fluxCalibrate()
    p.linearizeSpectra()
    p.storeProcessedScience(suffix="_1D")



def reduceStandard(p):
    """
    todo: add docstring

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.QECorrect()
    p.distortionCorrect()
    p.findSourceApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.extract1DSpectra()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.calculateSensitivity()
    #p.linearizeSpectra()  # TODO: needed?
    p.storeProcessedStandard()
    p.writeOutputs()



_default = reduceScience
