"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduceScience".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS'])


def reduceScience(p):
    """
    todo: add docstring

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.applyQECorrection()
    p.distortionCorrect()
    p.writeOutputs()
    p.findSourceApertures()
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.extract1DSpectra()
    p.linearizeSpectra()
    p.writeOutputs()


def reduceStandard(p):
    """
    todo: add docstring

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.applyQECorrection()
    p.distortionCorrect()
    p.findSourceApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.extract1DSpectra()
    p.linearizeSpectra()  # TODO: needed?
    p.calculateSensitivity()
    p.writeOutputs()


_default = reduceScience
