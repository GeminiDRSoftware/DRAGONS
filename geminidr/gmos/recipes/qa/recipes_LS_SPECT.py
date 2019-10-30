"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS'])


def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image (TODO: stacking).
    The image quality is measured at different points during the reduction.

    Parameters
    ----------
    p : :class:`geminidr.core.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.measureIQ(display=True)
    p.flatCorrect()
    p.distortionCorrect()
    # Some sort of stacking here, with addToList() etc
    p.findSourceApertures()
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.measureIQ(display=True)
    p.extract1DSpectra()
    p.linearizeSpectra()
    p.writeOutputs()


_default = reduce