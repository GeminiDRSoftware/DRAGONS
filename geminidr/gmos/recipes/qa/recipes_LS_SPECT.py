"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS'}


def reduceScience(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image (TODO: stacking).
    The image quality is measured at different points during the reduction.

    Parameters
    ----------
    p : :class:`geminidr.core.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.measureIQ(display=True)
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.QECorrect()
    p.distortionCorrect()
    p.measureIQ(display=True)
    p.findSourceApertures()
    p.skyCorrectFromSlit()
    p.measureIQ(display=True)

    # side stream to generate 1D spectra from individual frame, pre-stack
    p.traceApertures(outstream='prestack')
    p.extrac1DSpectra(stream='prestack')
    p.fluxCalibrate(stream='prestack')
    p.plotSpectraForQA(stream='prestack')

    # continuing with main stream of 2D pre-stack.
    p.addToList(purpose='forStack')
    p.getList(purpose='forStack')
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.findSourceApertures()
    p.measureIQ(display=True)
    p.traceApertures()
    p.extract1DSpectra()
    p.fluxCalibrate()
    p.plotSpectraForQA()


_default = reduceScience


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
    p.measureIQ(display=True)
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.QECorrect()
    p.distortionCorrect()
    p.findSourceApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.measureIQ(display=True)
    p.traceApertures()
    p.extract1DSpectra()
    p.plotSpectraForQA()
    p.addToList(purpose='forStack')
    p.getList(purpose='forStack')
    p.resampleToCommonFrame()
    p.stackFrames()
    p.plotSpectraForQA()
    p.calculateSensitivity()
    p.storeProcessedStandard()
