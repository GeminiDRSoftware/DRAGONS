"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS'}
blocked_tags = {'NODANDSHUFFLE'}

from time import sleep

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
    p.attachWavelengthSolution()
    p.flatCorrect()
    p.QECorrect()
    p.distortionCorrect()
    p.measureIQ(display=True)
    p.findApertures(max_apertures=10)
    p.skyCorrectFromSlit()
    p.measureIQ(display=True)

    # side stream to generate 1D spectra from individual frame, pre-stack
    p.traceApertures(outstream='prestack')
    p.extractSpectra(stream='prestack')
    p.fluxCalibrate(stream='prestack')
    p.plotSpectraForQA(stream='prestack')
    # The GUI polls for new data every 3 seconds.  The next steps can be
    # quicker than that leading to the second plotSpectra to hijack this one.
    # Hijacking issues were highlighted in integration tests.
    sleep(3)

    # continuing with main stream of 2D pre-stack.
    p.addToList(purpose='forStack')
    p.getList(purpose='forStack', max_frames=10)
    p.adjustWCSToReference()
    p.resampleToCommonFrame()  # default force_linear=True, ie. linearized.
    p.scaleCountsToReference()
    p.stackFrames()
    p.findApertures()
    p.measureIQ(display=True)
    p.traceApertures()
    p.extractSpectra()
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
    p.attachWavelengthSolution()
    p.flatCorrect()
    p.QECorrect()
    p.distortionCorrect()
    p.findApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.measureIQ(display=True)
    p.traceApertures()
    p.extractSpectra()
    p.plotSpectraForQA()
    # The GUI polls for new data every 3 seconds.  The next steps can be
    # quicker than that leading to the second plotSpectra to hijack this one.
    # Hijacking issues were highlighted in integration tests.
    sleep(3)

    p.addToList(purpose='forStack')
    p.getList(purpose='forStack', max_frames=10)
    p.resampleToCommonFrame()  # default force_linear=True, ie. linearized.
    p.scaleCountsToReference()
    p.stackFrames()
    p.plotSpectraForQA()
    p.calculateSensitivity()
    p.storeProcessedStandard()
