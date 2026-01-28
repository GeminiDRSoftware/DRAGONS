"""
Recipes available to data with tags ['F2', 'SPECT', LS'].
Default is "reduceScience".
"""
recipe_tags = {'F2', 'SPECT', 'LS'}

def reduceScience(p):
    """
    To be updated as development continues: This recipe processes F2 longslit
    spectroscopic data, currently up to basic extraction (no telluric correction).

    Parameters
    ----------
    p : :class:`geminidr.f2.primitives_f2_longslit.F2Longslit`

    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    p.darkCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.adjustWavelengthZeroPoint()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(conserve=True)
    # p.scaleCountsToReference()
    p.stackFrames()
    p.findApertures()
    p.skyCorrectFromSlit()  # This needs testing.
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extractSpectra()
    p.telluricCorrect()
    p.fluxCalibrate()
    p.storeProcessedScience(suffix="_1D")


def reduceTelluric(p):
    """
    To be updated as development continues: This recipe processes F2 longslit
    spectroscopic data, currently up to basic extraction (no telluric correction).

    Parameters
    ----------
    p : :class:`geminidr.f2.primitives_f2_longslit.F2Longslit`

    """
    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.darkCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.adjustWavelengthZeroPoint()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    # p.scaleCountsToReference()
    p.stackFrames()
    p.findApertures(max_apertures=1)
    p.skyCorrectFromSlit()  # This needs testing.
    p.traceApertures()
    p.extractSpectra()
    p.fitTelluric()
    p.storeProcessedTelluric()


def  makeWavecalFromSkyEmission(p):
    """
    Process F2 longslist science in order to create wavelength and distortion
    solutions using sky emission lines.

    Inputs are:
       * raw science with caldb requests:
          * procdark with matching exptime
          * procflat
    """

    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.darkCorrect()
    p.flatCorrect()
    p.stackFrames()
    p.determineWavelengthSolution()
    p.determineDistortion()
    p.storeProcessedArc(force=True)
    p.writeOutputs()

_default = reduceScience
