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
    p.nonlinearityCorrect() # non-linearity correction tbd even for gemini iraf
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
    p.findApertures()
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extractSpectra()
    p.storeProcessedScience(suffix="_1D")


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
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.flatCorrect()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc(force=True)
    p.writeOutputs()

def  makeWavecalFromSkyAbsorption(p):
    """
    Process F2 longslist science in order to create wavelength solution
    using telluric absorption in the target spectrum. Copy distortion model
    to the resulting calibration frame from the associated arc.

    Inputs are:
       * raw science with caldb requests:
          * procdark with matching exptime
          * procflat
          * processed arc
    """
    p.prepare()
    p.addDQ()
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.darkCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.copyInputs(instream="main", outstream="with_distortion_model")
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.cleanReadout()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(force_linear=False)
    p.scaleCountsToReference()
    p.stackFrames()
    p.findApertures()
    p.determineWavelengthSolution(absorption=True)
    p.transferDistortionModel(source="with_distortion_model")
    p.storeProcessedArc(force=True)

_default = reduceScience
