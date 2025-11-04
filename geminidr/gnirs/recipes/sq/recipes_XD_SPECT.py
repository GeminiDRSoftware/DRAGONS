"""
Recipes available to data with tags ['GNIRS', 'SPECT', XD'].
Default is "reduceScience".
"""
recipe_tags = {'GNIRS', 'SPECT', 'XD'}

def reduceScience(p):
    """
    To be updated as development continues: This recipe processes GNIRS cross-dispersed
    spectroscopic data, currently up to basic spectral extraction without telluric correction.

    Parameters
    ----------
    p : :class:`geminidr.gnirs.primitives_gnirs_crossdispersed.GNIRSCrossDispersed`

    """
    p.prepare()
    p.addDQ()
    # p.nonlinearityCorrect() # non-linearity correction tbd
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    # p.darkCorrect() # no dark correction for GNIRS data
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    # p.cleanReadout()  # algorithm doesn't work on cut data
    p.attachPinholeRectification()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(conserve=True)
    # p.scaleCountsToReference()  not in NIR LS recipes.
    p.stackFrames(scale=False, zero=False)
    p.findApertures()
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extractSpectra()
    p.telluricCorrect()
    p.fluxCalibrate()
    p.storeProcessedScience(suffix="_1D")


def reduceTelluric(p):
    """
    Reduce GNIRS longslit observations of a telluric standard, including
    fitting the telluric absorption features to provide an absorption profile
    and a sensitivity function.

    Parameters
    ----------
    p : :class:`geminidr.gnirs.primitives_gnirs_longslit.GNIRSLongslit`
    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    # p.cleanReadout()  # algorithm doesn't work on cut data
    p.attachPinholeRectification()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames(scale=False, zero=False)
    p.findApertures(max_apertures=1)
    #p.skyCorrectFromSlit()  # tellurics are bright, no need to do this
    p.traceApertures()
    p.extractSpectra()
    p.fitTelluric()
    p.storeProcessedTelluric()


def  makeWavecalFromSkyEmission(p):
    """
    Process GNIRS XD science in order to create wavelength and distortion
    solutions using sky emission lines.

    Inputs are:
      * raw science
      * processed flat
    """

    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    p.flatCorrect()
    p.attachPinholeRectification()
    p.stackFrames()
    p.determineWavelengthSolution()
    p.determineDistortion()
    p.storeProcessedArc(force=True)
    p.writeOutputs()


def  makeWavecalFromSkyAbsorption(p):
    """
    Process GNIRS longslist science in order to create wavelength solution
    using telluric absorption in the target spectrum. Copy distortion model
    to the resulting calibration frame from the associated arc.

    Inputs are:
      * raw science
      * processed arc
      * processed flat
    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.copyInputs(instream="main", outstream="with_distortion_model")
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    # p.cleanReadout()  # algorithm doesn't work on cut data
    p.attachPinholeRectification()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(output_wave_scale='reference', trim_spectral=True)
    p.stackFrames()
    p.findApertures()
    p.determineWavelengthSolution(absorption=True)
    p.transferDistortionModel(source="with_distortion_model")
    p.storeProcessedArc(force=True)


_default = reduceScience
