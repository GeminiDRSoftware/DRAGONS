"""
Recipes available to data with tags ['GNIRS', 'SPECT', LS'].
Default is "reduceScience".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS'}


def reduceScience(p):
    """
    Reduce GNIRS longslit spectroscopic data, including stacking, extraction,
    telluric correction, and flux calibration.

    Parameters
    ----------
    p : :class:`geminidr.gnirs.primitives_gnirs_longslit.GNIRSLongslit`
    """
    # todo: [Chris] I suspect scaleCountsToReference() could go back in but presumably hasn't been looked at whether it does something weird. skyCorrectFromSlit() should probably work OK, but it might also be better if it's before adjustWCSToReference(), so any residual sky is removed before that resampling takes place.
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    # p.darkCorrect() # no dark correction for GNIRS LS data
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.adjustWavelengthZeroPoint()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.cleanReadout()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(conserve=True)
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
    p.adjustWavelengthZeroPoint()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.cleanReadout()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.findApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.extractSpectra()
    p.fitTelluric()
    p.storeProcessedTelluric()


def  makeWavecalFromSkyEmission(p):
    """
    Process GNIRS longslist science in order to create wavelength and distortion
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
    p.cleanReadout()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(output_wave_scale='reference', trim_spectral=True)
    p.stackFrames()
    p.findApertures()
    p.determineWavelengthSolution(absorption=True)
    p.transferDistortionModel(source="with_distortion_model")
    p.storeProcessedArc(force=True)

_default = reduceScience
