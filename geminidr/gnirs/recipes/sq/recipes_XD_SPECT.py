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
    p.cleanReadout()
    p.attachPinholeModel()
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
    # p.nonlinearityCorrect() # non-linearity correction tbd
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.cleanReadout()
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames(scale=False, zero=False)
    p.findApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.extractSpectra()
    p.fitTelluric()
    p.storeProcessedTelluric()


def reduceScienceWithAdjustmentFromSkylines(p):
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
    p.selectFromInputs(outstream="temp")
    p.flatCorrect(stream="temp")
    p.attachWavelengthSolution(stream="temp")
    p.adjustWavelengthZeroPoint(stream="temp", shift=None)
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.cleanReadout()
    p.flatCorrect()
    p.transferAttribute(stream="main", source="temp", attribute="wcs")
    p.distortionCorrect()
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.scaleCountsToReference()
    p.stackFrames()
    p.findApertures()
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extractSpectra()
    p.storeProcessedScience(suffix="_1D")

_default = reduceScience
