"""
Recipes available to data with tags ['IGRINS-2', 'SPECT', 'XD''].
Default is "reduceScience".
"""
recipe_tags = {'IGRINS-2', 'SPECT', 'XD'}


def reduceScience(p):
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.ADUToElectrons()
    #p.nonlinearityCorrect()
    p.flexureCorrect()
    p.makeAB()
    p.cleanReadout()
    p.flatCorrect()  # cuts as well
    p.attachWavelengthSolution()
    p.distortionCorrect(outstream="2D", interpolant="linear")
    p.writeOutputs(stream="2D", suffix="_2D")
    p.measureSlitProfile(stream="2D")
    p.transferAttribute(source="2D", attribute="SLITPROF")
    p.extractSpectra()
    p.writeOutputs()
    p.telluricCorrect()
    p.fluxCalibrate()
    p.storeProcessedScience(suffix="_1D")


def reduceTelluric(p):
    #p.checkCALDB(caltypes=["processed_flat", "processed_arc"])
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.ADUToElectrons()
    #p.nonlinearityCorrect()
    p.flexureCorrect()
    p.makeAB()
    p.cleanReadout()
    p.flatCorrect()  # cuts as well
    p.attachWavelengthSolution()
    p.distortionCorrect(outstream="2D", interpolant="linear")
    #p.writeOutputs(stream="2D", suffix="_2D")
    p.measureSlitProfile(stream="2D")
    p.transferAttribute(source="2D", attribute="SLITPROF")
    p.extractSpectra()
    p.fitTelluric()
    p.storeProcessedTelluric()


_default = reduceScience


def makeArcFromScience(p):
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.ADUToElectrons()
    p.removeObjectsLeaveSky()
    p.writeOutputs()
    p.applySlitModel()
    p.determineWavelengthSolution()
    p.writeOutputs()
    #p.determineDistortion()


def oldMakeStellar(p):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.checkCALDB(caltypes=["processed_flat", "processed_arc"])
    p.prepare(require_wcs=False)
    p.addDQ()

    # FIXME we need to figure out how to inject badpix mask from the flat.

    # FIXME check if read-noise is really added. It seems not.
    p.addVAR(read_noise=True, poisson_noise=True)

    # ADUToElectrons requires saturation_level and nonlinearity_level in the
    # header. Since IGRINS does not have these values defined, we add them
    # here.
    # p.fixIgrinsHeader()
    p.ADUToElectrons()
    #p.nonlinearityCorrect()

    p.makeABOld() # This will make stacked A-B and do the reference pixel correction.
    p.estimateSlitProfile()
    p.extractSpectraSingle()

    p.saveTwodspec()
    p.saveDebugImage()

    return
