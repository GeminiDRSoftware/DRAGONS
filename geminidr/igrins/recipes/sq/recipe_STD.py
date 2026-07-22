"""
"""
recipe_tags = {'IGRINS-2', 'STANDARD'}

def makeStellar(p):
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

    p.makeAB() # This will make stacked A-B and do the reference pixel correction.
    p.estimateSlitProfile()
    p.extractSpectra()

    p.saveTwodspec()
    p.saveDebugImage()

    return

def makeStellarNew(p):
    #p.checkCALDB(caltypes=["processed_flat", "processed_arc"])
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True, poisson_noise=True)
    p.ADUToElectrons()
    #p.nonlinearityCorrect()
    p.makeABNew()  # this will make stacked A-B and do the reference pixel correction.
    p.cleanReadout()
    p.flatCorrect()  # cuts as well
    p.attachWavelengthSolution()
    return
    p.estimateSlitProfile()

    # Here's where we deviate from the IGRINSDR recipe
    p.distortionCorrect(outstream="xshifted")
    p.extractSpectrumUsingProfile(stream="xshifted")
    # Copy the slit profile map
    p.transferAttribute(stream="xshifted", source="main", attribute="SLITPROFILE_MAP")
    p.makeSyntheticImage(stream="xshifted")
    p.transferAttribute(source="xshifted", attribute="data", new_name="SYNTHMAP")
    p.clearStream(stream="xshifted")
    p.flagDiscrepantPixels()
    p.distortionCorrect()
    p.createDataCube(outstream="2D")
    p.storeProcessedScience(stream="2D", suffix="_2D")
    p.extractSpectrumUsingProfile()
    p.storeProcessedScience(suffix="_1D")


def makeStd(p):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    makeStellar(p)
    # normalize the spectra
    #p.storeProcessedStandard()

_default = makeStd
