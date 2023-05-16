"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'NODANDSHUFFLE'].

The default recipe is "reduce".

"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'NODANDSHUFFLE'}


def reduce(p):
    """
    This recipe reduces GMOS N&S longslit science data.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.attachWavelengthSolution()
    p.flatCorrect()
    p.flagCosmicRays()
    p.skyCorrectNodAndShuffle()
    p.QECorrect()
    p.flushPixels()
    p.distortionCorrect()
    p.combineNodAndShuffleBeams()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(conserve=True)  # default force_linear=True, ie. linearized.
    p.stackFrames()
    p.findApertures()
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extractSpectra()
    p.fluxCalibrate()
    p.storeProcessedScience(suffix="_1D")


_default = reduce
