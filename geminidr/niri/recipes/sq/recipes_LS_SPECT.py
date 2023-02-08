"""
Recipes available to data with tags ['NIRI', 'SPECT', LS'].
Default is "reduceScience".
"""
recipe_tags = {'NIRI', 'SPECT', 'LS'}

def reduceScience(p):
    """
    To be updated as development continues: This recipe processes NIRI longslit
    spectroscopic data, currently up to a basic spectral extraction without telluric correction.

    Parameters
    ----------
    p : :class:`geminidr.niri.primitives_niri_longslit.NIRILongslit`

    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    # p.darkCorrect() # no dark correction for NIRI LS data
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.cleanReadout()
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


_default = reduceScience
