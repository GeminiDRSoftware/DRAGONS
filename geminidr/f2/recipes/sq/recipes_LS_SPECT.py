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
    
    
_default = reduceScience
