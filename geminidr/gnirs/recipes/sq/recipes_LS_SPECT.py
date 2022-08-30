"""
Recipes available to data with tags ['GNIRS', 'SPECT', LS'].
Default is "reduceScience".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS'}

def reduceScience(p):
    """
    To be updated as development continues: This recipe processes GNIRS longslit 
    spectroscopic data, currently up to basic spectral extraction without telluric correction.

    Parameters
    ----------
    p : :class:`geminidr.gnirs.primitives_gnirs_longslit.GNIRSLongslit`

    """
    p.prepare()
    p.addDQ()
    # p.nonlinearityCorrect() # non-linearity correction tbd 
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    # p.darkCorrect() # no dark correction for GNIRS LS data 
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.distortionCorrect()
    p.writeOutputs()
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
