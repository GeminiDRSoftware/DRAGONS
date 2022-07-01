"""
Recipes available to data with tags ['GNIRS', 'SPECT', LS'].
Default is "reduceScience".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS'}

def reduceScience(p):
    """
    To be updated as development continues: This recipe processes GNIRS longslit 
    spectroscopic data, currently up to wavelength calibration.

    Parameters
    ----------
    p : :class:`geminidr.gnirs.primitives_gnirs_longslit.GNIRSLongslit`

    """
    p.prepare()
    p.addDQ()
    # p.nonlinearityCorrect() # non-linearity correction tbd 
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.darkCorrect()
    # p.writeOutputs() # delete me after dark correction verification
    p.flatCorrect()
    # p.writeOutputs() # delete me after flat correction verification
    p.attachWavelengthSolution()
    # p.writeOutputs() # delete me after wavecal verification
    # TBD below 
    p.distortionCorrect()
    # p.writeOutputs()
    # p.separateSky()
    # p.associateSky()
    # p.skyCorrect()
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
