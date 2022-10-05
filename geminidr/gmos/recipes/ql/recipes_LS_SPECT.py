"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
These are GMOS longslit observations.
Default is "reduceScience".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS'}

from geminidr.gmos.recipes.ql.recipes_common import makeIRAFCompatible

def reduceScience(p):
    """
    todo: add docstring

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.maskFaultyAmp(instrument='GMOS-S', bad_amps=5, valid_from='20220128')
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.getProcessedBias()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.getProcessedFlat()
    p.flatCorrect()
    p.getProcessedArc()
    p.QECorrect()
    p.distortionCorrect()
    p.findApertures()
    p.skyCorrectFromSlit()
    p.adjustWCSToReference()
    p.resampleToCommonFrame(conserve=True)  # default force_linear=True, ie. linearized.
    p.stackFrames()
    p.findApertures()
    p.traceApertures()
    p.storeProcessedScience(suffix="_2D")
    p.extractSpectra()
    p.getProcessedStandard()
    p.fluxCalibrate()
    p.storeProcessedScience(suffix="_1D")



def reduceStandard(p):
    """
    todo: add docstring

    Parameters
    ----------
    p : :class:`geminidr.gmos.primitives_gmos_longslit.GMOSLongslit`

    """
    p.prepare()
    p.maskFaultyAmp(instrument='GMOS-S', bad_amps=5, valid_from='20220128')
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.getProcessedBias()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.getProcessedFlat()
    p.flatCorrect()
    p.getProcessedArc()
    p.QECorrect()
    p.distortionCorrect()
    p.findApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.traceApertures()
    p.extractSpectra()
    p.resampleToCommonFrame(conserve=True)  # default force_linear=True, ie. linearized.
    p.stackFrames()
    p.calculateSensitivity()
    p.storeProcessedStandard()
    p.writeOutputs()

_default = reduceScience

