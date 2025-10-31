"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'XD', 'PINHOLE'].
These are GNIRS cross-dispersed (XD) pinholes.
Default is "makeProcessedPinhole".
"""
recipe_tags = {'GNIRS', 'SPECT', 'XD', 'PINHOLE'}

def makeProcessedPinhole(p):
    """
    Create a processed pinhole file for GNIRS cross-disersed data.
    Inputs are:
        * raw PINHOLE observations
        * processed FLAT
    No darks are needed due to the short exposures. It was found that using
    darks was just adding to the noise.

    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.applySlitModel()
    p.stackFrames()
    p.determinePinholeRectification()
    p.storeProcessedPinhole()

_default = makeProcessedPinhole
