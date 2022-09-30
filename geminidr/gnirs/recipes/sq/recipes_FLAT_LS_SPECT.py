"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'LS', 'FLAT'].
These are GNIRS longslit observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    """
    Create a processed flat for GNIRS longslit data.
    Inputs are:
      * raw LAMPON flats - no other calibrations required.
      (Questions remaining, see google doc)
    No darks are needed due to the short exposures.  It was found that using
    darks was just adding to the noise.
    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.stackFlats()
    p.determineSlitEdges()
    p.maskBeyondSlit()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.makeIRAFCompatible()
    p.storeProcessedFlat()

_default = makeProcessedFlat
