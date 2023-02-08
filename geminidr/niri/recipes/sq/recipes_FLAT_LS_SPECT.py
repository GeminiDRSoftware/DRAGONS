"""
Recipes available to data with tags ['NIRI', 'SPECT', 'LS', 'FLAT'].
These are GNIRS longslit observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'NIRI', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    """Create a processed flat for NIRI longslit spectroscopy.

    Note: NIRI uses lamp-on flats for all bands other than M (J, H, K, L),
    which instead uses lamp-off flats. No dark correction is performed.

    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    p.stackFlats()
    p.determineSlitEdges()
    p.maskBeyondSlit()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.makeIRAFCompatible()
    p.storeProcessedFlat()

_default = makeProcessedFlat
