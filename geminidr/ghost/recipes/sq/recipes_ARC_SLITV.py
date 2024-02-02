"""
Recipes available to data with tags ``['GHOST', 'CAL', 'SLITV', 'ARC']``.
Default is ``makeProcessedSlitArc``, which is an alias to
:any:`makeProcessedSlit`.
"""
recipe_tags = set(['GHOST', 'CAL', 'SLITV', 'ARC'])

def makeProcessedSlitArc(p):
    """
    This recipe processes GHOST science data.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.fixCosmicRays()
    p.weightSlitExposures()
    p.stackFrames()
    p.storeProcessedSlit()
    return

_default = makeProcessedSlitArc
