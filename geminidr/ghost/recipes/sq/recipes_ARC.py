"""
Recipes available to data with tags ``['GHOST', 'CAL', 'ARC']``.
Default is ``makeProcessedArc``, which is imported from the QA module.
"""
recipe_tags = set(['GHOST', 'CAL', 'ARC'])


def makeProcessedArc(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input arc images into a single stacked arc image. This output
    processed arc is stored on disk using storeProcessedArc and has a name
    equal to the name of the first input arc image with "_arc.fits" appended.
    The wavelength solution is also stored.

    Parameters
    ----------
    p : Primitives object
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
    p.tileArrays()
    p.stackArcs()
    p.extractProfile(flat_correct=False, write_result=True)
    p.fitWavelength()
    p.storeProcessedArc()
    return


_default = makeProcessedArc
