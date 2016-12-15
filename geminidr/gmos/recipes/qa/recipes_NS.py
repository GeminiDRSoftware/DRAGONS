"""
Recipes available to data with tags ['GMOS', 'NODANDSHUFFLE']
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'NODANDSHUFFLE'])

def reduce(p):
    """
    This recipe does a quick reduction of GMOS nod and shuffle data.
    The data is left in its 2D form, and only a sky correction is done.
    The seeing from the spectra cross-section is measured when possible.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.findAcquisitionSlits()
    p.skyCorrectNodShuffle()
    p.measureIQ(display=True)
    return

default = reduce
