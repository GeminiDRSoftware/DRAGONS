"""
Recipes available to data with tags ['GMOS', 'SPECT', 'NODANDSHUFFLE'].
Currently (2018-11-28), NODANDSHUFFLE datasets are slated as SPECT
type data. This may be revised in future with greater capabilities developed
under DRAGONS.

The default recipe is "reduce".

"""
recipe_tags = set(['GMOS', 'SPECT', 'NODANDSHUFFLE'])


def reduce(p):
    """
    This recipe does a quick reduction of GMOS nod and shuffle data.
    The data is left in its 2D form, and only a sky correction is done.
    The seeing from the spectra cross-section is measured when possible.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.findAcquisitionSlits()
    p.skyCorrectNodAndShuffle()
    p.measureIQ(display=True)
    p.writeOutputs()
    return


_default = reduce
