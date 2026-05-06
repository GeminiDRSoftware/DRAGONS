from geminidr.igrins2.primitives_igrins import IGRINS

recipe_tags = {'IGRINS-2', 'SIDEREAL'}

from .recipe_STD import makeStellar

def makeTgt(p: IGRINS):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    makeStellar(p)
    # normalize the spectra

_default = makeTgt
