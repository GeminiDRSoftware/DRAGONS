from geminidr.igrins.primitives_igrins import IGRINS2

recipe_tags = {'IGRINS-2', 'SIDEREAL'}

from .recipe_STD import makeStellar

def makeTgt(p: IGRINS2):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    makeStellar(p)
    # normalize the spectra

_default = makeTgt
