from igrinsdr.igrins.primitives_igrins import Igrins

recipe_tags = {'IGRINS', 'SCIENCE'}

from .recipe_STD import makeStellar

def makeTgt(p: Igrins):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    makeStellar(p)
    # normalize the spectra

_default = makeTgt
