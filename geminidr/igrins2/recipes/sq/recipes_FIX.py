"""
"""
from igrinsdr.igrins.primitives_igrins import Igrins

recipe_tags = {'IGRINS'}

def fixHeader(p: Igrins):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.fixHeader()
    p.writeOutputs()
    return

_default = fixHeader

