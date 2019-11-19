#
#                                                                        DRAGONS
#
#                                                          mappers.baseMapper.py
# ------------------------------------------------------------------------------
from builtins import object

from ..utils.mapper_utils import dotpath

# ------------------------------------------------------------------------------


class Mapper(object):
    """
    This is the base class for classes

    :class:`recipeMapper.RecipeMapper`

    and

    :class:`primitiveMapper.PrimitiveMapper`.

    It provides initialization only.

    Recipes and primitives are algorithmically selected via instropection of
    module and class attributes that match on a dataset's tags attribute.

    Parameters
    ----------

    dtags : <set>
            A set of AstroData tags from input dataset. These are decoupled
            from astrodata objects so as not to introduce 'ad' objects into
            mapper generators.

    ipkg  : <str>
            Instrument package name, lower case, as returned by,

                ad.instrument(generic=True).lower()

    drpkg : <str>
            The data reduction package to map. Default is 'geminidr'.
            This package *must* be importable.

    recipename : <str>
                 The recipe to use for processing. Passed by user
                 with -r or set by caller. Else 'default' recipe.

    mode : <str>
           Pipeline mode. Selection criterion for recipe sets.
           Supported modes:
           'sq' - Science Quality (default)
           'qa' - Quality Assessment
           'ql' - Quicklook

    """

    def __init__(self, dtags, ipkg, mode='sq', drpkg='geminidr', recipename='_default'):
        self.tags = dtags
        self.mode = mode
        self.dotpackage = dotpath(drpkg, ipkg)
        self.recipename = recipename
