#
#                                                                        DRAGONS
#
#                                                          mappers.baseMapper.py
# ------------------------------------------------------------------------------
from builtins import object

from ..utils.mapper_utils import dictify
from ..utils.mapper_utils import dotpath

# ------------------------------------------------------------------------------
class Mapper(object):
    """
    This is the base class for classes
    :class:`~recipe_system.mappers.recipeMapper.RecipeMapper`
    and
    :class:`~recipe_system.mappers.primitiveMapper.PrimitiveMapper`.

    It provides initialization only.

    Recipes and primitives are algorithmically selected via instropection of
    module and class attributes that match on a dataset's tags attribute.

    Parameters
    ----------

    adinputs : <list>
               A list of AstroData objects.

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

    usercals : <dict>
               A dict of user provided calibration files, keyed on cal type.
               E.g., {'processed_bias': 'foo_bias.fits'}

    uparms : <list>
             A list of tuples of user parameters like, (parameter, value).
             Each may have a specified primitive.
             E.g., [('foo','bar'), ('tileArrays:par1','val1')]

    upload : <list> A list of things to upload. e.g., ['metrics']


    """
    def __init__(self, adinputs, mode='sq', drpkg='geminidr', recipename='_default',
                 usercals=None, uparms=None, upload=None):

        self.adinputs = adinputs
        self.mode = mode
        self.pkg = adinputs[0].instrument(generic=True).lower()
        self.dotpackage = dotpath(drpkg, self.pkg)
        self.recipename = recipename
        self.tags = adinputs[0].tags
        self.usercals = usercals if usercals else {}
        self.userparams = dictify(uparms)
        self._upload = upload

    @property
    def upload(self):
        return self._upload

    @upload.setter
    def upload(self, upl):
        if upl is None:
            self._upload = None
        elif isinstance(upl, str):
            self._upload = [seg.lower().strip() for seg in upl.split(',')]
        elif isinstance(upl, list):
            self._upload = upl
        else:
            raise TypeError("'upload' must be one of None, <str>, or <list>")
        return
