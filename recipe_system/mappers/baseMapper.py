#
#                                                          mappers.baseMapper.py
# ------------------------------------------------------------------------------
from builtins import object
import sys
import importlib

from types import StringType

from ..utils.mapper_utils import dictify
from ..utils.mapper_utils import dotpath
# ------------------------------------------------------------------------------
class Mapper(object):
    """
    This is the base class for classes 

    RecipeMapper
    PrimitiveMapper

    It provides initialization only.

    Recipes and primitives are algorithmically selected via instropection of
    module and class attributes that match on a dataset's tags attribute.

    """
    def __init__(self, adinputs, context=['qa'], drpkg='geminidr', recipename='default',
                 usercals=None, uparms=None, upload_metrics=False):
        """
        :parameter adinputs: list of AstroData objects.
        :type adinputs: <list>

        :parameter drpkg: The data reduction package to map. Default is 'geminidr'.
                          This package *must* be importable.
        :type drpkg: <str>

        :parameter recipename: The recipe to use for processing. Passed by user 
                               with -r or set by caller. Else 'default' recipe.
        :type recipename: <str>

        :parameter context: The context. This defines which recipe set to use,
                            Default is 'QA'.
        :type context: <str>

        :parameter usercals: A dict of user provided calibration files, keyed
                             on cal type.

                             E.g.,
                                  {'processed_bias': 'foo_bias.fits'}
                             
        :type usercals: <dict>

        :parameter uparms: A set of user parameters passed via command line
                           or other caller.
        :type uparms: <list> list of (parameter, value) tuples. Each may have a 
                             specified primitive.
                             E.g., [('foo','bar'), ('tileArrays:par1','val1')]

        :parameter upload_metrics: Send Qa metrics to fitsstore.
        :type upload_metrics: <bool>

        """
        self.adinputs   = adinputs
        self._context   = context
        ainst = adinputs[0].instrument()
        self.pkg        = 'gmos' if "GMOS" in ainst else ainst.lower()
        self.dotpackage = dotpath(drpkg, self.pkg)
        self.recipename = recipename
        self.tags       = adinputs[0].tags
        self.usercals   = usercals if usercals else {}
        self.userparams = dictify(uparms)
        self.upload_metrics = upload_metrics


    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, ctx):
        if ctx is None:
            self._context = ['qa']         # Set default 'qa' [later, 'sq']
        elif isinstance(ctx, StringType):
            self._context = [seg.lower().strip() for seg in ctx.split(',')]
        elif isinstance(ctx, list):
            self._context = ctx
        return
