#
#                                                          mappers.baseMapper.py
# ------------------------------------------------------------------------------
import imp
import sys

from ..utils.mapper_utils import dictify

# ------------------------------------------------------------------------------
GMOS_INSTR    = ['GMOS-S', 'GMOS-N']
# ------------------------------------------------------------------------------
class Mapper(object):
    """
    This is the base class for RecipeMapper and PrimitiveMapper classes and 
    provide initialization only. 

    Recipes and primitives are algorithmically selected based on an dataset's 
    tags. 
    (See utils.mapper_utils.retrieve_primitive_set())

    Retrieve the appropriate primitive class for a dataset, using all
    defined defaults:

    >>> ad = astrodata.open(<fitsfile>)
    >>> adinputs = [ad]
    >>> rm = RecipeMapper(adinputs)
    >>> recipe = rm.get_applicable_recipe()
    >>> recipe.__name__ 
    'qaReduce'

    """
    def __init__(self, adinputs, recipename='default', context='QA', 
                 usercals=None, uparms=None):
        """
        :parameter ads: list of AstroData objects.
        :type ads: <list>

        :parameter recipename: The recipe to use for processing. Passed by
                               user with -r or set by caller. 
                               If None, 'default' recipe.
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

        """
        self.adinputs = adinputs
        self.adinit = adinputs[0]
        self.context = context
        self.tags = set(self.adinit.type()) # change to new ad, ad.tags
        self.pkg = self._set_pkg()          # change to new ad, ad.instrument_name
        self.recipename = recipename
        self.usercals = usercals if usercals else {}
        self.userparams = dictify(uparms)


    def _set_pkg(self):
        raw_inst = self.adinit.instrument().as_pytype()
        return 'GMOS' if raw_inst in GMOS_INSTR else raw_inst

    def _package_loader(self, pkgname):
        pfile, pkgpath, descr = imp.find_module(pkgname)
        loaded_pkg = imp.load_module(pkgname, pfile, pkgpath, descr)
        sys.path.extend(loaded_pkg.__path__)
        return loaded_pkg
