import os
from importlib import import_module

from astrodata.utils.Errors import RecipeNotFoundError

# ------------------------------------------------------------------------------
GMOS_INSTR    = ['GMOS-S', 'GMOS-N']
canonicals    = ['IMAGE', 'SPECT', 'NODANDSHUFFLE']
recipedir     = 'recipes'
primitives_in = "primitives"
mod_prefix    = "primitives_"
class_prefix  = "Primitives"

# ------------------------------------------------------------------------------
def dotpath(*args):
    """
    Build an import path from args.

    """
    ppath = ''
    for pkg in args:
        if ppath:
            ppath += '.'+pkg
        else:
            ppath += pkg
    ppath.rstrip('.')
    return ppath

# ------------------------------------------------------------------------------
class RecipeMapper(object):
    """
    build importable paths to a primitive set and a recipe.
    Import them, run.

    Some pseudo code because not sure what the final AstroData types
    will be. Let's go with 

    GMOS --> AstroDataGMOS 
    etc.

    """
    def __init__(self, ad, recipename='default', context='QA', uparms=None):
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

        :parameter uparms: A set of user parameters passed via command line
                           or other caller.
        :type uparms: <list> list of (parameter, value) tuples. Each may have a 
                             specified primitive.
                             E.g., [('foo','bar'), ('tileArrays:par1','val1')]

        """
        self.adinput = ad
        self.tags = ad.type()
        self.recipename = recipename
        self.context = context
        self.canonical = None
        self.pkg = None
        self.recipelib = None
        self.userparams = uparms 
        self._set_pkg()
        self._set_canonical()

    def set_recipe_library(self):
        """
        Sets the recipelib attribute for the canonical dataset type, like IMAGE,
        SPECT, etc. . In this prototype, the recipelib is an actual function
        library comprising the defined recipe functions.

        """
        self.recipelib = import_module(dotpath(self.pkg, recipedir, self.canonical))
        return

    def get_recipe_actual(self):
        try:
            recipe = getattr(self.recipelib, self.recipename)
        except AttributeError:
            emsg = "Recipe {} not found.".format(self.recipename)
            raise RecipeNotFoundError(emsg)

        return recipe

    def get_applicable_primitives(self):
        path = self._set_primitive_path()
        primitive_mod = import_module(path)
        primitiveclass = class_prefix + self.canonical
        primitive_actual = getattr(primitive_mod, primitiveclass)
        return primitive_actual([self.adinput])

    def _set_primitive_path(self):
        primitive_mod = mod_prefix + self.canonical
        ppath = dotpath(self.pkg, primitives_in, primitive_mod)
        return ppath

    def _set_pkg(self):
        """
        Determines the instrument package. Right now, this just uses
        the instrument descriptor on the ad.

        """
        if self.adinput.instrument().as_pytype() in GMOS_INSTR:
            self.pkg = "GMOS"
        else:
            self.pkg = self.adinput.instrument().as_pytype()
        return

    def _set_canonical(self):
        if "IMAGE" in self.tags:
            self.canonical = "IMAGE"
        elif "SPECT" in self.tags:
            self.canonical = "SPECT"
        # elif ...
        return
