from importlib import import_module

from astrodata.utils.Errors import RecipeNotFoundError

from utils.mapper_utils  import dictify
from utils.mapper_utils  import dotpath
from utils.mapper_utils  import configure_pkg

# ------------------------------------------------------------------------------
GMOS_INSTR    = ['GMOS-S', 'GMOS-N']
canonicals    = ['IMAGE', 'SPECT', 'NODANDSHUFFLE']
# ------------------------------------------------------------------------------
class RecipeMapper(object):
    """
    Build importable paths to a primitive set and a recipe.
    Import them, run.

    Some pseudo code because not sure what the final AstroData types
    will be. Let's go with 

    GMOS --> AstroDataGMOS 
    etc.

    """
    def __init__(self, adinputs, recipename='default', context='QA', uparms=None):
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
        self.adinputs = adinputs
        self.adinit = adinputs[0]
        self.tags = self.adinit.type()
        self.recipename = recipename
        self.context = context
        self.canonical = None
        self.pkg = None
        self.pkg_conf = configure_pkg()
        self.recipelib = None
        self.userparams = dictify(uparms)

    def get_applicable_primitives(self):
        path = self._set_primitive_path()
        primitive_mod = import_module(path)
        primitiveclass = self.pkg_conf.class_prefix + self.canonical
        primitive_actual = getattr(primitive_mod, primitiveclass)
        return primitive_actual(self.adinputs, uparms=self.userparams)

    def get_applicable_recipe(self):
        try:
            recipe = getattr(self.recipelib, self.recipename)
        except AttributeError:
            emsg = "Recipe {} not found.".format(self.recipename)
            raise RecipeNotFoundError(emsg)

        return recipe

    def set_recipe_library(self):
        """
        Calls to set the package and canonical dataset type. 

        Sets the recipelib attribute for the canonical dataset type, 
        such as IMAGE, SPECT, etc.. In this prototype, the recipelib is an 
        actual function library comprising the defined recipe functions.

        """
        self._set_pkg()
        self._set_canonical()
        recipedir = self.pkg_conf.recipe_path
        self.recipelib = import_module(dotpath(self.pkg, recipedir, 
                                               self.context, self.canonical))
        return

# ------------------------------- prive ----------------------------------------
    def _set_primitive_path(self):
        primitive_mod = self.pkg_conf.primitive_prefix + self.canonical
        ppath = dotpath(self.pkg, self.pkg_conf.primitive_path, primitive_mod)
        return ppath

    def _set_pkg(self):
        """
        Determines the instrument package. Right now, this just uses
        the instrument descriptor on the ad.

        """
        if self.adinit.instrument().as_pytype() in GMOS_INSTR:
            self.pkg = "GMOS"
        else:
            self.pkg = self.adinit.instrument().as_pytype()
        return

    def _set_canonical(self):
        if "IMAGE" in self.tags:
            self.canonical = "IMAGE"
        elif "SPECT" in self.tags:
            self.canonical = "SPECT"
        # elif ...
        return
