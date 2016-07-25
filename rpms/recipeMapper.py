from importlib import import_module

from utils.errors import RecipeNotFound
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
        self.context = context
        self.tags = set(self.adinit.type())
        self.pkg = self.adinit.instrument().as_pytype()
        self.pkg_conf = configure_pkg()
        self.recipelib = None
        self.recipename = recipename
        self.userparams = dictify(uparms)

    def get_applicable_primitives(self):
        matching_tags, primitive_actual = retrieve_primitive_set(self.tags, self.pkg)
        return primitive_actual(self.adinputs, uparms=self.userparams)

    def get_applicable_recipe(self):
        try:
            recipe = getattr(self.recipelib, self.recipename)
        except AttributeError:
            emsg = "Recipe {} not found.".format(self.recipename)
            raise RecipeNotFoundError(emsg)

        return recipe

