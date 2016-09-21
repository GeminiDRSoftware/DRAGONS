
from utils.errors import RecipeNotFound
from utils.mapper_utils import dictify
from utils.mapper_utils import find_user_recipe
from utils.mapper_utils import retrieve_primitive_set
from utils.mapper_utils import retrieve_recipe

# ------------------------------------------------------------------------------
GMOS_INSTR    = ['GMOS-S', 'GMOS-N']
# ------------------------------------------------------------------------------
class RecipeMapper(object):
    """
    Recipes and primitives are algorithmically selected based on an dataset's 
    tags. 
    (See utils.mapper_utils.retrieve_primitive_set())

    Retrieve the appropriate primitive class for a dataset, using all
    defined defaults:

    >>> ad = astrodata.open(<fitsfile>)
    >>> adinputs = [ad]
    >>> rm = RecipeMapper(adinputs)
    >>> primitive_set = rm.get_applicable_primitives()
    >>> primitive_set.__module__
    'primitives_IMAGE'
    >>> primitive_set.__class__
    <class 'primitives_IMAGE.PrimitivesIMAGE'>

    

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
        self.pkg = self._set_pkg()
        self.recipename = recipename
        self.userparams = dictify(uparms)

    def get_applicable_primitives(self):
        tag_match, primitive_actual = retrieve_primitive_set(self.tags, self.pkg)
        return primitive_actual(self.adinputs, uparms=self.userparams)

    def get_applicable_recipe(self):
        recipefn = find_user_recipe(self.recipename)
        if recipefn is None:
            tag_match, recipefn = retrieve_recipe(self.tags, self.pkg, 
                                                     self.recipename, self.context)

        if recipefn is None:
            raise RecipeNotFound("Recipe '{}' not found.".format(self.recipename))

        return recipefn

    def _set_pkg(self):
        raw_inst = self.adinit.instrument().as_pytype()
        return 'GMOS' if raw_inst in GMOS_INSTR else raw_inst
