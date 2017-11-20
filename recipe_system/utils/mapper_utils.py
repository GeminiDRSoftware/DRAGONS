#
#                                                          utils.mapper_utils.py
# ------------------------------------------------------------------------------
"""
Utility functions for Mappers.

    find_user_recipe  -- searches for a user specified recipe, if any.
    dictify()         -- list of user parameters -> dictionary
    dotpath()         -- build a python import path for dr packages.

"""
import os
import sys

from importlib import import_module
# ------------------------------------------------------------------------------
RECIPEMARKER = 'recipes'
# ------------------------------------------------------------------------------
def find_user_recipe(dashr):
    """
    Function receives the value of the reduce [-r, --recipe] flag, if passed.
    This will be a path to a recipe file and a dotted recipe name, which
    exists as a function in the recipe file. A properly specified user
    recipe shall contain one, and only one, dot operator.

    If the recipefile.recipename cannot be found on the path, whether
    specified or implied (as cwd), then None is returned.

    The string value of dashr will look like,

    -r '/path/to/users/recipes/recipefile.recipe_function'

    -r 'recipefile.recipe_function' -- recipe file in cwd.

    A recipe name with no dot operator implies a recipe name in the system
    recipe library.

    :parameter dashr: a path to a recipe file dotted with a recipe function name.
    :type dashr: <str>

    :returns: imported recipe function OR None
    :rtype:   <type 'function'> or  None

    """
    rpath = os.path.abspath(os.path.expanduser(dashr))
    addsyspath, recipe = os.path.split(rpath)
    try:
        modname, rname = recipe.split('.')
    except ValueError:
        return None

    sys.path.append(addsyspath)
    rmod = import_module(modname)
    try:
        recipefn = getattr(rmod, rname)
    except AttributeError:
        recipefn = None

    return recipefn

def dictify(parset):
    """
    Converts a list of tuples, nominally a set of user parameters passed via
    the 'reduce' command line, into a dictionary directly usable by a primitive
    class.

    :parameter parset: A list of user parameters tuples,
                       e.g., [('foo', 'bar'), ('foobar', 'bat')]
    :type parset: <list>

    :returns: A dictionary of those tuples as key=val pairs,
              e.g., {'foo': 'bar', 'foobar': 'bat'}
    :rtype: <dict>

    """
    pardict = {}
    if parset:
        for pset in parset:
            pardict[pset[0]] = pset[1]
    return pardict

def dotpath(*args):
    """
    Build an import path from args.

    :parameter args: a set of arguments of arbitrary length
    :type args:      <list>, implied by *

    :returns: a dot path to an importable module
    :rtype: <str>

    """
    return os.extsep.join(args)
