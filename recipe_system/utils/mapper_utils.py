#
#                                                          utils.mapper_utils.py
# ------------------------------------------------------------------------------
"""
Utility functions for Mappers.

    find_user_recipe  -- searches for a user specified recipe, if any.
    configure_pkg()   -- returns a PackageConfig object.
    get_config_file() -- looks for a $GEMINIDR env var
    dictify()         -- list of user parameters -> dictionary
    dotpath()         -- build a python import path or GeminiDR packages.

    The following functional equivalents of same named methods defined
    on the RecipeMapper and PrimitiveMapper classes.
    ==================================================================
    retrieve_recipe        -- stand alone recipe finder.
    retrieve_primitive_set -- stand alone primitive finder.

"""
import os
import imp
import sys
import pkgutil

from inspect import isclass
from importlib import import_module

from packageConfig import PackageConfig

# ------------------------------------------------------------------------------
def find_user_recipe(dashr):
    """
    Function recieves the value of the reduce [-r, --recipe] flag, if passed.
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

def configure_pkg():
    cfgfile = get_config_file()
    pkg_conf = PackageConfig()
    pkg_conf.configure_pkg(cfgfile)
    return pkg_conf

def get_config_file():
    """
    Find a GeminiDR package config file, pkg.cfg.
    Examines env var $GEMINIDR.

    """
    config_path = os.environ.get('GEMINIDR')
    default_cfg = 'pkg.cfg'
    config_file = os.path.join(config_path, default_cfg)
    return config_file

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



# ------------------------------------------------------------------------------
# public functions
#
#     -- retrieve_primtive_set()
#     -- retrieve_recipe()
#
# and supporting generators to introsepctively find primitive classes and recipe
# libraries.
# ------------------------------------------------------------------------------
RECIPEMARKER = 'recipes'
DRMARKER = 'geminidr'
# ------------------------------------------------------------------------------
def _package_loader(pkgname):
    pfile, pkgpath, descr = imp.find_module(pkgname)
    loaded_pkg = imp.load_module(pkgname, pfile, pkgpath, descr)
    sys.path.extend(loaded_pkg.__path__)
    return loaded_pkg

# ------------------------------------------------------------------------------
# Recipe search cascade

def _generate_context_libs(pkg):
    pkg_importer = pkgutil.ImpImporter(pkg)
    for pkgname, ispkg in pkg_importer.iter_modules():
        if not ispkg:
            yield pkgname, ispkg
        else:
            continue

def _generate_context_pkg(pkg, context):
    pkg_importer = pkgutil.ImpImporter(pkg)
    for pkgname, ispkg in pkg_importer.iter_modules():
        if ispkg and pkgname == context:
            break
        else:
            continue

    loaded_pkg = _package_loader(pkgname)
    for mod, ispkg in _generate_context_libs(loaded_pkg.__path__[0]):
        yield mod, ispkg

def _generate_recipe_modules(pkg, context, recipedir=RECIPEMARKER):
    pkg_importer = pkgutil.ImpImporter(pkg)
    for pkgname, ispkg in pkg_importer.iter_modules():
        if ispkg and pkgname == recipedir:
            break 
        else:
            continue

    loaded_pkg = _package_loader(pkgname)
    for context_pkg, ispkg in _generate_context_pkg(loaded_pkg.__path__[0],context):
        yield context_pkg, ispkg

def _get_tagged_recipes(pkgname, context):
    loaded_pkg = _package_loader(pkgname)
    for rmod, ispkg in _generate_recipe_modules(loaded_pkg.__path__[0], context):
        if not ispkg:
            yield import_module(rmod)
        else:
            continue

def retrieve_recipe(adtags, pkgname, rname, context):
    """
    Caller passes a recipe name, set of AstroData tags, the instrument package 
    name and a "context". Currently, this is defined as either "QA" or "SQ".

    :parameter rname:  name of requested recipe.
    :type rname:       <str>

    :parameter adtags: set of AstroData tags on an 'ad' instance.
    :type adtags:      <type 'set'>
                       E.g., set(['GMOS', 'SIDEREAL', 'SPECT', 'GMOS_S', 'GEMINI'])

    :parameter pkgname: An instrument package under GeminiDR.
    :type pkgname:     <str>, E.g., "GMOS"

    :parameter context: the context for recipe selection. 
    :type context:      <str> 

    :returns: tuple including the best tag set match and the primitive class
              that provided the match.
    :rtype: <tuple>, (set, class)

    """
    matched_set = (set([]), None)
    for rlib in _get_tagged_recipes(pkgname, context):
        if hasattr(rlib, 'recipe_tags'):
            if adtags.issuperset(rlib.recipe_tags):
                isect = rlib.recipe_tags
                matched_set = (isect, rlib) if isect > matched_set[0] else matched_set
            else:
                continue
        else:
            continue

    isection, rlib = matched_set
    try:
        recipe_actual = getattr(rlib, rname)
    except AttributeError:
        recipe_actual = None
    return isection, recipe_actual


# ------------------------------------------------------------------------------
# Primtive hunt cascade

def _generate_primitive_modules(pkg):
    pkg_importer = pkgutil.ImpImporter(pkg)
    for pkgname, ispkg in pkg_importer.iter_modules():
        if ispkg:
            continue
        else:
            yield (pkg_importer.path, pkgname)

def _get_tagged_primitives(pkgname):
    loaded_pkg = _package_loader(pkgname)
    for pkgpath, pkg in _generate_primitive_modules(loaded_pkg.__path__[0]):
        fd, path, descr = imp.find_module(pkg, [pkgpath])
        mod = imp.load_module(pkg, fd, path, descr)
        for atrname in dir(mod):
            if atrname.startswith('_'):        # no prive, no magic
                continue
                
            atr = getattr(mod, atrname)
            if isclass(atr) and hasattr(atr, 'tagset'):
                yield atr

def retrieve_primitive_set(adtags, pkgname):
    """
    Caller passes a set of AstroData tags and the instrument package name.

    :parameter adtags: set of AstroData tags on an 'ad' instance.
    :type adtags:      <type 'set'>
                       E.g., set(['GMOS', 'SIDEREAL', 'SPECT', 'SOUTH', 'GEMINI'])

    :parameter pkgname: An instrument package under GeminiDR.
    :type pkgname:     <str>, E.g., "GMOS"

    :returns: tuple including the best tag set match and the primitive class
              that provided the match.
    :rtype: <tuple>, (set, class)

    """
    matched_set = (set([]), None)
    for pclass in _get_tagged_primitives(pkgname):
        isection = adtags.intersection(pclass.tagset)
        matched_set = (isection, pclass) if isection > matched_set[0] else matched_set
    return matched_set
