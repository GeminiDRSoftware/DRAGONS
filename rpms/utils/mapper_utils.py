"""
Utility functions for RecipeMapper.

includes,

    configure_pkg()   -- returns a PackageConfig object.
    get_config_file() -- looks for a $GEMINIDR env var
    dictify()         -- list of user parameters -> dictionary
    dotpath()         -- build a python import path or GeminiDR packages.

"""
import os

from packageConfig import PackageConfig

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

    :parameter args: a list of arguments of arbitrary length
    :type args: <list>, implied by *

    :returns: a path to an importable module
    :rtype: <str>

    """
    ppath = ''
    for pkg in args:
        if ppath:
            ppath += '.'+pkg
        else:
            ppath += pkg
    ppath.rstrip('.')
    return ppath
