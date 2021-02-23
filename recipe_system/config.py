#
#                                                                        DRAGONS
#
#                                                                  recipe_system
#                                                                      config.py
# ------------------------------------------------------------------------------
# CONFIG SERVICE

"""
This module provides an interface to config files, and a globally available
config object, to share setup information across the application.

An instance of ConfigParser, `globalConf`, is initialized when first loading
this module, and it should be used as the only interface to the config system.

"""
import os
from configparser import ConfigParser

DEFAULT_DIRECTORY = '~/.geminidr'

try:
    STANDARD_REDUCTION_CONF = os.environ["DRAGONSRC"]
except KeyError:
    STANDARD_REDUCTION_CONF = '~/.geminidr/rsys.cfg'


def environment_variable_name(section, option):
    return '_GEM_{}_{}'.format(section.upper(), option.upper())


def load_config(filenames=None):
    """
    Updates the globalConf object by reading one or more config files.
    If no filename is passed, the default config file is read.

    Parameters
    ----------
    filenames : str/list/None
        filenames of config files to load
    """
    if filenames is None:
        filenames = (STANDARD_REDUCTION_CONF,)
    elif isinstance(filenames, str):
        filenames = (filenames,)

    for filename in filenames:
        globalConf.read(os.path.expanduser(filename))


globalConf = ConfigParser()
