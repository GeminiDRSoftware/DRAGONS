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
import os, warnings
from os.path import exists, expanduser
from configparser import ConfigParser

DEFAULT_DIRECTORY = '~/.dragons'

try:
    STANDARD_REDUCTION_CONF = os.environ["DRAGONSRC"]
except KeyError:
    STANDARD_REDUCTION_CONF = os.path.join(DEFAULT_DIRECTORY, 'dragonsrc')

OLD_REDUCTION_CONF = '~/.geminidr/rsys.cfg'


def environment_variable_name(section, option):
    return '_GEM_{}_{}'.format(section.upper(), option.upper())


def expand_filenames(filenames=None, deprecation_warning=True):
    """
    Expand the list of filenames to read the config from.

    Parameters
    ----------
    filenames : str/list/None
        filenames of config files to load, or None for default
    """
    if filenames is None:
        if exists(expanduser(STANDARD_REDUCTION_CONF)):
            return STANDARD_REDUCTION_CONF,
        elif exists(expanduser(OLD_REDUCTION_CONF)):
            if deprecation_warning:
                with warnings.catch_warnings():
                    warnings.simplefilter("always", DeprecationWarning)
                    warnings.warn("The ~/.geminidr/rsys.cfg file is deprecated. "
                                  "Please create a ~/.dragons/dragonsrc config file.",
                                  DeprecationWarning
                                  )
            return OLD_REDUCTION_CONF,
        else:
            return []
    elif isinstance(filenames, str):
        return filenames,
    elif filenames is None:
        return set()
    else:
        return filenames


def load_config(filenames=None):
    """
    Updates the globalConf object by reading one or more config files.
    If no filename is passed, the default config file is read.

    Parameters
    ----------
    filenames : str/list/None
        filenames of config files to load
    """
    filenames = expand_filenames(filenames)
    globalConf.read([expanduser(filename) for filename in filenames])


globalConf = ConfigParser()
