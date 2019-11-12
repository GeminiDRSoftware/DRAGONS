#
#                                                                        DRAGONS
#
#                                                                  recipe_system
#                                                                      config.py
# ------------------------------------------------------------------------------
#from future import standard_library
#standard_library.install_aliases()
#from builtins import map
#from builtins import str
#gitfrom builtins import object


# ------------------------------------------------------------------------------
# CONFIG SERVICE

"""
This module provides an interface to config files, and a globally available
config object, to share setup information across the application.

An instance of `ConfigObject`, `globalConf`, is initialized when first loading
this module, and it should be used as the only interface to the config system.

"""
import os
import types
from configparser import SafeConfigParser
from collections import defaultdict

DEFAULT_DIRECTORY = '~/.geminidr'
STANDARD_REDUCTION_CONF = '~/.geminidr/rsys.cfg'


class ConfigObject(object):
    """
    Generalized wrapper on SafeConfigParser.

    This class is used for reading, parsing, and updating configurations for
    DRAGONS calibration service.

    Attributes
    ----------
    _sections : <dict>

    _conv : <dict>
        A "converter" dictionary providing a key pointing to a native python
        data type. E.g., {(section, option): bool}. In the case of the DRAGONS
        calibration service, the boolean 'standalone' configuration option, this
        dictionary looks like,

        {('calibs', 'standalone'): bool}

    _exports : <dict>

    """

    def __init__(self):
        self._sections = {}
        self._conv = {}
        self._exports = defaultdict(set)

    def __getitem__(self, item):
        try:
            return self._sections[item.lower()]
        except KeyError:
            raise KeyError("There is no {0!r} section".format(item))

    def export_section(self, section):
        """
        Some options from the specified section may be published as
        environment variables, where spawned processes can find them.

        The exported variables would be the ones speficied using
        `update_exports`.

        Parameters
        ----------
        section : <str>
            Section name, e.g., 'calibs'

        """
        try:
            sect = self._sections[section]
        except KeyError:
            # Nothing to export...
            return

        for option in self._exports.get(section, ()):
            try:
                env = environment_variable_name(section, option)
                os.environ[env] = str(getattr(sect, option))
            except AttributeError:
                # The option was not defined...
                pass

    def load(self, filenames, defaults=None, env_override=False):
        """
        Loads all or some entries from the specified section in a config file.
        The extracted values are set as environment variables, so that they are
        available at a later point to other modules or spawned processes.

        Parameters
        ----------
        filenames : <str> or <iterable>
            A string or a sequence of strings containing the path(s) to
            configuration file(s). If a value is present in more than one file,
            the latest one to be processed overrides the preceding ones.

            Paths can start with `~/`, meaning the user home directory.

        defaults : <dict>, optional
            If some options are not found, and you want to set up a default
            value, specify them in here. Every key in the dictionary is the
            name of a section in the config file, and each element is another
            dictionary establishing attribute-value pairs for that section.

        env_override : <bool>, optional (default is False)
            If true, after loading values from the configuration files, the
            environment will be explored in search of options passed down by
            the parent process. Those options will override the ones taken
            from the config files.

        """
        try:
            basestring
        except NameError:
            basestring = str

        if isinstance(filenames, basestring):
            filenames = (filenames,)

        # Set the default values
        if defaults is not None:
            for section, sub_items in list(defaults.items()):
                current_section_conf = self._sections.get(section, Section({})).as_dict()
                for key, value in list(sub_items.items()):
                    if key not in current_section_conf:
                        current_section_conf[key] = value
                self._sections[section] = Section(current_section_conf)

        cp = SafeConfigParser()
        cp.read(list(map(os.path.expanduser, filenames)))
        translate = Converter(self._conv.copy(), cp)

        # Coerce values and apply overrides
        for section in cp.sections():
            values = {}
            for key in cp.options(section):
                try:
                    values[key] = os.path.expanduser(
                        translate.from_config_file(section, key))
                except TypeError:
                    values[key] = translate.from_config_file(section, key)
                except AttributeError:
                    values[key] = translate.from_config_file(section, key)

            if env_override:
                for key in values:
                    env = environment_variable_name(section, key)
                    if env in os.environ:
                        values[key] = translate.from_raw(section, key, os.environ[env])

            self.update(section, values)

    def update(self, section, values):
        """
        Regenerates a section from scratch. If the section had been loaded
        before, it will take the previous values as a basis and update them
        with the new ones.

        Parameters
        ----------
        section: <str>
            The configuration section to update.

        values: <dict>
            The values associated to the section.

        """
        prev = self._sections[section].as_dict() if section in self._sections else {}
        prev.update(values)
        self._sections[section] = Section(prev)

    def update_exports(self, expdict):
        """
        Updates the internal export table that will be used to share config
        information with process spawns.

        Parameters
        ----------
        expdict : <dict>
            Each key is the name of a section. The values of the dictionary are
            sequences of strings, with each string in the sequence being the
            name of a config entry in that section that will be exported, if
            found.

        """
        for section, opts in list(expdict.items()):
            self._exports[section].update(opts)

    def update_translation(self, conv):
        """
        Updates the internal mapping table for automatic translation of data
        types when reading from config files.

        Parameters
        ----------
        conv : <dict>
            A mapping `(section_name, item)` -> Python type. Used internally for
            type translation when reading values from the config file. If a
            section/item pair is missing then a fallback `(None, item)` will be
            tried. If no match is found, no translation will be performed.

            The only types to be considered are: `int`, `float`, `bool`

        """
        self._conv.update(conv)


class Converter(object):
    def __init__(self, conv_dict, cp):
        """
        The class provides an internal mapping table for automatic translation
        to native data types when reading from config files.

        Parameters
        ----------
        conv : <dict>
            A mapping `(section_name, item)` -> Python type. Used internally for
            type translation when reading values from the config file. If a
            section/item pair is missing then a fallback `(None, item)` will be
            tried. If no match is found, no translation will be performed.

            Mapped types:  `int`, `float`, `bool`

            E.g., {('calibs', 'standalone'): bool}

        cp: <obj>
            An instance of SafeConfigParser.

        """
        self._trans = dict(conv_dict)
        self._cp_default = cp.get
        self._type_to_cp = { bool: cp.getboolean,
                             float: cp.getfloat,
                             int: cp.getint
        }

    def from_config_file(self, section, key):
        try:
            return self._type_to_cp[self._trans[(section, key)]](section, key)
        except KeyError:
            return self._type_to_cp.get((None, key), self._cp_default)(section, key)

    def from_raw(self, section, key, value):
        return self._trans.get(key, str)(value)


class Section(object):
    """
    An instance of `Section` describes the contents for a section of an
    INI-style config file. Each entry in the section translates to an
    *attribute* of the instance. Thus, a piece of config file like this::

        [section]
        attribute1 = true
        attribute2 = /foo/bar

    could be accessed like this::

        >>> sect = globalConf[SECTION_NAME]
        >>> sect.attribute1
        'true'
        >>> sect.attribute2
        '/foo/bar'

    The attributes are read-only. Any attempt to set a new one, or change
    the value of an entry through instances of this class, will raise an
    exception.

    As the entries will be translated as Python attributes, this means that
    entry names **have to** be valid Python identifier names.

    There is only one reserved name: `as_dict`. This cannot be used as an
    entry name.

    """
    def __init__(self, values_dict):
        self._set('_contents', values_dict)

    def _set(self, name, value):
        if name == 'as_dict':
            raise RuntimeError("'as_dict' is a reserved name and cannot be "
                               "used as a config entry.")
        self.__dict__[name] = value

    def as_dict(self):
        "Returns a dictionary representation of this section"
        return self._contents.copy()

    def __setattr__(self, attr, value):
        raise RuntimeError("Attribute {0!r} is read-only".format(attr))

    def __getattr__(self, attr):
        try:
            return self._contents[attr]
        except KeyError:
            raise AttributeError("Unknown attribute {0!r}".format(attr))

    def __repr__(self):
        return "<Section [{0}]>".format(', '.join(list(self._contents.keys())))


def environment_variable_name(section, option):
    return '_GEM_{}_{}'.format(section.upper(), option.upper())


globalConf = ConfigObject()
