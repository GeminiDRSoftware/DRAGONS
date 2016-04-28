import os
import types
from ConfigParser import SafeConfigParser

STANDARD_REDUCTION_CONF = '~/.geminidr/rsys.cfg'

class Section(object):
    def __init__(self, values_dict):
        self._set('_contents', values_dict)

    def _set(self, name, value):
        self.__dict__[name] = value

    def as_dict(self):
        return self._contents.copy()

    def __setattr__(self, attr, value):
        raise RuntimeError("Attribute {0!r} is read-only".format(attr))

    def __getattr__(self, attr):
        try:
            return self._contents[attr]
        except KeyError:
            raise AttributeError("Unknown attribute {0!r}".format(attr))

    def __repr__(self):
        return "<Section [{0}]>".format(', '.join(self._contents.keys()))

class Converter(object):
    def __init__(self, conv_dict, cp):
        self._trans = {}
        self._default = cp.get

        for pair, type_ in conv_dict.items():
            if type_ == int:
                self._trans[pair] = cp.getint
            elif type_ == float:
                self._trans[pair] = cp.getfloat
            elif type_ == bool:
                self._trans[pair] = cp.getboolean

    def __call__(self, section, value):
        try:
            return self._trans[(section, value)](section, value)
        except KeyError:
            return self._trans.get((None, value), self._default)(section, value)

class ConfigObject(object):
    def __init__(self):
        self._sections = {}
        self._conv = {}

    def __getitem__(self, item):
        try:
            return self._sections[item.lower()]
        except KeyError:
            raise KeyError("There is no {0!r} section".format(item))

    def update(self, section, values):
        """Regenerates a section from scratch. If the section had been loaded
        before, it will take the previous values as a basis and update them
        with the new ones.

        Parameters
        ----------
        """
        prev = self._sections[section].as_dict() if section in self._sections else {}
        prev.update(values)
        self._sections[section] = Section(values)

    def update_translation(self, conv):
        """
        Parameters
        ----------
        conv : dict
            A mapping `(section_name, item)` -> Python type. Used internally for
            type translation when reading values from the config file. If a
            section/item pair is missing then a fallback `(None, item)` will be
            tried. If no match is found, no translation will be performed.

            The only types to be considered are: `int`, `float`, `bool`
        """
        self._conv.update(conv)

    def load(self, filenames, defaults=None):
        """Loads all or some entries from the specified section in a config file.
        The extracted values are set as environment variables, so that they are
        available at a later point to other modules or spawned processes.

        Parameters
        ----------
        filenames : string or iterable object
            A string or a sequence of string containing the path(s) to
            configuration file(s). If a value is present in more than one of the
            files, the latest one to be processed overrides the preceding ones.

            Paths can start with `~/`, meaning the user home directory.

        defaults : dict, optional
            If some options are not found, and you want to set up a default value,
            specify them in here.
        """

        if type(filenames) in types.StringTypes:
            filenames = (filenames,)


        cp = SafeConfigParser()
        cp.read(map(os.path.expanduser, filenames))

        translate = Converter(self._conv.copy(), cp)

        for section in cp.sections():
            values = {}
            if type(defaults) == dict:
                values.update(defaults)

            for key in cp.options(section):
                values[key] = translate(section, key)

            self.update(section, values)

globalConf = ConfigObject()
