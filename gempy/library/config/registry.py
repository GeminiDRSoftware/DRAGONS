#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from builtins import object

import collections
import copy

from .config import Config, FieldValidationError, _typeStr
from .configChoiceField import ConfigInstanceDict, ConfigChoiceField

__all__ = ("Registry", "makeRegistry", "RegistryField", "registerConfig", "registerConfigurable")


class ConfigurableWrapper(object):
    """A wrapper for configurables

    Used for configurables that don't contain a ConfigClass attribute,
    or contain one that is being overridden.
    """
    def __init__(self, target, ConfigClass):
        self.ConfigClass = ConfigClass
        self._target = target

    def __call__(self, *args, **kwargs):
        return self._target(*args, **kwargs)


class Registry(collections.Mapping):
    """A base class for global registries, mapping names to configurables.

    There are no hard requirements on configurable, but they typically create an algorithm
    or are themselves the algorithm, and typical usage is as follows:
    - configurable is a callable whose call signature is (config, ...extra arguments...)
    - All configurables added to a particular registry will have the same call signature
    - All configurables in a registry will typically share something important in common.
      For example all configurables in psfMatchingRegistry return a psf matching
      class that has a psfMatch method with a particular call signature.

    A registry acts like a read-only dictionary with an additional register method to add items.
    The dict contains configurables and each configurable has an instance ConfigClass.

    Example:
    registry = Registry()
    class FooConfig(Config):
        val = Field(dtype=int, default=3, doc="parameter for Foo")
    class Foo(object):
        ConfigClass = FooConfig
        def __init__(self, config):
            self.config = config
        def addVal(self, num):
            return self.config.val + num
    registry.register("foo", Foo)
    names = registry.keys() # returns ("foo",)
    fooConfigurable = registry["foo"]
    fooConfig = fooItem.ConfigClass()
    foo = fooConfigurable(fooConfig)
    foo.addVal(5) # returns config.val + 5
    """

    def __init__(self, configBaseType=Config):
        """Construct a registry of name: configurables

        @param configBaseType: base class for config classes in registry
        """
        if not issubclass(configBaseType, Config):
            raise TypeError("configBaseType=%s must be a subclass of Config" % _typeStr(configBaseType,))
        self._configBaseType = configBaseType
        self._dict = {}

    def register(self, name, target, ConfigClass=None):
        """Add a new item to the registry.

        @param target       A callable 'object that takes a Config instance as its first argument.
                            This may be a Python type, but is not required to be.
        @param ConfigClass  A subclass of pex_config Config used to configure the configurable;
                            if None then configurable.ConfigClass is used.

        @note: If ConfigClass is provided then then 'target' is wrapped in a new object that forwards
               function calls to it.  Otherwise the original 'target' is stored.

        @raise AttributeError if ConfigClass is None and target does not have attribute ConfigClass
        """
        if name in self._dict:
            raise RuntimeError("An item with name %r already exists" % name)
        if ConfigClass is None:
            wrapper = target
        else:
            wrapper = ConfigurableWrapper(target, ConfigClass)
        if not issubclass(wrapper.ConfigClass, self._configBaseType):
            raise TypeError("ConfigClass=%s is not a subclass of %r" %
                            (_typeStr(wrapper.ConfigClass), _typeStr(self._configBaseType)))
        self._dict[name] = wrapper

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def makeField(self, doc, default=None, optional=False, multi=False):
        return RegistryField(doc, self, default, optional, multi)


class RegistryAdaptor(collections.Mapping):
    """Private class that makes a Registry behave like the thing a ConfigChoiceField expects."""

    def __init__(self, registry):
        self.registry = registry

    def __getitem__(self, k):
        return self.registry[k].ConfigClass

    def __iter__(self):
        return iter(self.registry)

    def __len__(self):
        return len(self.registry)

    def __contains__(self, k):
        return k in self.registry


class RegistryInstanceDict(ConfigInstanceDict):
    def __init__(self, config, field):
        ConfigInstanceDict.__init__(self, config, field)
        self.registry = field.registry

    def _getTarget(self):
        if self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Multi-selection field has no attribute 'target'")
        return self._field.typemap.registry[self._selection]
    target = property(_getTarget)

    def _getTargets(self):
        if not self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Single-selection field has no attribute 'targets'")
        return [self._field.typemap.registry[c] for c in self._selection]
    targets = property(_getTargets)

    def apply(self, *args, **kw):
        """Call the active target(s) with the active config as a keyword arg

        If this is a multi-selection field, return a list obtained by calling
        each active target with its corresponding active config.

        Additional arguments will be passed on to the configurable target(s)
        """
        if self.active is None:
            msg = "No selection has been made.  Options: %s" % \
                (" ".join(list(self._field.typemap.registry.keys())))
            raise FieldValidationError(self._field, self._config, msg)
        if self._field.multi:
            retvals = []
            for c in self._selection:
                retvals.append(self._field.typemap.registry[c](*args, config=self[c], **kw))
            return retvals
        else:
            return self._field.typemap.registry[self.name](*args, config=self[self.name], **kw)

    def __setattr__(self, attr, value):
        if attr == "registry":
            object.__setattr__(self, attr, value)
        else:
            ConfigInstanceDict.__setattr__(self, attr, value)


class RegistryField(ConfigChoiceField):
    instanceDictClass = RegistryInstanceDict

    def __init__(self, doc, registry, default=None, optional=False, multi=False):
        types = RegistryAdaptor(registry)
        self.registry = registry
        ConfigChoiceField.__init__(self, doc, types, default, optional, multi)

    def __deepcopy__(self, memo):
        """Customize deep-copying, want a reference to the original registry.
        WARNING: this must be overridden by subclasses if they change the
            constructor signature!
        """
        other = type(self)(doc=self.doc, registry=self.registry,
                           default=copy.deepcopy(self.default),
                           optional=self.optional, multi=self.multi)
        other.source = self.source
        return other


def makeRegistry(doc, configBaseType=Config):
    """A convenience function to create a new registry.

    The returned value is an instance of a trivial subclass of Registry whose only purpose is to
    customize its doc string and set attrList.
    """
    cls = type("Registry", (Registry,), {"__doc__": doc})
    return cls(configBaseType=configBaseType)


def registerConfigurable(name, registry, ConfigClass=None):
    """A decorator that adds a class as a configurable in a Registry.

    If the 'ConfigClass' argument is None, the class's ConfigClass attribute will be used.
    """
    def decorate(cls):
        registry.register(name, target=cls, ConfigClass=ConfigClass)
        return cls
    return decorate


def registerConfig(name, registry, target):
    """A decorator that adds a class as a ConfigClass in a Registry, and associates it with the given
    configurable.
    """
    def decorate(cls):
        registry.register(name, target=target, ConfigClass=cls)
        return cls
    return decorate
