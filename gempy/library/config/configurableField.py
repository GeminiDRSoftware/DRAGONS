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
from builtins import str
from builtins import object

import copy

from .config import Config, Field, _joinNamePath, _typeStr, FieldValidationError
from .comparison import compareConfigs, getComparisonName
from .callStack import getCallStack, getStackFrame


class ConfigurableInstance(object):
    def __initValue(self, at, label):
        """
        if field.default is an instance of ConfigClass, custom construct
        _value with the correct values from default.
        otherwise call ConfigClass constructor
        """
        name = _joinNamePath(self._config._name, self._field.name)
        if type(self._field.default) == self.ConfigClass:
            storage = self._field.default._storage
        else:
            storage = {}
        value = self._ConfigClass(__name=name, __at=at, __label=label, **storage)
        object.__setattr__(self, "_value", value)

    def __init__(self, config, field, at=None, label="default"):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_field", field)
        object.__setattr__(self, "__doc__", config)
        object.__setattr__(self, "_target", field.target)
        object.__setattr__(self, "_ConfigClass", field.ConfigClass)
        object.__setattr__(self, "_value", None)

        if at is None:
            at = getCallStack()
        at += [self._field.source]
        self.__initValue(at, label)

        history = config._history.setdefault(field.name, [])
        history.append(("Targeted and initialized from defaults", at, label))

    """
    Read-only access to the targeted configurable
    """
    target = property(lambda x: x._target)
    """
    Read-only access to the ConfigClass
    """
    ConfigClass = property(lambda x: x._ConfigClass)

    """
    Read-only access to the ConfigClass instance
    """
    value = property(lambda x: x._value)

    def apply(self, *args, **kw):
        """
        Call the confirurable.
        With argument config=self.value along with any positional and kw args
        """
        return self.target(*args, config=self.value, **kw)

    """
    Target a new configurable and ConfigClass
    """
    def retarget(self, target, ConfigClass=None, at=None, label="retarget"):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config, "Cannot modify a frozen Config")

        try:
            ConfigClass = self._field.validateTarget(target, ConfigClass)
        except BaseException as e:
            raise FieldValidationError(self._field, self._config, e.message)

        if at is None:
            at = getCallStack()
        object.__setattr__(self, "_target", target)
        if ConfigClass != self.ConfigClass:
            object.__setattr__(self, "_ConfigClass", ConfigClass)
            self.__initValue(at, label)

        history = self._config._history.setdefault(self._field.name, [])
        msg = "retarget(target=%s, ConfigClass=%s)" % (_typeStr(target), _typeStr(ConfigClass))
        history.append((msg, at, label))

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __setattr__(self, name, value, at=None, label="assignment"):
        """
        Pretend to be an isntance of  ConfigClass.
        Attributes defined by ConfigurableInstance will shadow those defined in ConfigClass
        """
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config, "Cannot modify a frozen Config")

        if name in self.__dict__:
            # attribute exists in the ConfigurableInstance wrapper
            object.__setattr__(self, name, value)
        else:
            if at is None:
                at = getCallStack()
            self._value.__setattr__(name, value, at=at, label=label)

    def __delattr__(self, name, at=None, label="delete"):
        """
        Pretend to be an isntance of  ConfigClass.
        Attributes defiend by ConfigurableInstance will shadow those defined in ConfigClass
        """
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config, "Cannot modify a frozen Config")

        try:
            # attribute exists in the ConfigurableInstance wrapper
            object.__delattr__(self, name)
        except AttributeError:
            if at is None:
                at = getCallStack()
            self._value.__delattr__(name, at=at, label=label)


class ConfigurableField(Field):
    """
    A variant of a ConfigField which has a known configurable target

    Behaves just like a ConfigField except that it can be 'retargeted' to point
    at a different configurable. Further you can 'apply' to construct a fully
    configured configurable.


    """

    def validateTarget(self, target, ConfigClass):
        if ConfigClass is None:
            try:
                ConfigClass = target.ConfigClass
            except Exception:
                raise AttributeError("'target' must define attribute 'ConfigClass'")
        if not issubclass(ConfigClass, Config):
            raise TypeError("'ConfigClass' is of incorrect type %s."
                            "'ConfigClass' must be a subclass of Config" % _typeStr(ConfigClass))
        if not hasattr(target, '__call__'):
            raise ValueError("'target' must be callable")
        if not hasattr(target, '__module__') or not hasattr(target, '__name__'):
            raise ValueError("'target' must be statically defined"
                             "(must have '__module__' and '__name__' attributes)")
        return ConfigClass

    def __init__(self, doc, target, ConfigClass=None, default=None, check=None):
        """
        @param target is the configurable target. Must be callable, and the first
                parameter will be the value of this field
        @param ConfigClass is the class of Config object expected by the target.
                If not provided by target.ConfigClass it must be provided explicitly in this argument
        """
        ConfigClass = self.validateTarget(target, ConfigClass)

        if default is None:
            default = ConfigClass
        if default != ConfigClass and type(default) != ConfigClass:
            raise TypeError("'default' is of incorrect type %s. Expected %s" %
                            (_typeStr(default), _typeStr(ConfigClass)))

        source = getStackFrame()
        self._setup(doc=doc, dtype=ConfigurableInstance, default=default,
                    check=check, optional=False, source=source)
        self.target = target
        self.ConfigClass = ConfigClass

    def __getOrMake(self, instance, at=None, label="default"):
        value = instance._storage.get(self.name, None)
        if value is None:
            if at is None:
                at = getCallStack(1)
            value = ConfigurableInstance(instance, self, at=at, label=label)
            instance._storage[self.name] = value
        return value

    def __get__(self, instance, owner=None, at=None, label="default"):
        if instance is None or not isinstance(instance, Config):
            return self
        else:
            return self.__getOrMake(instance, at=at, label=label)

    def __set__(self, instance, value, at=None, label="assignment"):
        if instance._frozen:
            raise FieldValidationError(self, instance, "Cannot modify a frozen Config")
        if at is None:
            at = getCallStack()
        oldValue = self.__getOrMake(instance, at=at)

        if isinstance(value, ConfigurableInstance):
            oldValue.retarget(value.target, value.ConfigClass, at, label)
            oldValue.update(__at=at, __label=label, **value._storage)
        elif type(value) == oldValue._ConfigClass:
            oldValue.update(__at=at, __label=label, **value._storage)
        elif value == oldValue.ConfigClass:
            value = oldValue.ConfigClass()
            oldValue.update(__at=at, __label=label, **value._storage)
        else:
            msg = "Value %s is of incorrect type %s. Expected %s" % \
                (value, _typeStr(value), _typeStr(oldValue.ConfigClass))
            raise FieldValidationError(self, instance, msg)

    def rename(self, instance):
        fullname = _joinNamePath(instance._name, self.name)
        value = self.__getOrMake(instance)
        value._rename(fullname)

    def save(self, outfile, instance):
        fullname = _joinNamePath(instance._name, self.name)
        value = self.__getOrMake(instance)
        target = value.target

        if target != self.target:
            # not targeting the field-default target.
            # save target information
            ConfigClass = value.ConfigClass
            for module in set([target.__module__, ConfigClass.__module__]):
                outfile.write(u"import {}\n".format(module))
            outfile.write(u"{}.retarget(target={}, ConfigClass={})\n\n".format(fullname,
                                                                               _typeStr(target),
                                                                               _typeStr(ConfigClass)))
        # save field values
        value._save(outfile)

    def freeze(self, instance):
        value = self.__getOrMake(instance)
        value.freeze()

    def toDict(self, instance):
        value = self.__get__(instance)
        return value.toDict()

    def validate(self, instance):
        value = self.__get__(instance)
        value.validate()

        if self.check is not None and not self.check(value):
            msg = "%s is not a valid value" % str(value)
            raise FieldValidationError(self, instance, msg)

    def __deepcopy__(self, memo):
        """Customize deep-copying, because we always want a reference to the original typemap.

        WARNING: this must be overridden by subclasses if they change the constructor signature!
        """
        return type(self)(doc=self.doc, target=self.target, ConfigClass=self.ConfigClass,
                          default=copy.deepcopy(self.default))

    def _compare(self, instance1, instance2, shortcut, rtol, atol, output):
        """Helper function for Config.compare; used to compare two fields for equality.

        @param[in] instance1  LHS Config instance to compare.
        @param[in] instance2  RHS Config instance to compare.
        @param[in] shortcut   If True, return as soon as an inequality is found.
        @param[in] rtol       Relative tolerance for floating point comparisons.
        @param[in] atol       Absolute tolerance for floating point comparisons.
        @param[in] output     If not None, a callable that takes a string, used (possibly repeatedly)
                              to report inequalities.

        Floating point comparisons are performed by numpy.allclose; refer to that for details.
        """
        c1 = getattr(instance1, self.name)._value
        c2 = getattr(instance2, self.name)._value
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        return compareConfigs(name, c1, c2, shortcut=shortcut, rtol=rtol, atol=atol, output=output)
