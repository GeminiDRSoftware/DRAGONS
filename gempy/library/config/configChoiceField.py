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
from __future__ import print_function
from builtins import str

import copy
import collections

from .config import Config, Field, FieldValidationError, _typeStr, _joinNamePath
from .comparison import getComparisonName, compareScalars, compareConfigs
from .callStack import getCallStack, getStackFrame

__all__ = ["ConfigChoiceField"]


class SelectionSet(collections.MutableSet):
    """
    Custom set class used to track the selection of multi-select
    ConfigChoiceField.

    This class allows user a multi-select ConfigChoiceField to add/discard
    items from the set of active configs. Each change to the selection is
    tracked in the field's history.
    """
    def __init__(self, dict_, value, at=None, label="assignment", setHistory=True):
        if at is None:
            at = getCallStack()
        self._dict = dict_
        self._field = self._dict._field
        self._config = self._dict._config
        self.__history = self._config._history.setdefault(self._field.name, [])
        if value is not None:
            try:
                for v in value:
                    if v not in self._dict:
                        # invoke __getitem__ to ensure it's present
                        self._dict.__getitem__(v, at=at)
            except TypeError:
                msg = "Value %s is of incorrect type %s. Sequence type expected"(value, _typeStr(value))
                raise FieldValidationError(self._field, self._config, msg)
            self._set = set(value)
        else:
            self._set = set()

        if setHistory:
            self.__history.append(("Set selection to %s" % self, at, label))

    def add(self, value, at=None):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config,
                                       "Cannot modify a frozen Config")

        if at is None:
            at = getCallStack()

        if value not in self._dict:
            # invoke __getitem__ to make sure it's present
            self._dict.__getitem__(value, at=at)

        self.__history.append(("added %s to selection" % value, at, "selection"))
        self._set.add(value)

    def discard(self, value, at=None):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config,
                                       "Cannot modify a frozen Config")

        if value not in self._dict:
            return

        if at is None:
            at = getCallStack()

        self.__history.append(("removed %s from selection" % value, at, "selection"))
        self._set.discard(value)

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._set)

    def __contains__(self, value):
        return value in self._set

    def __repr__(self):
        return repr(list(self._set))

    def __str__(self):
        return str(list(self._set))


class ConfigInstanceDict(collections.Mapping):
    """A dict of instantiated configs, used to populate a ConfigChoiceField.

    typemap must support the following:
    - typemap[name]: return the config class associated with the given name
    """
    def __init__(self, config, field):
        collections.Mapping.__init__(self)
        self._dict = dict()
        self._selection = None
        self._config = config
        self._field = field
        self._history = config._history.setdefault(field.name, [])
        self.__doc__ = field.doc

    types = property(lambda x: x._field.typemap)

    def __contains__(self, k):
        return k in self._field.typemap

    def __len__(self):
        return len(self._field.typemap)

    def __iter__(self):
        return iter(self._field.typemap)

    def _setSelection(self, value, at=None, label="assignment"):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config, "Cannot modify a frozen Config")

        if at is None:
            at = getCallStack(1)

        if value is None:
            self._selection = None
        elif self._field.multi:
            self._selection = SelectionSet(self, value, setHistory=False)
        else:
            if value not in self._dict:
                self.__getitem__(value, at=at)  # just invoke __getitem__ to make sure it's present
            self._selection = value
        self._history.append((value, at, label))

    def _getNames(self):
        if not self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Single-selection field has no attribute 'names'")
        return self._selection

    def _setNames(self, value):
        if not self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Single-selection field has no attribute 'names'")
        self._setSelection(value)

    def _delNames(self):
        if not self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Single-selection field has no attribute 'names'")
        self._selection = None

    def _getName(self):
        if self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Multi-selection field has no attribute 'name'")
        return self._selection

    def _setName(self, value):
        if self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Multi-selection field has no attribute 'name'")
        self._setSelection(value)

    def _delName(self):
        if self._field.multi:
            raise FieldValidationError(self._field, self._config,
                                       "Multi-selection field has no attribute 'name'")
        self._selection = None

    """
    In a multi-selection ConfigInstanceDict, list of names of active items
    Disabled In a single-selection _Regsitry)
    """
    names = property(_getNames, _setNames, _delNames)

    """
    In a single-selection ConfigInstanceDict, name of the active item
    Disabled In a multi-selection _Regsitry)
    """
    name = property(_getName, _setName, _delName)

    def _getActive(self):
        if self._selection is None:
            return None

        if self._field.multi:
            return [self[c] for c in self._selection]
        else:
            return self[self._selection]

    """
    Readonly shortcut to access the selected item(s)
    for multi-selection, this is equivalent to: [self[name] for name in self.names]
    for single-selection, this is equivalent to: self[name]
    """
    active = property(_getActive)

    def __getitem__(self, k, at=None, label="default"):
        try:
            value = self._dict[k]
        except KeyError:
            try:
                dtype = self._field.typemap[k]
            except Exception:
                raise FieldValidationError(self._field, self._config,
                                           "Unknown key %r in Registry/ConfigChoiceField" % k)
            name = _joinNamePath(self._config._name, self._field.name, k)
            if at is None:
                at = getCallStack()
                at.insert(0, dtype._source)
            value = self._dict.setdefault(k, dtype(__name=name, __at=at, __label=label))
        return value

    def __setitem__(self, k, value, at=None, label="assignment"):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config, "Cannot modify a frozen Config")

        try:
            dtype = self._field.typemap[k]
        except Exception:
            raise FieldValidationError(self._field, self._config, "Unknown key %r" % k)

        if value != dtype and type(value) != dtype:
            msg = "Value %s at key %k is of incorrect type %s. Expected type %s" % \
                (value, k, _typeStr(value), _typeStr(dtype))
            raise FieldValidationError(self._field, self._config, msg)

        if at is None:
            at = getCallStack()
        name = _joinNamePath(self._config._name, self._field.name, k)
        oldValue = self._dict.get(k, None)
        if oldValue is None:
            if value == dtype:
                self._dict[k] = value(__name=name, __at=at, __label=label)
            else:
                self._dict[k] = dtype(__name=name, __at=at, __label=label, **value._storage)
        else:
            if value == dtype:
                value = value()
            oldValue.update(__at=at, __label=label, **value._storage)

    def _rename(self, fullname):
        for k, v in self._dict.items():
            v._rename(_joinNamePath(name=fullname, index=k))

    def __setattr__(self, attr, value, at=None, label="assignment"):
        if hasattr(getattr(self.__class__, attr, None), '__set__'):
            # This allows properties to work.
            object.__setattr__(self, attr, value)
        elif attr in self.__dict__ or attr in ["_history", "_field", "_config", "_dict",
                                               "_selection", "__doc__"]:
            # This allows specific private attributes to work.
            object.__setattr__(self, attr, value)
        else:
            # We throw everything else.
            msg = "%s has no attribute %s" % (_typeStr(self._field), attr)
            raise FieldValidationError(self._field, self._config, msg)


class ConfigChoiceField(Field):
    """
    ConfigChoiceFields allow the config to choose from a set of possible Config types.
    The set of allowable types is given by the typemap argument to the constructor

    The typemap object must implement typemap[name], which must return a Config subclass.

    While the typemap is shared by all instances of the field, each instance of
    the field has its own instance of a particular sub-config type

    For example:

      class AaaConfig(Config):
        somefield = Field(int, "...")
      TYPEMAP = {"A", AaaConfig}
      class MyConfig(Config):
          choice = ConfigChoiceField("doc for choice", TYPEMAP)

      instance = MyConfig()
      instance.choice['AAA'].somefield = 5
      instance.choice = "AAA"

    Alternatively, the last line can be written:
      instance.choice.name = "AAA"

    Validation of this field is performed only the "active" selection.
    If active is None and the field is not optional, validation will fail.

    ConfigChoiceFields can allow single selections or multiple selections.
    Single selection fields set selection through property name, and
    multi-selection fields use the property names.

    ConfigChoiceFields also allow multiple values of the same type:
      TYPEMAP["CCC"] = AaaConfig
      TYPEMAP["BBB"] = AaaConfig

    When saving a config with a ConfigChoiceField, the entire set is saved, as well as the active selection
    """
    instanceDictClass = ConfigInstanceDict

    def __init__(self, doc, typemap, default=None, optional=False, multi=False):
        source = getStackFrame()
        self._setup(doc=doc, dtype=self.instanceDictClass, default=default, check=None, optional=optional,
                    source=source)
        self.typemap = typemap
        self.multi = multi

    def _getOrMake(self, instance, label="default"):
        instanceDict = instance._storage.get(self.name)
        if instanceDict is None:
            at = getCallStack(1)
            instanceDict = self.dtype(instance, self)
            instanceDict.__doc__ = self.doc
            instance._storage[self.name] = instanceDict
            history = instance._history.setdefault(self.name, [])
            history.append(("Initialized from defaults", at, label))

        return instanceDict

    def __get__(self, instance, owner=None):
        if instance is None or not isinstance(instance, Config):
            return self
        else:
            return self._getOrMake(instance)

    def __set__(self, instance, value, at=None, label="assignment"):
        if instance._frozen:
            raise FieldValidationError(self, instance, "Cannot modify a frozen Config")
        if at is None:
            at = getCallStack()
        instanceDict = self._getOrMake(instance)
        if isinstance(value, self.instanceDictClass):
            for k, v in value.items():
                instanceDict.__setitem__(k, v, at=at, label=label)
            instanceDict._setSelection(value._selection, at=at, label=label)

        else:
            instanceDict._setSelection(value, at=at, label=label)

    def rename(self, instance):
        instanceDict = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)
        instanceDict._rename(fullname)

    def validate(self, instance):
        instanceDict = self.__get__(instance)
        if instanceDict.active is None and not self.optional:
            msg = "Required field cannot be None"
            raise FieldValidationError(self, instance, msg)
        elif instanceDict.active is not None:
            if self.multi:
                for a in instanceDict.active:
                    a.validate()
            else:
                instanceDict.active.validate()

    def toDict(self, instance):
        instanceDict = self.__get__(instance)

        dict_ = {}
        if self.multi:
            dict_["names"] = instanceDict.names
        else:
            dict_["name"] = instanceDict.name

        values = {}
        for k, v in instanceDict.items():
            values[k] = v.toDict()
        dict_["values"] = values

        return dict_

    def freeze(self, instance):
        instanceDict = self.__get__(instance)
        for v in instanceDict.values():
            v.freeze()

    def save(self, outfile, instance):
        instanceDict = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)
        for v in instanceDict.values():
            v._save(outfile)
        if self.multi:
            outfile.write(u"{}.names={!r}\n".format(fullname, instanceDict.names))
        else:
            outfile.write(u"{}.name={!r}\n".format(fullname, instanceDict.name))

    def __deepcopy__(self, memo):
        """Customize deep-copying, because we always want a reference to the original typemap.

        WARNING: this must be overridden by subclasses if they change the constructor signature!
        """
        other = type(self)(doc=self.doc, typemap=self.typemap, default=copy.deepcopy(self.default),
                           optional=self.optional, multi=self.multi)
        other.source = self.source
        return other

    def _compare(self, instance1, instance2, shortcut, rtol, atol, output):
        """Helper function for Config.compare; used to compare two fields for equality.

        Only the selected config(s) are compared, as the parameters of any others do not matter.

        @param[in] instance1  LHS Config instance to compare.
        @param[in] instance2  RHS Config instance to compare.
        @param[in] shortcut   If True, return as soon as an inequality is found.
        @param[in] rtol       Relative tolerance for floating point comparisons.
        @param[in] atol       Absolute tolerance for floating point comparisons.
        @param[in] output     If not None, a callable that takes a string, used (possibly repeatedly)
                              to report inequalities.

        Floating point comparisons are performed by numpy.allclose; refer to that for details.
        """
        d1 = getattr(instance1, self.name)
        d2 = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        if not compareScalars("selection for %s" % name, d1._selection, d2._selection, output=output):
            return False
        if d1._selection is None:
            return True
        if self.multi:
            nested = [(k, d1[k], d2[k]) for k in d1._selection]
        else:
            nested = [(d1._selection, d1[d1._selection], d2[d1._selection])]
        equal = True
        for k, c1, c2 in nested:
            result = compareConfigs("%s[%r]" % (name, k), c1, c2, shortcut=shortcut,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
