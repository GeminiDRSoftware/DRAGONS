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

import collections

from .config import Field, FieldValidationError, _typeStr, _autocast, _joinNamePath
from .comparison import getComparisonName, compareScalars
from .callStack import getCallStack, getStackFrame

__all__ = ["DictField"]


class Dict(collections.MutableMapping):
    """
    Config-Internal mapping container
    Emulates a dict, but adds validation and provenance.
    """

    def __init__(self, config, field, value, at, label, setHistory=True):
        self._field = field
        self._config = config
        self._dict = {}
        self._history = self._config._history.setdefault(self._field.name, [])
        self.__doc__ = field.doc
        if value is not None:
            try:
                for k in value:
                    # do not set history per-item
                    self.__setitem__(k, value[k], at=at, label=label, setHistory=False)
            except TypeError:
                msg = "Value %s is of incorrect type %s. Mapping type expected." % \
                    (value, _typeStr(value))
                raise FieldValidationError(self._field, self._config, msg)
        if setHistory:
            self._history.append((dict(self._dict), at, label))

    """
    Read-only history
    """
    history = property(lambda x: x._history)

    def __getitem__(self, k):
        return self._dict[k]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, k):
        return k in self._dict

    def __setitem__(self, k, x, at=None, label="setitem", setHistory=True):
        if self._config._frozen:
            msg = "Cannot modify a frozen Config. "\
                "Attempting to set item at key %r to value %s" % (k, x)
            raise FieldValidationError(self._field, self._config, msg)

        # validate keytype
        k = _autocast(k, self._field.keytype)
        if type(k) != self._field.keytype:
            msg = "Key %r is of type %s, expected type %s" % \
                (k, _typeStr(k), _typeStr(self._field.keytype))
            raise FieldValidationError(self._field, self._config, msg)

        # validate itemtype
        x = _autocast(x, self._field.itemtype)
        if self._field.itemtype is None:
            if type(x) not in self._field.supportedTypes and x is not None:
                msg = "Value %s at key %r is of invalid type %s" % (x, k, _typeStr(x))
                raise FieldValidationError(self._field, self._config, msg)
        else:
            if type(x) != self._field.itemtype and x is not None:
                msg = "Value %s at key %r is of incorrect type %s. Expected type %s" % \
                    (x, k, _typeStr(x), _typeStr(self._field.itemtype))
                raise FieldValidationError(self._field, self._config, msg)

        # validate item using itemcheck
        if self._field.itemCheck is not None and not self._field.itemCheck(x):
            msg = "Item at key %r is not a valid value: %s" % (k, x)
            raise FieldValidationError(self._field, self._config, msg)

        if at is None:
            at = getCallStack()

        self._dict[k] = x
        if setHistory:
            self._history.append((dict(self._dict), at, label))

    def __delitem__(self, k, at=None, label="delitem", setHistory=True):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config,
                                       "Cannot modify a frozen Config")

        del self._dict[k]
        if setHistory:
            if at is None:
                at = getCallStack()
            self._history.append((dict(self._dict), at, label))

    def __repr__(self):
        return repr(self._dict)

    def __str__(self):
        return str(self._dict)

    def __setattr__(self, attr, value, at=None, label="assignment"):
        if hasattr(getattr(self.__class__, attr, None), '__set__'):
            # This allows properties to work.
            object.__setattr__(self, attr, value)
        elif attr in self.__dict__ or attr in ["_field", "_config", "_history", "_dict", "__doc__"]:
            # This allows specific private attributes to work.
            object.__setattr__(self, attr, value)
        else:
            # We throw everything else.
            msg = "%s has no attribute %s" % (_typeStr(self._field), attr)
            raise FieldValidationError(self._field, self._config, msg)


class DictField(Field):
    """
    Defines a field which is a mapping of values

    Both key and item types are restricted to builtin POD types:
        (int, float, complex, bool, str)

    Users can provide two check functions:
        dictCheck: used to validate the dict as a whole, and
        itemCheck: used to validate each item individually

    For example to define a field which is a mapping from names to int values:

    class MyConfig(Config):
        field = DictField(
                doc="example string-to-int mapping field",
                keytype=str, itemtype=int,
                default= {})
    """
    DictClass = Dict

    def __init__(self, doc, keytype, itemtype, default=None, optional=False, dictCheck=None, itemCheck=None):
        source = getStackFrame()
        self._setup(doc=doc, dtype=Dict, default=default, check=None,
                    optional=optional, source=source)
        if keytype not in self.supportedTypes:
            raise ValueError("'keytype' %s is not a supported type" %
                             _typeStr(keytype))
        elif itemtype is not None and itemtype not in self.supportedTypes:
            raise ValueError("'itemtype' %s is not a supported type" %
                             _typeStr(itemtype))
        if dictCheck is not None and not hasattr(dictCheck, "__call__"):
            raise ValueError("'dictCheck' must be callable")
        if itemCheck is not None and not hasattr(itemCheck, "__call__"):
            raise ValueError("'itemCheck' must be callable")

        self.keytype = keytype
        self.itemtype = itemtype
        self.dictCheck = dictCheck
        self.itemCheck = itemCheck

    def validate(self, instance):
        """
        DictField validation ensures that non-optional fields are not None,
            and that non-None values comply with dictCheck.
        Individual Item checks are applied at set time and are not re-checked.
        """
        Field.validate(self, instance)
        value = self.__get__(instance)
        if value is not None and self.dictCheck is not None \
                and not self.dictCheck(value):
            msg = "%s is not a valid value" % str(value)
            raise FieldValidationError(self, instance, msg)

    def __set__(self, instance, value, at=None, label="assignment"):
        if instance._frozen:
            msg = "Cannot modify a frozen Config. "\
                  "Attempting to set field to value %s" % value
            raise FieldValidationError(self, instance, msg)

        if at is None:
            at = getCallStack()
        if value is not None:
            value = self.DictClass(instance, self, value, at=at, label=label)
        else:
            history = instance._history.setdefault(self.name, [])
            history.append((value, at, label))

        instance._storage[self.name] = value

    def toDict(self, instance):
        value = self.__get__(instance)
        return dict(value) if value is not None else None

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
        d1 = getattr(instance1, self.name)
        d2 = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        if not compareScalars("isnone for %s" % name, d1 is None, d2 is None, output=output):
            return False
        if d1 is None and d2 is None:
            return True
        if not compareScalars("keys for %s" % name, set(d1.keys()), set(d2.keys()), output=output):
            return False
        equal = True
        for k, v1 in d1.items():
            v2 = d2[k]
            result = compareScalars("%s[%r]" % (name, k), v1, v2, dtype=self.itemtype,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
