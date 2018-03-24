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
from builtins import zip
from builtins import str
from builtins import range

import collections

from .config import Field, FieldValidationError, _typeStr, _autocast, _joinNamePath
from .comparison import compareScalars, getComparisonName
from .callStack import getCallStack, getStackFrame

__all__ = ["ListField"]


class List(collections.MutableSequence):
    def __init__(self, config, field, value, at, label, setHistory=True):
        self._field = field
        self._config = config
        self._history = self._config._history.setdefault(self._field.name, [])
        self._list = []
        self.__doc__ = field.doc
        if value is not None:
            try:
                for i, x in enumerate(value):
                    self.insert(i, x, setHistory=False)
            except TypeError:
                msg = "Value %s is of incorrect type %s. Sequence type expected" % (value, _typeStr(value))
                raise FieldValidationError(self._field, self._config, msg)
        if setHistory:
            self.history.append((list(self._list), at, label))

    def validateItem(self, i, x):

        if not isinstance(x, self._field.itemtype) and x is not None:
            msg = "Item at position %d with value %s is of incorrect type %s. Expected %s" % \
                (i, x, _typeStr(x), _typeStr(self._field.itemtype))
            raise FieldValidationError(self._field, self._config, msg)

        if self._field.itemCheck is not None and not self._field.itemCheck(x):
            msg = "Item at position %d is not a valid value: %s" % (i, x)
            raise FieldValidationError(self._field, self._config, msg)

    def list(self):
        return self._list

    """
    Read-only history
    """
    history = property(lambda x: x._history)

    def __contains__(self, x):
        return x in self._list

    def __len__(self):
        return len(self._list)

    def __setitem__(self, i, x, at=None, label="setitem", setHistory=True):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config,
                                       "Cannot modify a frozen Config")
        if isinstance(i, slice):
            k, stop, step = i.indices(len(self))
            for j, xj in enumerate(x):
                xj = _autocast(xj, self._field.itemtype)
                self.validateItem(k, xj)
                x[j] = xj
                k += step
        else:
            x = _autocast(x, self._field.itemtype)
            self.validateItem(i, x)

        self._list[i] = x
        if setHistory:
            if at is None:
                at = getCallStack()
            self.history.append((list(self._list), at, label))

    def __getitem__(self, i):
        return self._list[i]

    def __delitem__(self, i, at=None, label="delitem", setHistory=True):
        if self._config._frozen:
            raise FieldValidationError(self._field, self._config,
                                       "Cannot modify a frozen Config")
        del self._list[i]
        if setHistory:
            if at is None:
                at = getCallStack()
            self.history.append((list(self._list), at, label))

    def __iter__(self):
        return iter(self._list)

    def insert(self, i, x, at=None, label="insert", setHistory=True):
        if at is None:
            at = getCallStack()
        self.__setitem__(slice(i, i), [x], at=at, label=label, setHistory=setHistory)

    def __repr__(self):
        return repr(self._list)

    def __str__(self):
        return str(self._list)

    def __eq__(self, other):
        try:
            if len(self) != len(other):
                return False

            for i, j in zip(self, other):
                if i != j:
                    return False
            return True
        except AttributeError:
            # other is not a sequence type
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __setattr__(self, attr, value, at=None, label="assignment"):
        if hasattr(getattr(self.__class__, attr, None), '__set__'):
            # This allows properties to work.
            object.__setattr__(self, attr, value)
        elif attr in self.__dict__ or attr in ["_field", "_config", "_history", "_list", "__doc__"]:
            # This allows specific private attributes to work.
            object.__setattr__(self, attr, value)
        else:
            # We throw everything else.
            msg = "%s has no attribute %s" % (_typeStr(self._field), attr)
            raise FieldValidationError(self._field, self._config, msg)


class ListField(Field):
    """
    Defines a field which is a container of values of type dtype

    If length is not None, then instances of this field must match this length
    exactly.
    If minLength is not None, then instances of the field must be no shorter
    then minLength
    If maxLength is not None, then instances of the field must be no longer
    than maxLength
    
    If single is True, a single object of dtype rather than a list is OK

    Additionally users can provide two check functions:
    listCheck - used to validate the list as a whole, and
    itemCheck - used to validate each item individually
    """
    def __init__(self, doc, dtype, default=None, optional=False,
                 listCheck=None, itemCheck=None,
                 length=None, minLength=None, maxLength=None, single=False):
        if isinstance(dtype, tuple):
            if any([x not in self.supportedTypes for x in dtype]):
                raise ValueError("Unsupported Field dtype in %s" % repr(dtype))
        elif dtype not in self.supportedTypes:
            raise ValueError("Unsupported Field dtype %s" % _typeStr(dtype))
        if length is not None:
            if length <= 0:
                raise ValueError("'length' (%d) must be positive" % length)
            minLength = None
            maxLength = None
        else:
            if maxLength is not None and maxLength <= 0:
                raise ValueError("'maxLength' (%d) must be positive" % maxLength)
            if minLength is not None and maxLength is not None \
                    and minLength > maxLength:
                raise ValueError("'maxLength' (%d) must be at least"
                                 " as large as 'minLength' (%d)" % (maxLength, minLength))

        if listCheck is not None and not hasattr(listCheck, "__call__"):
            raise ValueError("'listCheck' must be callable")
        if itemCheck is not None and not hasattr(itemCheck, "__call__"):
            raise ValueError("'itemCheck' must be callable")

        source = getStackFrame()
        self._setup(doc=doc, dtype=(List, dtype) if single else List,
                    default=default, check=None, optional=optional, source=source)
        self.listCheck = listCheck
        self.itemCheck = itemCheck
        self.itemtype = dtype
        self.length = length
        self.minLength = minLength
        self.maxLength = maxLength
        self.single = single

    def validate(self, instance):
        """
        ListField validation ensures that non-optional fields are not None,
            and that non-None values comply with length requirements and
            that the list passes listCheck if supplied by the user.
        Individual Item checks are applied at set time and are not re-checked.
        """
        Field.validate(self, instance)
        value = self.__get__(instance)
        if not self.single and value is not None:
            lenValue = len(value)
            if self.length is not None and not lenValue == self.length:
                msg = "Required list length=%d, got length=%d" % (self.length, lenValue)
                raise FieldValidationError(self, instance, msg)
            elif self.minLength is not None and lenValue < self.minLength:
                msg = "Minimum allowed list length=%d, got length=%d" % (self.minLength, lenValue)
                raise FieldValidationError(self, instance, msg)
            elif self.maxLength is not None and lenValue > self.maxLength:
                msg = "Maximum allowed list length=%d, got length=%d" % (self.maxLength, lenValue)
                raise FieldValidationError(self, instance, msg)
            elif self.listCheck is not None and not self.listCheck(value):
                msg = "%s is not a valid value" % str(value)
                raise FieldValidationError(self, instance, msg)

    def __set__(self, instance, value, at=None, label="assignment"):
        if instance._frozen:
            raise FieldValidationError(self, instance, "Cannot modify a frozen Config")

        if at is None:
            at = getCallStack()

        if value is not None:
            if not self.single or isinstance(value, collections.Iterable):
                value = List(instance, self, value, at, label)
            else:
                value = _autocast(value, self.dtype)
                try:
                    self._validateValue(value)
                except BaseException as e:
                    raise FieldValidationError(self, instance, str(e))
        else:
            history = instance._history.setdefault(self.name, [])
            history.append((value, at, label))

        instance._storage[self.name] = value

    def toDict(self, instance):
        value = self.__get__(instance)
        return list(value) if value is not None else None

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
        l1 = getattr(instance1, self.name)
        l2 = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        if not compareScalars("isnone for %s" % name, l1 is None, l2 is None, output=output):
            return False
        if l1 is None and l2 is None:
            return True
        if not compareScalars("size for %s" % name, len(l1), len(l2), output=output):
            return False
        equal = True
        for n, v1, v2 in zip(range(len(l1)), l1, l2):
            result = compareScalars("%s[%d]" % (name, n), v1, v2, dtype=self.dtype,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
