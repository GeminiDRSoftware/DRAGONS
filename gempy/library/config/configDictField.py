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

from .config import Config, FieldValidationError, _autocast, _typeStr, _joinNamePath
from .dictField import Dict, DictField
from .comparison import compareConfigs, compareScalars, getComparisonName
from .callStack import getCallStack, getStackFrame

__all__ = ["ConfigDictField"]


class ConfigDict(Dict):
    """
    Config-Insternal representation of a dict of config classes

    Much like Dict, ConfigDict is a custom MutableMapper which tracks the
    history of changes to any of its items.
    """
    def __init__(self, config, field, value, at, label):
        Dict.__init__(self, config, field, value, at, label, setHistory=False)
        self.history.append(("Dict initialized", at, label))

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
        dtype = self._field.itemtype
        if type(x) != self._field.itemtype and x != self._field.itemtype:
            msg = "Value %s at key %r is of incorrect type %s. Expected type %s" % \
                (x, k, _typeStr(x), _typeStr(self._field.itemtype))
            raise FieldValidationError(self._field, self._config, msg)

        if at is None:
            at = getCallStack()
        name = _joinNamePath(self._config._name, self._field.name, k)
        oldValue = self._dict.get(k, None)
        if oldValue is None:
            if x == dtype:
                self._dict[k] = dtype(__name=name, __at=at, __label=label)
            else:
                self._dict[k] = dtype(__name=name, __at=at, __label=label, **x._storage)
            if setHistory:
                self.history.append(("Added item at key %s" % k, at, label))
        else:
            if x == dtype:
                x = dtype()
            oldValue.update(__at=at, __label=label, **x._storage)
            if setHistory:
                self.history.append(("Modified item at key %s" % k, at, label))

    def __delitem__(self, k, at=None, label="delitem"):
        if at is None:
            at = getCallStack()
        Dict.__delitem__(self, k, at, label, False)
        self.history.append(("Removed item at key %s" % k, at, label))


class ConfigDictField(DictField):
    """
    Defines a field which is a mapping between a POD and a config class.

    This behaves exactly like a DictField with the slight difference that
        itemtype must be an subclass of Config.

    This allows config writters to create name-to-config mappings. One use case
    is for configuring mappings for dataset types in a butler. In this case,
    the dataset type names are arbitrary and user-selected; the mapping
    configurations are known and fixed.
    """

    DictClass = ConfigDict

    def __init__(self, doc, keytype, itemtype, default=None, optional=False, dictCheck=None, itemCheck=None):
        source = getStackFrame()
        self._setup(doc=doc, dtype=ConfigDict, default=default, check=None,
                    optional=optional, source=source)
        if keytype not in self.supportedTypes:
            raise ValueError("'keytype' %s is not a supported type" %
                             _typeStr(keytype))
        elif not issubclass(itemtype, Config):
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

    def rename(self, instance):
        configDict = self.__get__(instance)
        if configDict is not None:
            for k in configDict:
                fullname = _joinNamePath(instance._name, self.name, k)
                configDict[k]._rename(fullname)

    def validate(self, instance):
        value = self.__get__(instance)
        if value is not None:
            for k in value:
                item = value[k]
                item.validate()
                if self.itemCheck is not None and not self.itemCheck(item):
                    msg = "Item at key %r is not a valid value: %s" % (k, item)
                    raise FieldValidationError(self, instance, msg)
        DictField.validate(self, instance)

    def toDict(self, instance):
        configDict = self.__get__(instance)
        if configDict is None:
            return None

        dict_ = {}
        for k in configDict:
            dict_[k] = configDict[k].toDict()

        return dict_

    def save(self, outfile, instance):
        configDict = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)
        if configDict is None:
            outfile.write(u"{}={!r}\n".format(fullname, configDict))
            return

        outfile.write(u"{}={!r}\n".format(fullname, {}))
        for v in configDict.values():
            outfile.write(u"{}={}()\n".format(v._name, _typeStr(v)))
            v._save(outfile)

    def freeze(self, instance):
        configDict = self.__get__(instance)
        if configDict is not None:
            for k in configDict:
                configDict[k].freeze()

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
        if not compareScalars("keys for %s" % name, set(d1.keys()), set(d2.keys()), output=output):
            return False
        equal = True
        for k, v1 in d1.items():
            v2 = d2[k]
            result = compareConfigs("%s[%r]" % (name, k), v1, v2, shortcut=shortcut,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
