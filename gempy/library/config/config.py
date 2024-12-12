#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from collections import OrderedDict

import os
import io
import sys
import math
import copy
import tempfile
import shutil
import itertools

from .comparison import getComparisonName, compareScalars, compareConfigs
from .callStack import getStackFrame, getCallStack
from future.utils import with_metaclass

from astrodata import AstroData

__all__ = ("Config", "Field", "FieldValidationError")


def _joinNamePath(prefix=None, name=None, index=None):
    """
    Utility function for generating nested configuration names
    """
    if not prefix and not name:
        raise ValueError("Invalid name: cannot be None")
    elif not name:
        name = prefix
    elif prefix and name:
        name = prefix + "." + name

    if index is not None:
        return "%s[%r]" % (name, index)
    else:
        return name


def _autocast(x, dtype):
    """
    If appropriate perform type casting of value x to type dtype,
    otherwise return the original value x
    """
    if (dtype == float or (isinstance(dtype, tuple)
                           and float in dtype and int not in dtype)):
        try:
            if x == float(x):
                x = float(x)
        except ValueError:
            pass
    if (dtype == int or (isinstance(dtype, tuple)
                         and int in dtype)):
        try:
            if x == int(x):
                x = int(x)
        except ValueError:
            pass
    if isinstance(x, str):
        for type in (int, float, bool):
            if dtype == type or (isinstance(dtype, tuple) and type in dtype):
                if type == bool and x in ("True", "False"):
                    return x == "True"
                elif type != bool:
                    try:
                        return type(x)
                    except ValueError:  # Carry on and try a different coercion
                        pass
    return x


def _typeStr(x):
    """
    Utility function to generate a fully qualified type name.

    This is used primarily in writing config files to be
    executed later upon 'load'.
    """
    if hasattr(x, '__module__') and hasattr(x, '__name__'):
        xtype = x
    else:
        xtype = type(x)
    if (sys.version_info.major <= 2 and xtype.__module__ == '__builtin__') or xtype.__module__ == 'builtins':
        return xtype.__name__
    else:
        return "%s.%s" % (xtype.__module__, xtype.__name__)


class ConfigMeta(type):
    """
    A metaclass for Config

    Adds a dictionary containing all Field class attributes
    as a class attribute called '_fields', and adds the name of each field as
    an instance variable of the field itself (so you don't have to pass the
    name of the field to the field constructor).
    """
    def __init__(self, name, bases, dict_):
        type.__init__(self, name, bases, dict_)
        self._fields = OrderedDict()
        self._source = getStackFrame()

        def getFields(classtype):
            fields = OrderedDict()
            bases = list(classtype.__bases__)
            bases.reverse()
            for b in bases:
                fields.update(getFields(b))

            field_dict = {k: v for k, v in classtype.__dict__.items() if isinstance(v, Field)}
            for k, v in sorted(field_dict.items(), key=lambda x: x[1]._creation_order):
                fields[k] = v
            return fields

        fields = getFields(self)
        for k, v in fields.items():
            setattr(self, k, copy.deepcopy(v))

    def __setattr__(self, name, value):
        if isinstance(value, Field):
            value.name = name
            self._fields[name] = value
        type.__setattr__(self, name, value)


class FieldValidationError(ValueError):
    """
    Custom exception class which holds additional information useful to
    debuggin Config errors:
    - fieldType: type of the Field which incurred the error
    - fieldName: name of the Field which incurred the error
    - fullname: fully qualified name of the Field instance
    - history: full history of all changes to the Field instance
    - fieldSource: file and line number of the Field definition
    """
    def __init__(self, field, config, msg):
        self.fieldType = type(field)
        self.fieldName = field.name
        self.fullname = _joinNamePath(config._name, field.name)
        self.history = config.history.setdefault(field.name, [])
        self.fieldSource = field.source
        self.configSource = config._source
        #error = "%s '%s' failed validation: %s\n"\
        #        "For more information read the Field definition at:\n%s"\
        #        "And the Config definition at:\n%s" % \
        #    (self.fieldType.__name__, self.fullname, msg,
        #     self.fieldSource.format(), self.configSource.format())
        error = ("{} '{}' ({}) failed validation: {}".
                 format(self.fieldType.__name__, self.fullname, field.doc, msg))
        ValueError.__init__(self, error)


class Field(object):
    """A field in a a Config.

    Instances of Field should be class attributes of Config subclasses:
    Field only supports basic data types (int, float, complex, bool, str)

    class Example(Config):
        myInt = Field(int, "an integer field!", default=0)
    """
    # Must be able to support str and future str as we can not guarantee that
    # code will pass in a future str type on Python 2
    supportedTypes = set((str, bool, float, int, complex, tuple, AstroData))
    _counter = itertools.count()

    def __init__(self, doc, dtype, default=None, check=None, optional=False):
        """Initialize a Field.

        dtype ------ Data type for the field.
        doc -------- Documentation for the field.
        default ---- A default value for the field.
        check ------ A callable to be called with the field value that returns
                     False if the value is invalid.  More complex inter-field
                     validation can be written as part of Config validate()
                     method; this will be ignored if set to None.
        optional --- When False, Config validate() will fail if value is None
        """
        if isinstance(dtype, list):
            dtype = tuple(dtype)
        if isinstance(dtype, tuple):
            if any([x not in self.supportedTypes for x in dtype]):
                raise ValueError("Unsupported Field dtype in %s" % repr(dtype))
        elif dtype not in self.supportedTypes:
            raise ValueError("Unsupported Field dtype %s" % _typeStr(dtype))

        source = getStackFrame()
        self._setup(doc=doc, dtype=dtype, default=default, check=check, optional=optional, source=source)

    def _setup(self, doc, dtype, default, check, optional, source):
        """
        Convenience function provided to simplify initialization of derived
        Field types
        """
        self.dtype = dtype
        self.doc = doc
        self.__doc__ = doc
        self.default = default
        self.check = check
        self.optional = optional
        self.source = source
        self._creation_order = next(Field._counter)

    def rename(self, instance):
        """
        Rename an instance of this field, not the field itself.
        This is invoked by the owning config object and should not be called
        directly

        Only useful for fields which hold sub-configs.
        Fields which hold subconfigs should rename each sub-config with
        the full field name as generated by _joinNamePath
        """
        pass

    def validate(self, instance):
        """
        Base validation for any field.
        Ensures that non-optional fields are not None.
        Ensures type correctness
        Ensures that user-provided check function is valid
        Most derived Field types should call Field.validate if they choose
        to re-implement validate
        """
        value = self.__get__(instance)
        if not self.optional and value is None:
            raise FieldValidationError(self, instance, "Required value cannot be None")

    def freeze(self, instance):
        """
        Make this field read-only.
        Only important for fields which hold sub-configs.
        Fields which hold subconfigs should freeze each sub-config.
        """
        pass

    def _validateValue(self, value):
        """
        Validate a value that is not None

        This is called from __set__
        This is not part of the Field API. However, simple derived field types
            may benifit from implementing _validateValue
        """
        if value is None:
            return

        if not isinstance(value, self.dtype):
            if isinstance(self.dtype, tuple):
                msg = "Value %s is of incorrect type %s. Expected types %s" % \
                  ( value, _typeStr(value), [_typeStr(dt) for dt in self.dtype])
            else:
                msg = "Value %s is of incorrect type %s. Expected type %s" % \
                    (value, _typeStr(value), _typeStr(self.dtype))
            raise TypeError(msg)
        if self.check is not None and not self.check(value):
            msg = "Value %s is not a valid value" % str(value)
            raise ValueError(msg)

    def save(self, outfile, instance):
        """
        Saves an instance of this field to file.
        This is invoked by the owning config object, and should not be called
        directly

        outfile ---- an open output stream.
        """
        value = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)

        # write full documentation string as comment lines (i.e. first character is #)
        doc = "# " + str(self.doc).replace("\n", "\n# ")
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            # non-finite numbers need special care
            outfile.write(u"{}\n{}=float('{!r}')\n\n".format(doc, fullname, value))
        else:
            outfile.write(u"{}\n{}={!r}\n\n".format(doc, fullname, value))

    def toDict(self, instance):
        """
        Convert the field value so that it can be set as the value of an item
        in a dict.
        This is invoked by the owning config object and should not be called
        directly

        Simple values are passed through. Complex data structures must be
        manipulated. For example, a field holding a sub-config should, instead
        of the subconfig object, return a dict where the keys are the field
        names in the subconfig, and the values are the field values in the
        subconfig.
        """
        return self.__get__(instance)

    def __get__(self, instance, owner=None, at=None, label="default"):
        """
        Define how attribute access should occur on the Config instance
        This is invoked by the owning config object and should not be called
        directly

        When the field attribute is accessed on a Config class object, it
        returns the field object itself in order to allow inspection of
        Config classes.

        When the field attribute is access on a config instance, the actual
        value described by the field (and held by the Config instance) is
        returned.
        """
        if instance is None or not isinstance(instance, Config):
            return self
        else:
            if self.name in instance:
                return instance._storage[self.name]
            else:
                raise AttributeError("'{}' object has no attribute '{}'".
                                     format(instance.__class__.__name__, self.name))

    def __set__(self, instance, value, at=None, label='assignment'):
        """
        Describe how attribute setting should occur on the config instance.
        This is invoked by the owning config object and should not be called
        directly

        Derived Field classes may need to override the behavior. When overriding
        __set__, Field authors should follow the following rules:
        * Do not allow modification of frozen configs
        * Validate the new value *BEFORE* modifying the field. Except if the
            new value is None. None is special and no attempt should be made to
            validate it until Config.validate is called.
        * Do not modify the Config instance to contain invalid values.
        * If the field is modified, update the history of the field to reflect the
            changes

        In order to decrease the need to implement this method in derived Field
        types, value validation is performed in the method _validateValue. If
        only the validation step differs in the derived Field, it is simpler to
        implement _validateValue than to re-implement __set__. More complicated
        behavior, however, may require a reimplementation.
        """
        if instance._frozen:
            raise FieldValidationError(self, instance, "Cannot modify a frozen Config")

        if at is None:
            at = getCallStack()
        # setDefaults() gets a free pass due to our mashing of inheritance
        if self.name not in instance._fields:
            #if any('setDefaults' in stk.function for stk in at):
            #    return
            raise AttributeError("{} has no attribute {}".format(instance.__class__.__name__, self.name))

        history = instance._history.setdefault(self.name, [])
        if value is not None:
            value = _autocast(value, self.dtype)
            try:
                self._validateValue(value)
            except BaseException as e:
                raise FieldValidationError(self, instance, str(e))

        instance._storage[self.name] = value
        # We don't want to put an actual AD object here, so just the filename
        value_to_append = value.filename if isinstance(value, AstroData) else value
        history.append((value_to_append, at, label))

    def __delete__(self, instance, at=None, label='deletion'):
        """
        Describe how attribute deletion should occur on the Config instance.
        This is invoked by the owning config object and should not be called
        directly
        """
        if at is None:
            at = getCallStack()
        self.__set__(instance, None, at=at, label=label)

    def _compare(self, instance1, instance2, shortcut, rtol, atol, output):
        """Helper function for Config.compare; used to compare two fields for equality.

        Must be overridden by more complex field types.

        @param[in] instance1  LHS Config instance to compare.
        @param[in] instance2  RHS Config instance to compare.
        @param[in] shortcut   If True, return as soon as an inequality is found.
        @param[in] rtol       Relative tolerance for floating point comparisons.
        @param[in] atol       Absolute tolerance for floating point comparisons.
        @param[in] output     If not None, a callable that takes a string, used (possibly repeatedly)
                              to report inequalities.

        Floating point comparisons are performed by numpy.allclose; refer to that for details.
        """
        v1 = getattr(instance1, self.name)
        v2 = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        return compareScalars(name, v1, v2, dtype=self.dtype, rtol=rtol, atol=atol, output=output)


class RecordingImporter(object):
    """An Importer (for sys.meta_path) that records which modules are being imported.

    Objects also act as Context Managers, so you can:
        with RecordingImporter() as importer:
            import stuff
        print("Imported: " + importer.getModules())
    This ensures it is properly uninstalled when done.

    This class makes no effort to do any importing itself.
    """
    def __init__(self):
        """Create and install the Importer"""
        self._modules = set()

    def __enter__(self):

        self.origMetaPath = sys.meta_path
        sys.meta_path = [self] + sys.meta_path
        return self

    def __exit__(self, *args):
        self.uninstall()
        return False  # Don't suppress exceptions

    def uninstall(self):
        """Uninstall the Importer"""
        sys.meta_path = self.origMetaPath

    def find_module(self, fullname, path=None):
        """Called as part of the 'import' chain of events.

        We return None because we don't do any importing.
        """
        self._modules.add(fullname)
        return None

    def getModules(self):
        """Return the set of modules that were imported."""
        return self._modules


class Config(with_metaclass(ConfigMeta, object)):
    """Base class for control objects.

    A Config object will usually have several Field instances as class
    attributes; these are used to define most of the base class behavior.
    Simple derived class should be able to be defined simply by setting those
    attributes.

    Config also emulates a dict of field name: field value

    CJS: Edited these so only the _fields are exposed. _storage retains
    items that have been deleted
    """

    def __iter__(self):
        """!Iterate over fields
        """
        return self._fields.__iter__()

    def keys(self):
        """!Return the list of field names
        """
        return list(self._fields)

    def values(self):
        """!Return the list of field values
        """
        return self.toDict().values()

    def items(self):
        """!Return the list of (field name, field value) pairs
        """
        return self.toDict().items()

    def iteritems(self):
        """!Iterate over (field name, field value) pairs
        """
        return self.toDict().iteritems()

    def itervalues(self):
        """!Iterate over field values
        """
        return self.toDict().itervalues()

    def iterkeys(self):
        """!Iterate over field names
        """
        return self.toDict().iterkeys()

    def iterfields(self):
        """!Iterate over field objects
        """
        return iter(self._fields.values())

    def doc(self, field):
        """Return docstring for field
        """
        return self._fields[field].doc

    def __contains__(self, name):
        """!Return True if the specified field exists in this config

        @param[in] name  field name to test for
        """
        return self._storage.__contains__(name) and name in self._fields

    def __new__(cls, *args, **kw):
        """!Allocate a new Config object.

        In order to ensure that all Config object are always in a proper
        state when handed to users or to derived Config classes, some
        attributes are handled at allocation time rather than at initialization

        This ensures that even if a derived Config class implements __init__,
        the author does not need to be concerned about when or even if he
        should call the base Config.__init__
        """
        name = kw.pop("__name", None)
        at = kw.pop("__at", getCallStack())
        # remove __label and ignore it
        kw.pop("__label", "default")

        instance = object.__new__(cls)
        instance._frozen = False
        instance._name = name
        instance._storage = {}
        instance._history = {}
        instance._imports = set()
        # load up defaults
        instance.reset(at=at)
        # set custom default-overides
        instance.setDefaults()
        # set constructor overides
        instance.update(__at=at, **kw)
        return instance

    def __reduce__(self):
        """Reduction for pickling (function with arguments to reproduce).

        We need to condense and reconstitute the Config, since it may contain lambdas
        (as the 'check' elements) that cannot be pickled.
        """
        # The stream must be in characters to match the API but pickle requires bytes
        stream = io.StringIO()
        self.saveToStream(stream)
        return (unreduceConfig, (self.__class__, stream.getvalue().encode()))

    def reset(self, at=None):
        """Reset all values to their defaults"""
        if at is None:
            at = getCallStack()
        for field in self._fields.values():
            self._history[field.name] = []
            field.__set__(self, field.default, at=at + [field.source], label="default")

    def setDefaults(self):
        """
        Derived config classes that must compute defaults rather than using the
        Field defaults should do so here.
        To correctly use inherited defaults, implementations of setDefaults()
        must call their base class' setDefaults()
        """
        pass

    def update(self, **kw):
        """!Update values specified by the keyword arguments

        @warning The '__at' and '__label' keyword arguments are special internal
        keywords. They are used to strip out any internal steps from the
        history tracebacks of the config. Modifying these keywords allows users
        to lie about a Config's history. Please do not do so!
        """
        at = kw.pop("__at", getCallStack())
        label = kw.pop("__label", "update")

        for name, value in kw.items():
            try:
                field = self._fields[name]
                field.__set__(self, value, at=at, label=label)
            except KeyError:
                #raise KeyError("No field of name %s exists in config type %s" % (name, _typeStr(self)))
                raise KeyError("{} has no field named {}".format(type(self).__name__.replace('Config', ''), name))

    def load(self, filename, root="config"):
        """!Modify this config in place by executing the Python code in the named file.

        @param[in] filename  name of file containing config override code
        @param[in] root  name of variable in file that refers to the config being overridden

        For example: if the value of root is "config" and the file contains this text:
        "config.myField = 5" then this config's field "myField" is set to 5.

        @deprecated For purposes of backwards compatibility, older config files that use
        root="root" instead of root="config" will be loaded with a warning printed to sys.stderr.
        This feature will be removed at some point.
        """
        with open(filename, "r") as f:
            code = compile(f.read(), filename=filename, mode="exec")
            self.loadFromStream(stream=code, root=root)

    def loadFromStream(self, stream, root="config", filename=None):
        """!Modify this config in place by executing the python code in the provided stream.

        @param[in] stream  open file object, string or compiled string containing config override code
        @param[in] root  name of variable in stream that refers to the config being overridden
        @param[in] filename  name of config override file, or None if unknown or contained
            in the stream; used for error reporting

        For example: if the value of root is "config" and the stream contains this text:
        "config.myField = 5" then this config's field "myField" is set to 5.

        @deprecated For purposes of backwards compatibility, older config files that use
        root="root" instead of root="config" will be loaded with a warning printed to sys.stderr.
        This feature will be removed at some point.
        """
        with RecordingImporter() as importer:
            try:
                local = {root: self}
                exec(stream, {}, local)
            except NameError as e:
                if root == "config" and "root" in e.args[0]:
                    if filename is None:
                        # try to determine the file name; a compiled string has attribute "co_filename",
                        # an open file has attribute "name", else give up
                        filename = getattr(stream, "co_filename", None)
                        if filename is None:
                            filename = getattr(stream, "name", "?")
                    sys.stderr.write(u"Config override file %r" % (filename,) +
                                     u" appears to use 'root' instead of 'config'; trying with 'root'")
                    local = {"root": self}
                    exec(stream, {}, local)
                else:
                    raise

        self._imports.update(importer.getModules())

    def save(self, filename, root="config"):
        """!Save a python script to the named file, which, when loaded, reproduces this Config

        @param[in] filename  name of file to which to write the config
        @param[in] root  name to use for the root config variable; the same value must be used when loading
        """
        d = os.path.dirname(filename)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=d) as outfile:
            self.saveToStream(outfile, root)
            # tempfile is hardcoded to create files with mode '0600'
            # for an explantion of these antics see:
            # https://stackoverflow.com/questions/10291131/how-to-use-os-umask-in-python
            umask = os.umask(0o077)
            os.umask(umask)
            os.chmod(outfile.name, (~umask & 0o666))
            # chmod before the move so we get quasi-atomic behavior if the
            # source and dest. are on the same filesystem.
            # os.rename may not work across filesystems
            shutil.move(outfile.name, filename)

    def saveToStream(self, outfile, root="config"):
        """!Save a python script to a stream, which, when loaded, reproduces this Config

        @param outfile [inout] open file object to which to write the config. Accepts strings not bytes.
        @param root [in] name to use for the root config variable; the same value must be used when loading
        """
        tmp = self._name
        self._rename(root)
        try:
            configType = type(self)
            typeString = _typeStr(configType)
            outfile.write(u"import {}\n".format(configType.__module__))
            outfile.write(u"assert type({})=={}, 'config is of type %s.%s ".format(root, typeString))
            outfile.write(u"instead of {}' % (type({}).__module__, type({}).__name__)\n".format(typeString,
                                                                                                root,
                                                                                                root))
            self._save(outfile)
        finally:
            self._rename(tmp)

    def freeze(self):
        """!Make this Config and all sub-configs read-only
        """
        self._frozen = True
        for field in self._fields.values():
            field.freeze(self)

    def _save(self, outfile):
        """!Save this Config to an open stream object
        """
        for imp in self._imports:
            if imp in sys.modules and sys.modules[imp] is not None:
                outfile.write(u"import {}\n".format(imp))
        for field in self._fields.values():
            field.save(outfile, self)

    def toDict(self):
        """!Return a dict of field name: value

        Correct behavior is dependent on proper implementation of  Field.toDict. If implementing a new
        Field type, you may need to implement your own toDict method.
        """
        dict_ = OrderedDict()
        for name, field in self._fields.items():
            dict_[name] = field.toDict(self)
        return dict_

    def _rename(self, name):
        """!Rename this Config object in its parent config

        @param[in] name  new name for this config in its parent config

        Correct behavior is dependent on proper implementation of Field.rename. If implementing a new
        Field type, you may need to implement your own rename method.
        """
        self._name = name
        for field in self._fields.values():
            field.rename(self)

    def validate(self):
        """!Validate the Config; raise an exception if invalid

        The base class implementation performs type checks on all fields by
        calling Field.validate().

        Complex single-field validation can be defined by deriving new Field
        types. As syntactic sugar, some derived Field types are defined in
        this module which handle recursing into sub-configs
        (ConfigField, ConfigChoiceField)

        Inter-field relationships should only be checked in derived Config
        classes after calling this method, and base validation is complete
        """
        for field in self._fields.values():
            field.validate(self)

    def formatHistory(self, name=None, **kwargs):
        """!Format the specified config field's history to a more human-readable format

        @param[in] name  name of field whose history is wanted
        @param[in] kwargs  keyword arguments for lsst.pex.config.history.format
        @return a string containing the formatted history
        """
        from . import history as pexHist
        return pexHist.format(self, name, **kwargs)

    """
    Read-only history property
    """
    history = property(lambda x: x._history)

    def __setattr__(self, attr, value, at=None, label="assignment"):
        """!Regulate which attributes can be set

        Unlike normal python objects, Config objects are locked such
        that no additional attributes nor properties may be added to them
        dynamically.

        Although this is not the standard Python behavior, it helps to
        protect users from accidentally mispelling a field name, or
        trying to set a non-existent field.
        """
        if attr in self._fields:
            if at is None:
                at = getCallStack()
            # This allows Field descriptors to work.
            self._fields[attr].__set__(self, value, at=at, label=label)
        elif hasattr(getattr(self.__class__, attr, None), '__set__'):
            # This allows properties and other non-Field descriptors to work.
            return object.__setattr__(self, attr, value)
        elif attr in self.__dict__ or attr in ("_name", "_history", "_storage", "_frozen", "_imports"):
            # This allows specific private attributes to work.
            self.__dict__[attr] = value
        else:
            # We throw everything else.
            raise AttributeError("%s has no attribute %s" % (_typeStr(self), attr))

    def __delattr__(self, attr, at=None, label="deletion"):
        # CJS: Hacked to allow setDefaults() to delete non-existent fields
        if at is None:
            at = getCallStack()
        if attr in self._fields:
            #self._fields[attr].__delete__(self, at=at, label=label)
            del self._fields[attr]
        elif not any(stk.function== 'setDefaults' for stk in at):
            object.__delattr__(self, attr)

    def __eq__(self, other):
        if type(other) == type(self):
            for name in self._fields:
                thisValue = getattr(self, name)
                otherValue = getattr(other, name)
                if isinstance(thisValue, float) and math.isnan(thisValue):
                    if not math.isnan(otherValue):
                        return False
                elif thisValue != otherValue:
                    return False
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.toDict())

    def __repr__(self):
        return "%s(%s)" % (
            _typeStr(self),
            ", ".join("%s=%r" % (k, v) for k, v in self.toDict().items() if v is not None)
        )

    def compare(self, other, shortcut=True, rtol=1E-8, atol=1E-8, output=None):
        """!Compare two Configs for equality; return True if equal

        If the Configs contain RegistryFields or ConfigChoiceFields, unselected Configs
        will not be compared.

        @param[in] other      Config object to compare with self.
        @param[in] shortcut   If True, return as soon as an inequality is found.
        @param[in] rtol       Relative tolerance for floating point comparisons.
        @param[in] atol       Absolute tolerance for floating point comparisons.
        @param[in] output     If not None, a callable that takes a string, used (possibly repeatedly)
                              to report inequalities.

        Floating point comparisons are performed by numpy.allclose; refer to that for details.
        """
        name1 = self._name if self._name is not None else "config"
        name2 = other._name if other._name is not None else "config"
        name = getComparisonName(name1, name2)
        return compareConfigs(name, self, other, shortcut=shortcut,
                              rtol=rtol, atol=atol, output=output)


def unreduceConfig(cls, stream):
    config = cls()
    config.loadFromStream(stream)
    return config
