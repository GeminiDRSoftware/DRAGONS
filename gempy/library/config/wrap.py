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
import inspect
import re
import importlib

from .config import Config, Field
from .listField import ListField, List
from .configField import ConfigField
from .callStack import getCallerFrame, getCallStack

__all__ = ("wrap", "makeConfigClass")

# Mapping from C++ types to Python type: assumes we can round-trip between these using
# the usual pybind11 converters, but doesn't require they be binary equivalent under-the-hood
# or anything.
_dtypeMap = {
    "bool": bool,
    "int": int,
    "double": float,
    "float": float,
    "std::int64_t": int,
    "std::string": str
}

_containerRegex = re.compile(r"(std::)?(vector|list)<\s*(?P<type>[a-z0-9_:]+)\s*>")


def makeConfigClass(ctrl, name=None, base=Config, doc=None, module=0, cls=None):
    """A function that creates a Python config class that matches a  C++ control object class.

    @param ctrl        C++ control class to wrap.
    @param name        Name of the new config class; defaults to the __name__ of the control
                       class with 'Control' replaced with 'Config'.
    @param base        Base class for the config class.
    @param doc         Docstring for the config class.
    @param module      Either a module object, a string specifying the name of the module, or an
                       integer specifying how far back in the stack to look for the module to use:
                       0 is the immediate caller of pex.config.wrap.  This will be used to
                       set __module__ for the new config class, and the class will also be added
                       to the module.  Ignored if None or if cls is not None, but note that the default
                       is to use the callers' module.
    @param cls         An existing config class to use instead of creating a new one; name, base
                       doc, and module will be ignored if this is not None.

    See the 'wrap' decorator as a way to use makeConfigClass that may be more convenient.

    To use makeConfigClass, in C++, write a control object, using the LSST_CONTROL_FIELD macro in
    lsst/pex/config.h (note that it must have sensible default constructor):

    @code
    // myHeader.h

    struct InnerControl {
        LSST_CONTROL_FIELD(wim, std::string, "documentation for field 'wim'");
    };

    struct FooControl {
        LSST_CONTROL_FIELD(bar, int, "documentation for field 'bar'");
        LSST_CONTROL_FIELD(baz, double, "documentation for field 'baz'");
        LSST_NESTED_CONTROL_FIELD(zot, myWrappedLib, InnerControl, "documentation for field 'zot'");

        FooControl() : bar(0), baz(0.0) {}
    };
    @endcode


    You can use LSST_NESTED_CONTROL_FIELD to nest control objects.  Now, wrap those control objects as
    you would any other C++ class, but make sure you include lsst/pex/config.h before including the header
    file where the control object class is defined:

    Now, in Python, do this:

    @code
    import myWrappedLib
    import lsst.pex.config
    InnerConfig = lsst.pex.config.makeConfigClass(myWrappedLib.InnerControl)
    FooConfig = lsst.pex.config.makeConfigClass(myWrappedLib.FooControl)
    @endcode

    This will add fully-fledged "bar", "baz", and "zot" fields to FooConfig, set
    FooConfig.Control = FooControl, and inject makeControl and readControl
    methods to create a FooControl and set the FooConfig from the FooControl,
    respectively.  In addition, if FooControl has a validate() member function,
    a custom validate() method will be added to FooConfig that uses it.   And,
    of course, all of the above will be done for InnerControl/InnerConfig too.

    Any field that would be injected that would clash with an existing attribute of the
    class will be silently ignored; this allows the user to customize fields and
    inherit them from wrapped control classes.  However, these names will still be
    processed when converting between config and control classes, so they should generally
    be present as base class fields or other instance attributes or descriptors.

    While LSST_CONTROL_FIELD will work for any C++ type, automatic Config generation
    only supports bool, int, std::int64_t, double, and std::string  fields, along
    with std::list and std::vectors of those types.
    """
    if name is None:
        if "Control" not in ctrl.__name__:
            raise ValueError("Cannot guess appropriate Config class name for %s." % ctrl)
        name = ctrl.__name__.replace("Control", "Config")
    if cls is None:
        cls = type(name, (base,), {"__doc__": doc})
        if module is not None:
            # Not only does setting __module__ make Python pretty-printers more useful,
            # it's also necessary if we want to pickle Config objects.
            if isinstance(module, int):
                frame = getCallerFrame(module)
                moduleObj = inspect.getmodule(frame)
                moduleName = moduleObj.__name__
            elif isinstance(module, str):
                moduleName = module
                moduleObj = __import__(moduleName)
            else:
                moduleObj = module
                moduleName = moduleObj.__name__
            cls.__module__ = moduleName
            setattr(moduleObj, name, cls)
    if doc is None:
        doc = ctrl.__doc__
    fields = {}
    # loop over all class attributes, looking for the special static methods that indicate a field
    # defined by one of the macros in pex/config.h.
    for attr in dir(ctrl):
        if attr.startswith("_type_"):
            k = attr[len("_type_"):]
            getDoc = "_doc_" + k
            getModule = "_module_" + k
            getType = attr
            if hasattr(ctrl, k) and hasattr(ctrl, getDoc):
                doc = getattr(ctrl, getDoc)()
                ctype = getattr(ctrl, getType)()
                if hasattr(ctrl, getModule):  # if this is present, it's a nested control object
                    nestedModuleName = getattr(ctrl, getModule)()
                    if nestedModuleName == moduleName:
                        nestedModuleObj = moduleObj
                    else:
                        nestedModuleObj = importlib.import_module(nestedModuleName)
                    try:
                        dtype = getattr(nestedModuleObj, ctype).ConfigClass
                    except AttributeError:
                        raise AttributeError("'%s.%s.ConfigClass' does not exist" % (moduleName, ctype))
                    fields[k] = ConfigField(doc=doc, dtype=dtype)
                else:
                    try:
                        dtype = _dtypeMap[ctype]
                        FieldCls = Field
                    except KeyError:
                        dtype = None
                        m = _containerRegex.match(ctype)
                        if m:
                            dtype = _dtypeMap.get(m.group("type"), None)
                            FieldCls = ListField
                    if dtype is None:
                        raise TypeError("Could not parse field type '%s'." % ctype)
                    fields[k] = FieldCls(doc=doc, dtype=dtype, optional=True)

    # Define a number of methods to put in the new Config class.  Note that these are "closures";
    # they have access to local variables defined in the makeConfigClass function (like the fields dict).
    def makeControl(self):
        """Construct a C++ Control object from this Config object.

        Fields set to None will be ignored, and left at the values defined by the
        Control object's default constructor.
        """
        r = self.Control()
        for k, f in fields.items():
            value = getattr(self, k)
            if isinstance(f, ConfigField):
                value = value.makeControl()
            if value is not None:
                if isinstance(value, List):
                    setattr(r, k, value._list)
                else:
                    setattr(r, k, value)
        return r

    def readControl(self, control, __at=None, __label="readControl", __reset=False):
        """Read values from a C++ Control object and assign them to self's fields.

        The __at, __label, and __reset arguments are for internal use only; they are used to
        remove internal calls from the history.
        """
        if __at is None:
            __at = getCallStack()
        values = {}
        for k, f in fields.items():
            if isinstance(f, ConfigField):
                getattr(self, k).readControl(getattr(control, k),
                                             __at=__at, __label=__label, __reset=__reset)
            else:
                values[k] = getattr(control, k)
        if __reset:
            self._history = {}
        self.update(__at=__at, __label=__label, **values)

    def validate(self):
        """Validate the config object by constructing a control object and using
        a C++ validate() implementation."""
        super(cls, self).validate()
        r = self.makeControl()
        r.validate()

    def setDefaults(self):
        """Initialize the config object, using the Control objects default ctor
        to provide defaults."""
        super(cls, self).setDefaults()
        try:
            r = self.Control()
            # Indicate in the history that these values came from C++, even if we can't say which line
            self.readControl(r, __at=[(ctrl.__name__ + " C++", 0, "setDefaults", "")], __label="defaults",
                             __reset=True)
        except Exception:
            pass  # if we can't instantiate the Control, don't set defaults

    ctrl.ConfigClass = cls
    cls.Control = ctrl
    cls.makeControl = makeControl
    cls.readControl = readControl
    cls.setDefaults = setDefaults
    if hasattr(ctrl, "validate"):
        cls.validate = validate
    for k, field in fields.items():
        if not hasattr(cls, k):
            setattr(cls, k, field)
    return cls


def wrap(ctrl):
    """A decorator that adds fields from a C++ control class to a Python config class.

    Used like this:

    @wrap(MyControlClass)
    class MyConfigClass(Config):
        pass

    See makeConfigClass for more information; this is equivalent to calling makeConfigClass
    with the decorated class as the 'cls' argument.
    """
    def decorate(cls):
        return makeConfigClass(ctrl, cls=cls)
    return decorate
