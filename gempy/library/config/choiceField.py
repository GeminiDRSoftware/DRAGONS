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
oldStringType = str  # Need to keep hold of original str type
from builtins import str

from .config import Field, _typeStr
from .callStack import getStackFrame

__all__ = ["ChoiceField"]


class ChoiceField(Field):
    """
    Defines a Config Field which allows only a set of values
    All allowed must be of the same type.
    Allowed values should be provided as a dict of value, doc string pairs
    """
    def __init__(self, doc, dtype, allowed, default=None, optional=False):
        self.allowed = dict(allowed)
        if optional and None not in self.allowed:
            self.allowed[None] = "Field is optional"

        if len(self.allowed) == 0:
            raise ValueError("ChoiceFields must allow at least one choice")

        # Use standard string type if we are given a future str
        if dtype == str:
            dtype = oldStringType

        doc += "\nAllowed values:\n"
        for choice, choiceDoc in self.allowed.items():
            if choice is not None and not isinstance(choice, dtype):
                raise ValueError("ChoiceField's allowed choice %s is of incorrect type %s. Expected %s" %
                                 (choice, _typeStr(choice), _typeStr(dtype)))
            doc += "\t%s\t%s\n" % (str(choice), choiceDoc)

        Field.__init__(self, doc=doc, dtype=dtype, default=default,
                       check=None, optional=optional)
        self.source = getStackFrame()

    def _validateValue(self, value):
        Field._validateValue(self, value)
        if value not in self.allowed:
            msg = "Value {} is not allowed.\n" \
                "\tAllowed values: [{}]".format(value, ", ".join(str(key) for key in self.allowed))
            raise ValueError(msg)
