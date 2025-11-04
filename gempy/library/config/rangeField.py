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
from .config import Field, _typeStr
from .callStack import getStackFrame


__all__ = ["RangeField"]


class RangeField(Field):
    """
    Defines a Config Field which allows only a range of values.
    The range is defined by providing min and/or max values.
    If min or max is None, the range will be open in that direction
    If inclusive[Min|Max] is True the range will include the [min|max] value

    """

    supportedTypes = set((int, float))

    def __init__(self, doc, dtype, default=None, optional=False,
                 min=None, max=None, inclusiveMin=True, inclusiveMax=False,
                 *, fix_start_to_min=False, fix_end_to_max=False):
        """Initialize a RangeField object.

        Parameters
        ----------
        doc : str
            The documentation string for the RangeField object.
        dtype : type
            The data type of the RangeField.
        default : optional
            The default value for the RangeField (default: None).
        optional : bool
            Whether the RangeField is optional (default: False).
        min : optional
            The minimum value of the RangeField (default: None).
        max : optional
            The maximum value of the RangeField (default: None).
        inclusiveMin : bool
            Whether the minimum value is inclusive (default: True).
        inclusiveMax : bool
            Whether the maximum value is inclusive (default: False).
        fix_start_to_min : bool
            Whether to fix the start value to the minimum value (default: False).
        fix_end_to_max : bool
            Whether to fix the end value to the maximum value (default: False).

        Raises
        ------
        ValueError
            If the dtype is not supported or if both min and max are None.
        ValueError
            If min is greater than max or if min equals max and both are not inclusive.

        """
        if dtype not in self.supportedTypes:
            raise ValueError("Unsupported RangeField dtype %s" % (_typeStr(dtype)))
        source = getStackFrame()
        if min is None and max is None:
            raise ValueError("min and max cannot both be None")

        if min is not None and max is not None:
            if min > max:
                raise ValueError("min = %s > %s = max" % (min, max))
            elif min == max and not (inclusiveMin and inclusiveMax):
                raise ValueError("min = max = %s and min and max not both inclusive" % (min,))

        self.min = min
        self.max = max
        self.fix_start_to_min = fix_start_to_min
        self.fix_end_to_max = fix_end_to_max

        self.rangeString = "%s%s,%s%s" % \
            (("[" if inclusiveMin else "("),
             ("-inf" if self.min is None else self.min),
             ("inf" if self.max is None else self.max),
             ("]" if inclusiveMax else ")"))
        doc += "\n\tValid Range = " + self.rangeString
        if inclusiveMax:
            self.maxCheck = lambda x, y: True if y is None else x <= y
        else:
            self.maxCheck = lambda x, y: True if y is None else x < y
        if inclusiveMin:
            self.minCheck = lambda x, y: True if y is None else x >= y
        else:
            self.minCheck = lambda x, y: True if y is None else x > y
        self._setup(doc, dtype=dtype, default=default, check=None, optional=optional, source=source)

    def _validateValue(self, value):
        Field._validateValue(self, value)
        if not self.minCheck(value, self.min) or \
                not self.maxCheck(value, self.max):
            msg = "%s is outside of valid range %s" % (value, self.rangeString)
            raise ValueError(msg)
