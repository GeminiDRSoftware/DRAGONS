#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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
"""
Helper functions for comparing Configs.

The function here should be use for any comparison in a Config.compare
or Field._compare implementation, as they take care of writing messages
as well as floating-point comparisons and shortcuts.
"""

import numpy

__all__ = ("getComparisonName", "compareScalars", "compareConfigs")


def getComparisonName(name1, name2):
    if name1 != name2:
        return "%s / %s" % (name1, name2)
    return name1


def compareScalars(name, v1, v2, output, rtol=1E-8, atol=1E-8, dtype=None):
    """Helper function for Config.compare; used to compare two scalar values for equality.

    @param[in] name       Name to use when reporting differences
    @param[in] dtype      Data type for comparison; may be None if it's definitely not floating-point.
    @param[in] v1         LHS value to compare
    @param[in] v2         RHS value to compare
    @param[in] output     If not None, a callable that takes a string, used (possibly repeatedly)
                          to report inequalities.
    @param[in] rtol       Relative tolerance for floating point comparisons.
    @param[in] atol       Absolute tolerance for floating point comparisons.
    @param[in] dtype      Data type for comparison; may be None if it's definitely not floating-point.

    Floating point comparisons are performed by numpy.allclose; refer to that for details.
    """
    if isinstance(dtype, tuple):
        dtype = type(v1)
    if v1 is None or v2 is None:
        result = (v1 == v2)
    elif dtype in (float, complex):
        result = numpy.allclose(v1, v2, rtol=rtol, atol=atol) or (numpy.isnan(v1) and numpy.isnan(v2))
    else:
        result = (v1 == v2)
    if not result and output is not None:
        output("Inequality in %s: %r != %r" % (name, v1, v2))
    return result


def compareConfigs(name, c1, c2, shortcut=True, rtol=1E-8, atol=1E-8, output=None):
    """Helper function for Config.compare; used to compare two Configs for equality.

    If the Configs contain RegistryFields or ConfigChoiceFields, unselected Configs
    will not be compared.

    @param[in] name       Name to use when reporting differences
    @param[in] c1         LHS config to compare
    @param[in] c2         RHS config to compare
    @param[in] shortcut   If True, return as soon as an inequality is found.
    @param[in] rtol       Relative tolerance for floating point comparisons.
    @param[in] atol       Absolute tolerance for floating point comparisons.
    @param[in] output     If not None, a callable that takes a string, used (possibly repeatedly)
                          to report inequalities.

    Floating point comparisons are performed by numpy.allclose; refer to that for details.
    """
    assert name is not None
    if c1 is None:
        if c2 is None:
            return True
        else:
            if output is not None:
                output("LHS is None for %s" % name)
            return False
    else:
        if c2 is None:
            if output is not None:
                output("RHS is None for %s" % name)
            return False
    if type(c1) != type(c1):
        if output is not None:
            output("Config types do not match for %s: %s != %s" % (name, type(c1), type(c2)))
        return False
    equal = True
    for field in c1._fields.values():
        result = field._compare(c1, c2, shortcut=shortcut, rtol=rtol, atol=atol, output=output)
        if not result and shortcut:
            return False
        equal = equal and result
    return equal
