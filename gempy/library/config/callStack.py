#
# LSST Data Management System
# Copyright 2017 AURA/LSST.
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

from __future__ import print_function, division, absolute_import

__all__ = ['getCallerFrame', 'getStackFrame', 'StackFrame', 'getCallStack']

from builtins import object

import inspect
import linecache


def getCallerFrame(relative=0):
    """Retrieve the frame for the caller

    By "caller", we mean our user's caller.

    Parameters
    ----------
    relative : `int`, non-negative
        Number of frames above the caller to retrieve.

    Returns
    -------
    frame : `__builtin__.Frame`
        Frame for the caller.
    """
    frame = inspect.currentframe().f_back.f_back  # Our caller's caller
    for ii in range(relative):
        frame = frame.f_back
    return frame


def getStackFrame(relative=0):
    """Retrieve the stack frame for the caller

    By "caller", we mean our user's caller.

    Parameters
    ----------
    relative : `int`, non-negative
        Number of frames above the caller to retrieve.

    Returns
    -------
    frame : `StackFrame`
        Stack frame for the caller.
    """
    frame = getCallerFrame(relative + 1)
    return StackFrame.fromFrame(frame)


class StackFrame(object):
    """A single element of the stack trace

    This differs slightly from the standard system mechanisms for
    getting a stack trace by the fact that it does not look up the
    source code until it is absolutely necessary, reducing the I/O.

    Parameters
    ----------
    filename : `str`
        Name of file containing the code being executed.
    lineno : `int`
        Line number of file being executed.
    function : `str`
        Function name being executed.
    content : `str` or `None`
        The actual content being executed. If not provided, it will be
        loaded from the file.
    """
    _STRIP = "/DRAGONS/"  # String to strip from the filename

    def __init__(self, filename, lineno, function, content=None):
        loc = filename.rfind(self._STRIP)
        if loc > 0:
            filename = filename[loc + len(self._STRIP):]
        self.filename = filename
        self.lineno = lineno
        self.function = function
        self._content = content

    @property
    def content(self):
        """Getter for content being executed

        Load from file on demand.
        """
        if self._content is None:
            self._content = linecache.getline(self.filename, self.lineno).strip()
        return self._content

    @classmethod
    def fromFrame(cls, frame):
        """Construct from a Frame object

        inspect.currentframe() provides a Frame object. This is
        a convenience constructor to interpret that Frame object.

        Parameters
        ----------
        frame : `Frame`
            Frame object to interpret.

        Returns
        -------
        output : `StackFrame`
            Constructed object.
        """
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        function = frame.f_code.co_name
        return cls(filename, lineno, function)

    def __repr__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__, self.filename, self.lineno, self.function)

    def format(self, full=False):
        """Format for printing

        Parameters
        ----------
        full : `bool`
            Print full details, including content being executed?

        Returns
        -------
        result : `str`
            Formatted string.
        """
        result = "  File %s:%s (%s)" % (self.filename, self.lineno, self.function)
        if full:
            result += "\n    %s" % (self.content,)
        return result


def getCallStack(skip=0):
    """Retrieve the call stack for the caller

    By "caller", we mean our user's caller - we don't include ourselves
    or our caller.

    The result is ordered with the most recent frame last.

    Parameters
    ----------
    skip : `int`, non-negative
        Number of stack frames above caller to skip.

    Returns
    -------
    output : `list` of `StackFrame`
        The call stack.
    """
    frame = getCallerFrame(skip + 1)
    stack = []
    while frame:
        stack.append(StackFrame.fromFrame(frame))
        frame = frame.f_back
    return list(reversed(stack))
