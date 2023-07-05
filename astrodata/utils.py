import inspect
import warnings
from collections import namedtuple
from functools import wraps
from traceback import format_stack

import numpy as np

INTEGER_TYPES = (int, np.integer)

__all__ = ('assign_only_single_slice', 'astro_data_descriptor',
           'AstroDataDeprecationWarning', 'astro_data_tag', 'deprecated',
           'normalize_indices', 'returns_list', 'TagSet', 'Section')


class AstroDataDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("always", AstroDataDeprecationWarning)


def deprecated(reason):
    def decorator_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kw):
            current_source = '|'.join(format_stack(inspect.currentframe()))
            if current_source not in wrapper.seen:
                wrapper.seen.add(current_source)
                warnings.warn(reason, AstroDataDeprecationWarning)
            return fn(*args, **kw)
        wrapper.seen = set()
        return wrapper
    return decorator_wrapper


def normalize_indices(slc, nitems):
    multiple = True
    if isinstance(slc, slice):
        start, stop, step = slc.indices(nitems)
        indices = list(range(start, stop, step))
    elif (isinstance(slc, INTEGER_TYPES) or
          (isinstance(slc, tuple) and
           all(isinstance(i, INTEGER_TYPES) for i in slc))):
        if isinstance(slc, INTEGER_TYPES):
            slc = (int(slc),)   # slc's type m
            multiple = False
        else:
            multiple = True
        # Normalize negative indices...
        indices = [(x if x >= 0 else nitems + x) for x in slc]
    else:
        raise ValueError("Invalid index: {}".format(slc))

    if any(i >= nitems for i in indices):
        raise IndexError("Index out of range")

    return indices, multiple


class TagSet(namedtuple('TagSet', 'add remove blocked_by blocks if_present')):
    """
    Named tuple that is used by tag methods to return which actions should be
    performed on a tag set. All the attributes are optional, and any
    combination of them can be used, allowing to create complex tag structures.
    Read the documentation on the tag-generating algorithm if you want to
    better understand the interactions.

    The simplest TagSet, though, tends to just add tags to the global set.

    It can be initialized by position, like any other tuple (the order of the
    arguments is the one in which the attributes are listed below). It can
    also be initialized by name.

    Attributes
    ----------
    add : set of str, optional
        Tags to be added to the global set
    remove : set of str, optional
        Tags to be removed from the global set
    blocked_by : set of str, optional
        Tags that will prevent this TagSet from being applied
    blocks : set of str, optional
        Other TagSets containing these won't be applied
    if_present : set of str, optional
        This TagSet will be applied only *all* of these tags are present

    Examples
    ---------
    >>> TagSet()
    TagSet(add=set(), remove=set(), blocked_by=set(), blocks=set(), if_present=set())
    >>> TagSet({'BIAS', 'CAL'})
    TagSet(add={'BIAS', 'CAL'}, remove=set(), blocked_by=set(), blocks=set(), if_present=set())
    >>> TagSet(remove={'BIAS', 'CAL'})
    TagSet(add=set(), remove={'BIAS', 'CAL'}, blocked_by=set(), blocks=set(), if_present=set())

    """
    def __new__(cls, add=None, remove=None, blocked_by=None, blocks=None,
                if_present=None):
        return super().__new__(cls, add or set(),
                               remove or set(),
                               blocked_by or set(),
                               blocks or set(),
                               if_present or set())


def astro_data_descriptor(fn):
    """
    Decorator that will mark a class method as an AstroData descriptor.
    Useful to produce list of descriptors, for example.

    If used in combination with other decorators, this one *must* be the
    one on the top (ie. the last one applying). It doesn't modify the
    method in any other way.

    Args
    -----
    fn : method
        The method to be decorated

    Returns
    --------
    The tagged method (not a wrapper)
    """
    fn.descriptor_method = True
    return fn


def returns_list(fn):
    """
    Decorator to ensure that descriptors that should return a list (of one
    value per extension) only returns single values when operating on
    single slices; and vice versa.

    This is a common case, and you can use the decorator to simplify the
    logic of your descriptors.

    Args
    -----
    fn : method
        The method to be decorated

    Returns
    --------
    A function
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        ret = fn(self, *args, **kwargs)
        if self.is_single:
            if isinstance(ret, list):
                # TODO: log a warning if the list is >1 element
                if len(ret) > 1:
                    pass
                return ret[0]
            else:
                return ret
        else:
            if isinstance(ret, list):
                if len(ret) == len(self):
                    return ret
                else:
                    raise IndexError(
                        "Incompatible numbers of extensions and elements in {}"
                        .format(fn.__name__))
            else:
                return [ret] * len(self)
    return wrapper


def assign_only_single_slice(fn):
    """Raise `ValueError` if assigning to a non-single slice."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that "
                             "is not a single slice")
        return fn(self, *args, **kwargs)
    return wrapper


def astro_data_tag(fn):
    """
    Decorator that marks methods of an `AstroData` derived class as part of the
    tag-producing system.

    It wraps the method around a function that will ensure a consistent return
    value: the wrapped method can return any sequence of sequences of strings,
    and they will be converted to a TagSet. If the wrapped method
    returns None, it will be turned into an empty TagSet.

    Args
    -----
    fn : method
        The method to be decorated

    Returns
    --------
    A wrapper function
    """
    @wraps(fn)
    def wrapper(self):
        try:
            ret = fn(self)
            if ret is not None:
                if not isinstance(ret, TagSet):
                    raise TypeError("Tag function {} didn't return a TagSet"
                                    .format(fn.__name__))

                return TagSet(*tuple(set(s) for s in ret))
        except KeyError:
            pass

        # Return empty TagSet for the "doesn't apply" case
        return TagSet()

    wrapper.tag_method = True
    return wrapper


class Section(tuple):
    """A class to handle n-dimensional sections"""

    def __new__(cls, *args, **kwargs):
        # Ensure that the order of keys is what we want
        axis_names = [x for axis in "xyzuvw"
                      for x in (f"{axis}1", f"{axis}2")]
        _dict = {k: v for k, v in zip(axis_names, args +
                                      ('',) * len(kwargs))}
        _dict.update(kwargs)
        if list(_dict.values()).count('') or (len(_dict) % 2):
            raise ValueError("Cannot initialize 'Section' object")
        instance = tuple.__new__(cls, tuple(_dict.values()))
        instance._axis_names = tuple(_dict.keys())
        if not all(np.diff(instance)[::2] > 0):
            raise ValueError("Not all 'Section' end coordinates exceed the "
                             "start coordinates")
        return instance

    @property
    def __dict__(self):
        return dict(zip(self._axis_names, self))

    def __getnewargs__(self):
        return tuple(self)

    def __getattr__(self, attr):
        if attr in self._axis_names:
            return self.__dict__[attr]
        raise AttributeError(f"No such attribute '{attr}'")

    def __repr__(self):
        return ("Section(" +
                ", ".join([f"{k}={self.__dict__[k]}"
                           for k in self._axis_names]) + ")")

    @property
    def ndim(self):
        return len(self) // 2

    @staticmethod
    def from_shape(value):
        """produce a Section object defining a given shape"""
        return Section(*[y for x in reversed(value) for y in (0, x)])

    @staticmethod
    def from_string(value):
        """The inverse of __str__, produce a Section object from a string"""
        # if we were sent None, return None
        if value is None:
            return None
        return Section(*[y for x in value.strip("[]").split(",")
                         for start, end in [x.split(":")]
                         for y in (None if start == '' else int(start)-1,
                                   None if end == '' else int(end))])

    def asIRAFsection(self, binning=None):
        """Produce string of style '[x1:x2,y1:y2]' that is 1-indexed
        and end-inclusive

        Parameters
        ----------
        binning : iterable
            A length-2 iterable of (x_binning, y_binning). Binning is assumed
            to be 1 for all axes if not given.
        """
        if binning is None:
            binning = [1] * len(self._axis_names)
        return ("[" +
                ",".join([":".join([str(bin_*self.__dict__[axis]+1),
                                    str(bin_*self.__dict__[axis.replace("1", "2")])])
                          for axis, bin_ in zip(self._axis_names[::2], binning)])
                + "]")

    def asslice(self, add_dims=0):
        """Return the Section object as a slice/list of slices.
        Higher dimensionality can be achieved with the add_dims parameter."""
        return ((slice(None),) * add_dims +
                tuple(slice(self.__dict__[axis],
                            self.__dict__[axis.replace("1", "2")])
                      for axis in reversed(self._axis_names[::2])))

    def contains(self, section):
        """Return True if the supplied section is entirely within self"""
        if self.ndim != section.ndim:
            raise ValueError("Sections have different dimensionality")
        return (all(s2 >= s1 for s1, s2 in zip(self[::2], section[::2])) and
                all(s2 <= s1 for s1, s2 in zip(self[1::2], section[1::2])))

    def is_same_size(self, section):
        """Return True if the Sections are the same size"""
        return np.array_equal(np.diff(self)[::2], np.diff(section)[::2])

    def overlap(self, section):
        """Determine whether the two sections overlap. If so, the Section
        common to both is returned, otherwise None"""
        if self.ndim != section.ndim:
            raise ValueError("Sections have different dimensionality")
        mins = [max(s1, s2) for s1, s2 in zip(self[::2], section[::2])]
        maxs = [min(s1, s2) for s1, s2 in zip(self[1::2], section[1::2])]
        try:
            return self.__class__(*[v for pair in zip(mins, maxs) for v in pair])
        except ValueError:
            return

    def shift(self, *shifts):
        """Shift a section in each direction by the specified amount"""
        if len(shifts) != self.ndim:
            raise ValueError(f"Number of shifts {len(shifts)} incompatible "
                             f"with dimensionality {self.ndim}")
        return self.__class__(*[x + s for x, s in
                                zip(self, [ss for s in shifts for ss in [s] * 2])])
