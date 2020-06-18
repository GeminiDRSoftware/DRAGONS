import inspect
import warnings
from collections import namedtuple
from functools import wraps
from traceback import format_stack

import numpy as np

INTEGER_TYPES = (int, np.integer)

__all__ = ('assign_only_single_slice', 'astro_data_descriptor',
           'AstroDataFitsDeprecationWarning', 'astro_data_tag', 'deprecated',
           'normalize_indices', 'returns_list', 'TagSet')


# FIXME: Should be AstroDataDeprecationWarning ?
class AstroDataFitsDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("always", AstroDataFitsDeprecationWarning)


def deprecated(reason):
    def decorator_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kw):
            current_source = '|'.join(format_stack(inspect.currentframe()))
            if current_source not in wrapper.seen:
                wrapper.seen.add(current_source)
                warnings.warn(reason, AstroDataFitsDeprecationWarning)
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
    add : set of str, or None
        Tags to be added to the global set
    remove : set of str, or None
        Tags to be removed from the global set
    blocked_by : set of str, or None
        Tags that will prevent this TagSet from being applied
    blocks : set of str, or None
        Other TagSets containing these won't be applied
    if_present : set of str, or None
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
