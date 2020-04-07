import inspect
import warnings
from functools import wraps
from traceback import format_stack
import numpy as np

INTEGER_TYPES = (int, np.integer)


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
