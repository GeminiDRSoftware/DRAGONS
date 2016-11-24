from abc import ABCMeta, abstractmethod, abstractproperty
from functools import wraps
import inspect
from collections import namedtuple
from copy import deepcopy

class TagSet(namedtuple('TagSet', 'add remove blocked_by blocks if_present')):
    def __new__(cls, add=None, remove=None, blocked_by=None, blocks=None, if_present=None):
        return super(TagSet, cls).__new__(cls, add or set(),
                                               remove or set(),
                                               blocked_by or set(),
                                               blocks or set(),
                                               if_present or set())

def astro_data_descriptor(fn):
    fn.descriptor_method = True
    return fn

# Ensures descriptors coded to return a list (one value per extension)
# only return a value if sent a slice with a single extension
def returns_list(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        ret = fn(self, *args, **kwargs)
        if self._dataprov.is_single:
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
                    raise IndexError("Incompatible numbers of extensions and elements in {}".format(fn.__name__))
            else:
                return [ret] * len(self)
    return wrapper

def descriptor_list(ad):
    members = inspect.getmembers(ad, lambda x: hasattr(x, 'descriptor_method'))
    return tuple(mname for (mname, method) in members)

def astro_data_tag(fn):
    @wraps(fn)
    def wrapper(self):
        try:
            ret = fn(self)
            if ret is not None:
                if not isinstance(ret, TagSet):
                    raise TypeError("Tag function {} didn't return a TagSet".format(fn.__name__))

                return TagSet(*tuple(set(s) for s in ret))
        except KeyError:
            pass

        # Return empty TagSet for the "doesn't apply" case
        return TagSet()

    wrapper.tag_method = True
    return wrapper

class AstroDataError(Exception):
    pass

class DataProvider(object):
    __metaclass__ = ABCMeta

    @property
    def is_sliced(self):
        return False

    @property
    def is_single(self):
        return False

    @abstractproperty
    def header(self):
        pass

    @abstractmethod
    def settable(self, attribute):
        pass

    @abstractmethod
    def append(self, ext):
        pass

    @abstractmethod
    def __getitem__(self, slice):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iadd__(self, oper):
        pass

    @abstractmethod
    def __isub__(self, oper):
        pass

    @abstractmethod
    def __imul__(self, oper):
        pass

    @abstractmethod
    def __idiv__(self, oper):
        pass

    @property
    def exposed(self):
        return ()

    @abstractproperty
    def data(self):
        pass

    @abstractproperty
    def uncertainty(self):
        pass

    @abstractproperty
    def mask(self):
        pass

    @abstractproperty
    def variance(self):
        pass

def simple_descriptor_mapping(**kw):
    def decorator(cls):
        for descriptor, descriptor_def in kw.items():
            setattr(cls, descriptor, property(descriptor_def))
        return cls
    return decorator

class AstroData(object):

    # Simply a value that nobody is going to try to set an NDData attribute to
    _IGNORE = -23

    def __init__(self, provider):
        if not isinstance(provider, DataProvider):
            raise ValueError("AstroData is initialized with a DataProvider object. You may want to use ad.open('...') instead")
        self._dataprov = provider
        self._processing_tags = False

    def __deepcopy__(self, memo):
        # Force the data provider to load data, if needed
        len(self._dataprov)
        dp = deepcopy(self._dataprov, memo)
        ad = self.__class__(dp)
        return ad

    def __process_tags(self):
        try:
            # This prevents infinite recursion
            if self._processing_tags:
                return set()
            self._processing_tags = True
            try:
                results = []
                for mname, method in inspect.getmembers(self, lambda x: hasattr(x, 'tag_method')):
                    ts = method()
                    plus, minus, blocked_by, blocks, if_present = ts
                    if plus or minus or blocks:
                        results.append(ts)

                # Sort by the length of substractions... those that substract from others go first
                results = sorted(results, key=lambda x: len(x.remove) + len(x.blocks), reverse=True)
                # Sort by length of blocked_by... those that are never disabled go first
                results = sorted(results, key=lambda x: len(x.blocked_by))
                # Sort by length of if_present... those that need other tags to be present go last
                results = sorted(results, key=lambda x: len(x.if_present))

                tags = set()
                removals = set()
                blocked = set()
                for plus, minus, blocked_by, blocks, is_present in results:
                    if is_present:
                        # If this TagSet requires other tags to be present, make sure that all of
                        # them are. Otherwise, skip...
                        if len(tags & is_present) != len(is_present):
                            continue
                    allowed = (len(tags & blocked_by) + len(plus & blocked)) == 0
                    if allowed:
                        # This set is not being blocked by others...
                        removals.update(minus)
                        tags.update(plus - removals)
                        blocked.update(blocks)
            finally:
                self._processing_tags = False

            return tags
        except AttributeError as e:
            return set()

    @property
    def tags(self):
        return self.__process_tags()

    def __getitem__(self, slicing):
        return self.__class__(self._dataprov[slicing])

    def __delitem__(self, idx):
        del self._dataprov[idx]

    def __getattr__(self, attribute):
        try:
            return getattr(self._dataprov, attribute)
        except AttributeError:
            clsname = self.__class__.__name__
            raise AttributeError("{!r} object has no attribute {!r}".format(clsname, attribute))

    def __setattr__(self, attribute, value):
        if attribute != '_dataprov' and '_dataprov' in self.__dict__:
            if self._dataprov.settable(attribute):
                setattr(self._dataprov, attribute, value)
                return
        super(AstroData, self).__setattr__(attribute, value)

    def __delattr__(self, attribute):
        try:
            if self._dataprov.is_sliced:
                self._dataprov.__delattr__(attribute)
            else:
                super(AstroData, self).__delattr__(attribute)
        except AttributeError:
            if self._dataprov.is_sliced:
                raise AttributeError("{!r} sliced object has no attribute {!r}".format(self.__class__.__name__, attribute))
            else:
                raise

    def __contains__(self, attribute):
        return attribute in self._dataprov.exposed

    def __len__(self):
        return len(self._dataprov)

    @abstractmethod
    def info(self):
        pass

    def __add__(self, oper):
        copy = deepcopy(self)
        copy += oper
        return copy

    def __sub__(self, oper):
        copy = deepcopy(self)
        copy -= oper
        return copy

    def __mul__(self, oper):
        copy = deepcopy(self)
        copy *= oper
        return copy

    def __div__(self, oper):
        copy = deepcopy(self)
        copy /= oper
        return copy

    def __iadd__(self, oper):
        self._dataprov += oper
        return self

    def __isub__(self, oper):
        self._dataprov -= oper
        return self

    def __imul__(self, oper):
        self._dataprov *= oper
        return self

    def __idiv__(self, oper):
        self._dataprov /= oper
        return self

    add = __iadd__
    subtract = __isub__
    multiply = __imul__
    divide = __idiv__

    def append(self, extension):
        self._dataprov.append(extension)

    def operate(self, operator, *args, **kwargs):
        # Ensure we can iterate, even on a single slice
        for ext in [self] if self._dataprov.is_single else self:
            ext.data = operator(ext.data, *args, **kwargs)
            if ext.mask is not None:
                ext.mask = operator(ext.mask, *args, **kwargs)
            if ext.variance is not None:
                ext.variance = operator(ext.variance, *args, **kwargs)

    def reset(self, data, mask=_IGNORE, variance=_IGNORE, check=True):
        if not self._dataprov.is_single:
            raise ValueError("Trying to reset a non-sliced AstroData object")

        # In case data is an NDData object
        try:
            self.data = data
        except AttributeError:
            self.data = data.data
        # Set mask, with checking if required
        try:
            if mask.shape != self.data.shape and check:
                raise ValueError("Mask shape incompatible with data shape")
        except AttributeError:
            if mask is None:
                self.mask = mask
            elif mask == self._IGNORE:
                if hasattr(data, 'mask'):
                    self.mask = data.mask
            else:
                raise TypeError("Attempt to set mask inappropriately")
        else:
            self.mask = mask
        # Set variance, with checking if required
        try:
            if variance.shape != self.data.shape and check:
                raise ValueError("Variance shape incompatible with data shape")
        except AttributeError:
            if variance is None:
                self.uncertainty = None
            elif variance == self._IGNORE:
                if hasattr(data, 'uncertainty'):
                    self.uncertainty = data.uncertainty
            else:
                raise TypeError("Attempt to set variance inappropriately")
        else:
            self.variance = variance
