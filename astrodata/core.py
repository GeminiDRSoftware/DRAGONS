from abc import abstractmethod, abstractproperty
from functools import wraps
import inspect
from types import StringTypes

def astro_data_tag(fn):
    @wraps(fn)
    def wrapper(self):
        try:
            ret = fn(self)
            if ret is not None:
                if not isinstance(ret, tuple):
                    raise TypeError("Tag function {} didn't return a tuple".format(self._meth.__name__))

                return tuple((s if isinstance(s, set) else set()) for s in ret)
        except KeyError:
            pass

        # Return empty sets for the "doesn't apply" case
        return (set(), set())

    wrapper.tag_method = True
    return wrapper

class AstroDataError(Exception):
    pass

class DataProvider(object):
    @abstractproperty
    def header(self):
        pass

    @abstractproperty
    def data(self):
        pass

    @abstractmethod
    def __getitem__(self, slice):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    def exposed(self):
        return ()

def simple_descriptor_mapping(**kw):
    def decorator(cls):
        for descriptor, descriptor_def in kw.items():
            setattr(cls, descriptor, property(descriptor_def))
        return cls
    return decorator

class AstroData(object):
    def __init__(self, provider):
        if not isinstance(provider, DataProvider):
            raise ValueError("AstroData is initialized with a DataProvider object. You may want to use ad.open('...') instead")
        self._dataprov = provider
        self._processing_tags = False

    def __process_tags(self):
        # This prevents infinite recursion
        if self._processing_tags:
            return set()
        self._processing_tags = True
        try:
            results = []
            for mname, method in inspect.getmembers(self, lambda x: hasattr(x, 'tag_method')):
                plus, minus = method()
                if plus or minus:
                    results.append((plus, minus))

            # Sort by the length of substractions...
            results = sorted(results, key=lambda x: len(x[1]), reverse=True)

            tags = set()
            removals = set()
            for plus, minus in results:
                if (plus - removals) == plus:
                    tags.update(plus)
                    removals.update(minus)
        finally:
            self._processin_tags = False

        return tags

    @property
    def tags(self):
        return self.__process_tags()

    @property
    def nddata(self):
        return self._dataprov.nddata

    @property
    def table(self):
        return self._dataprov.table

    def __getitem__(self, slicing):
        return self.__class__(self._dataprov[slicing])

    def __getattr__(self, attribute):
        if attribute in self._dataprov.exposed:
            return getattr(self._dataprov, attribute)
        else:
            clsname = self.__class__.__name__
            raise AttributeError("{!r} object has no attribute {!r}".format(clsname, attribute))

    def __contains__(self, attribute):
        return attribute in self._dataprov.exposed

    def __len__(self):
        return len(self._dataprov)

    @abstractmethod
    def info(self):
        pass
