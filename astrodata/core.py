from abc import abstractmethod, abstractproperty
from types import StringTypes

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
        self._dataprov = provider

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
