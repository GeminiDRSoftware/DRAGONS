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

class KeywordCallableWrapper(object):
    def __init__(self, keyword):
        self.kw = keyword

    def __call__(self, adobj):
        return getattr(adobj.keyword, self.kw)

def descriptor_keyword_mapping(**kw):
    def decorator(cls):
        for descriptor, keyword in kw.items():
            setattr(cls, descriptor, property(KeywordCallableWrapper(keyword)))
        return cls
    return decorator

class AstroData(object):
    def __init__(self, provider):
        self._dataprov = provider
        self._kwmanip = provider.manipulator

    @property
    def keyword(self):
        return self._kwmanip

    @property
    def nddata(self):
        return self._dataprov.nddata

    @property
    def table(self):
        return self._dataprov.table

    def __getitem__(self, slicing):
        return self.__class__(self._dataprov[slicing])
