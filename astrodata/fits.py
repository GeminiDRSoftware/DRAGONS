from types import StringTypes
from abc import abstractmethod
from collections import defaultdict
from functools import partial

from .core import *

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

class KeywordCallableWrapper(object):
    def __init__(self, keyword, on_ext=None):
        self.kw = keyword
        self.on_ext = on_ext

    def __call__(self, adobj):
        def wrapper():
            if self.on_ext is None:
                return getattr(adobj.phu, self.kw)
            else:
                return getattr(adobj.ext(self.on_ext if self.on_ext != "*" else None), self.kw)
        return wrapper

class FitsKeywordManipulator(object):
    def __init__(self, headers, on_extensions=False):
        self.__dict__.update({
            "_headers": headers,
            "_on_ext": on_extensions
        })

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except KeyError as err:
            try:
                vals = err.values
                for n in err.missing_at:
                    vals[n] = default
                return vals
            except AttributeError:
                return default

    def get_comment(self, key):
        if self._on_ext:
            return [header.comments[key] for header in self._headers]
        else:
            return self._headers[0].comments[key]

    def set_comment(self, key, comment):
        def _inner_set_comment(header):
            try:
                header[key] = (header[key], comment)
            except KeyError:
                raise KeyError("Keyword {!r} not available".format(key))

        if self._on_ext:
            for n, header in enumerate(self._headers):
                try:
                    _inner_set_comment(header)
                except KeyError as err:
                    err.message = err.message + " at header {}".format(n)
                    raise
        else:
            _inner_set_comment(self._headers[0])

    def __getattr__(self, key):
        if self._on_ext:
            raised = False
            missing_at = []
            ret = []
            for n, header in enumerate(self._headers):
                try:
                    ret.append(header[key])
                except KeyError:
                    missing_at.append(n)
                    ret.append(None)
                    raised = True
            if raised:
                error = KeyError("The keyword couldn't be found at headers: {}".format(tuple(missing_at)))
                error.missing_at = missing_at
                error.values = ret
                raise error
            return ret
        else:
            return self._headers[0][key]

    def __setattr__(self, key, value):
        if self._on_ext:
            for header in self._headers:
                header[key] = value
        else:
            self._headers[0][key] = value

class TableManipulator(object):
    def __init__(self, tables):
        self.__dict__["_tables"] = tables

    def __getattr__(self, name):
        try:
            return self._tables[name]
        except KeyError:
            AttributeError("No such table: {!r}".format(name))

    def __setattr__(self, name, value):
        if name in self._tables:
            raise ValueError("Table {!r} has a value. Can't assign a new one".format(name))
        self._tables[name] = value

class FitsProvider(DataProvider):
    def __init__(self):
        self._sliced = False
        self._header = None
        self._nddata = None
        self._hdulist = None
        self.path = None

    def _slice(self, indices):
        if not self._sliced and not self._hdulist:
            # Force the loading of data, we may need it later
            self.nddata
        scopy = self.__class__()
        scopy._sliced = True
        scopy.path = self.path
        scopy._hdulist = [self._hdulist[0]] + [self._hdulist[n+1] for n in indices]
        scopy._header = [self._header[0]] + [self._header[n+1] for n in indices]
        if self._nddata:
            scopy._nddata = [self._nddata[n] for n in indices]

        return scopy

    def __getitem__(self, slc):
        nitems = len(self._header) - 1 # The Primary HDU does not count
        if isinstance(slc, slice):
            start, stop, step = slc.indices(nitems)
            indices = range(start, stop, step)
        else:
            if isinstance(slc, int):
                slc = (slc,)
            # Normalize negative indices...
            indices = [(x if x >= 0 else nitems + x) for x in slc]
        if any(i >= nitems for i in indices):
            raise IndexError("Index out of range")

        return self._slice(indices)

    @abstractmethod
    def _set_headers(self, hdulist):
        pass

    @abstractmethod
    def _reset_members(self, hdulist):
        pass

    @property
    def header(self):
        return self._header

    @property
    def nddata(self):
        if self._nddata is None:
            if self.path:
                self._reset_members(fits.open(self.path))
            else:
                self._reset_members(self._hdulist)

        return self._nddata

    @property
    def phu(self):
        return self.header[0]

    @property
    def phu_manipulator(self):
        return FitsKeywordManipulator(self.header[:1])

    def ext_manipulator(self, extname):
        return FitsKeywordManipulator(self.header[1:], on_extensions=True)

class RawFitsProvider(FitsProvider):
    def _set_headers(self, hdulist):
        self._header = [x.header for x in hdulist]

    def _reset_members(self, hdulist):
        self._hdulist = hdulist
        self._nddata = []
        for unit in hdulist:
            if isinstance(unit, fits.ImageHDU):
                obj = NDData(unit.data, meta={'hdu': unit.header})
                self._nddata.append(obj)

    def table_names(self):
        return ()

class ProcessedFitsProvider(FitsProvider):
    SKIP_HEADERS = set(('DQ', 'VAR', 'OBJMASK'))

    def __init__(self):
        super(ProcessedFitsProvider, self).__init__()
        self._tables = None

    @property
    def table(self):
        return TableManipulator(self._tables)

    def table_names(self):
        if not self._tables:
            # Force the loading of data
            # TODO: This should be done in a better way...
            self.nddata
        return self._tables.keys()

    def ext_manipulator(self, extname):
        if extname is not None:
            headers = [h for h in self.header[1:] if h.get('EXTNAME') == extname]

            if len(headers) == 0:
                raise KeyError("No extensions with name {!r}".format(extname))

            return FitsKeywordManipulator(headers, on_extensions=True)
        else:
            return super(ProcessedFitsProvider, self).ext_manipulator(None)

    def _slice(self, indices):
        scopy = super(ProcessedFitsProvider, self)._slice(indices)
        scopy._tables = {}
        for name, content in self._tables.items():
            if name in ('MDF', 'REFCAT'):
                scopy._tables[name] = self.content
            else:
                scopy._tables[name] = [lst[n] for n in indices]

        return scopy

    def _set_headers(self, hdulist):
        self._header = [x.header for x in hdulist
                                 if x.header.get('EXTNAME') not in self.SKIP_HEADERS]

    def _reset_members(self, hdulist):
        self._hdulist = hdulist
        def search_for_unit(name, ver):
            units = [x for x in hdulist
                       if x.header.get('EXTVER') == ver and x.header['EXTNAME'] == name]
            if units:
                return units[0]

            return None

        self._nddata = []
        self._tables = defaultdict(list)

        seen_refcat = False
        for unit in hdulist:
            header = unit.header
            if isinstance(unit, fits.PrimaryHDU):
                continue

            extname = header['EXTNAME']
            if extname in self.SKIP_HEADERS:
                continue
            elif extname == 'SCI':
                self._header.append(header)
                ver = header.get('EXTVER')
                obj = NDData(unit.data, meta={'hdu': header, 'ver': ver})
                dq = search_for_unit('DQ', ver)
                if dq:
                    obj.mask = dq
                var = search_for_unit('VAR', ver)
                if var:
                    # TODO: set the uncertainty.
                    # obj.uncertainty = VarUncertainty(unit.data)
                    pass
                objm = search_for_unit('OBJMASK', ver)
                if objm:
                    obj.meta.update({'objmask': objm})
                self._nddata.append(obj)
            elif isinstance(unit, fits.BinTableHDU):
                # REFCAT is the same, no matter how many copies. Have only one of them.
                obj = Table(unit.data, meta={'hdu': header, 'ver': header.get('EXTVER')})
                if extname in ('REFCAT', 'MDF'):
                    self._tables[extname] = obj
                else:
                    self._tables[extname].append(obj)
            else:
                raise Exception("I don't know what to do with extensions of type {}...".format(type(unit)))

class FitsLoader(FitsProvider):
    @staticmethod
    def is_prepared(hdulist):
        # Gemini raw HDUs have no EXTNAME
        return all(h.header.get('EXTNAME') is not None for h in hdulist[1:])

    @staticmethod
    def provider_for_hdulist(hdulist):
        """
        Returns an instance of the appropriate DataProvider class,
        according to the HDUList object
        """
        cls = ProcessedFitsProvider if FitsLoader.is_prepared(hdulist) else RawFitsProvider
        return cls()

    @staticmethod
    def from_path(path):
        hdulist = fits.open(path, memmap=True, do_not_scale_image_data=True)
        provider = FitsLoader.provider_for_hdulist(hdulist)
        provider.path = path
        provider._set_headers(hdulist)

        return provider

    @staticmethod
    def from_hdulist(hdulist, path=None):
        provider = FitsLoader.provider_for_hdulist(hdulist)
        provider.path = path
        provider._hdulist = hdulist
        provider._set_headers(hdulist)
        provider._reset_members(hdulist)

        return provider

@simple_descriptor_mapping(
        instrument = KeywordCallableWrapper('INSTRUME'),
        object = KeywordCallableWrapper('OBJECT'),
        telescope = KeywordCallableWrapper('TELESCOP'),
        ut_date = KeywordCallableWrapper('DATE-OBS')
        )
class AstroDataFits(AstroData):
    @staticmethod
    def _matches_data(dataprov):
        # This one is trivial. As long as we get a FITS file...
        return True

    def tables(self):
        return self._dataprov.table_names()

    @property
    def phu(self):
        return self._dataprov.phu_manipulator

    def ext(self, extname=None):
        return self._dataprov.ext_manipulator(extname)
