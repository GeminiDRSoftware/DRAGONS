from types import StringTypes
from abc import abstractmethod
from collections import defaultdict

from .core import *

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

class FitsKeywordManipulator(object):
    def __init__(self, headers):
        self.__dict__["_headers"] = headers

    def _select_header(self, ext):
        if ext is None:
            return self._headers[0]
        else:
            return self._headers[ext]

    def get(self, key, default=None, ext=None):
        return self._select_header(ext).get(key, default)

    def comment(self, key, ext=None):
        return self._select_header(ext).comments[key]

    def set_comment(self, key, comment, ext=None):
        h = self._select_header(ext)
        if key not in h:
            raise AttributeError("Keyword {!r} not available".format(key))
        h[key] = (h[key], comment)

    def get_all(self, key):
        found = []
        for n, h in enumerate(self._headers):
            if key in h:
                found.append((('*', n), h[key]))

        if found:
            return dict(found)
        else:
            raise KeyError("Keyword {!r} not available".format(key))

    def __getattr__(self, key):
        return self._headers[0][key]

    def __setattr__(self, key, value):
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
    def manipulator(self):
        return FitsKeywordManipulator(self.header)

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

class ProcessedFitsProvider(FitsProvider):
    SKIP_HEADERS = set(('DQ', 'VAR', 'OBJMASK'))

    def __init__(self):
        super(ProcessedFitsProvider, self).__init__()
        self._tables = None

    @property
    def table(self):
        return TableManipulator(self._tables)

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

@descriptor_keyword_mapping(
        instrument = 'INSTRUME',
        object = 'OBJECT',
        telescope = 'TELESCOP',
        ut_date = 'DATE-OBS'
        )
class AstroDataFits(AstroData):
    @staticmethod
    def _matches_data(dataprov):
        # This one is trivial. As long as we get a FITS file...
        return True
