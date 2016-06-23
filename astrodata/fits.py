from types import StringTypes
from abc import abstractmethod
from collections import defaultdict
from functools import partial

from .core import *

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

NO_DEFAULT = object()

class KeywordCallableWrapper(object):
    def __init__(self, keyword, default=NO_DEFAULT, on_ext=None):
        self.kw = keyword
        self.on_ext = on_ext
        self.default = default

    def __call__(self, adobj):
        def wrapper():
            if self.on_ext is None:
                try:
                    return getattr(adobj.phu, self.kw)
                except KeyError:
                    if self.default is NO_DEFAULT:
                        raise
                    return self.default
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

class ProcessedFitsProvider(FitsProvider):
    def __init__(self):
        super(ProcessedFitsProvider, self).__init__()
        self._tables = None
        self._exposed = []

    @property
    def exposed(self):
        return set(self._exposed)

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
        self._header = [hdulist[0].header] + [x.header for x in hdulist if x.header.get('SCI')]

    def _reset_members(self, hdulist):
        self._hdulist = hdulist
        seen = set([hdulist[0]])

        def search_for_associated(ver):
            return [x for x in hdulist
                      if x.header.get('EXTVER') == ver and x.header['EXTNAME'] != 'SCI']

        def process_meta_unit(nd, meta, add=True):
            eheader = meta.header
            name = eheader.get('EXTNAME')
            data = meta.data
            if name == 'DQ':
                nd.mask = data
            elif name == 'VAR':
                # TODO: set the uncertainty.
                # obj.uncertainty = VarUncertainty(data)
                pass
            else:
                if isinstance(meta, fits.BinTableHDU):
                    meta_obj = Table(data, meta={'hdu': eheader})
                elif isinstance(meta, fits.ImageHDU):
                    meta_obj = NDData(data, meta={'hdu': eheader})
                else:
                    raise ValueError("Unknown extension type: {!r}".format(name))
                if add:
                    setattr(nd, name, meta_obj)
                    nd.meta['other'].append(name)
                else:
                    return meta_obj

        self._nddata = []
        sci_units = [x for x in hdulist[1:] if x.header['EXTNAME'] == 'SCI']

        for unit in sci_units:
            seen.add(unit)
            header = unit.header
            ver = header['EXTVER']
            nd = NDData(unit.data, meta={'hdu': header, 'ver': ver, 'other': []})
            self._nddata.append(nd)

            for extra_unit in search_for_associated(ver):
                seen.add(extra_unit)
                process_meta_unit(nd, extra_unit)

        for other in self._hdulist:
            if other in seen:
                continue
            name = other.header['EXTNAME']
            if other.header.get('EXTVER', -1) >= 0:
                raise ValueError("Extension {!r} has EXTVER, but doesn't match any of SCI".format(name))
            setattr(self, name, process_meta_unit(None, other, add=False))
            self._exposed.append(name)

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

    @property
    def phu(self):
        return self._dataprov.phu_manipulator

    def ext(self, extname=None):
        return self._dataprov.ext_manipulator(extname)
