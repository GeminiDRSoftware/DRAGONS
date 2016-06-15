from types import StringTypes

from .core import *
from abc import abstractmethod

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

class FitsProvider(DataProvider):
    def __init__(self):
        self._header = None
        self._nddata = None
        self._tables = None
        self._hdulist = None
        self.path = None

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
        self._nddata = []
        for unit in hdulist:
            if isinstance(unit, fits.ImageHDU):
                obj = NDData(unit.data, meta={'hdu': header})
                self._nddata.append(obj)

class ProcessedFitsProvider(FitsProvider):
    SKIP_HEADERS = ['DQ', 'VAR']

    def __init__(self):
        super(ProcessedFitsProvider, self).__init__()

    def _set_headers(self, hdulist):
        self._header = [x.header for x in hdulist
                                 if x.header['EXTNAME'] not in self.SKIP_HEADERS]

    def _reset_members(self, hdulist):
        def search_for_unit(name, ver):
            units = [x for x in hdulist
                       if x.header['EXTVER'] == ver and x.header['EXTNAME'] == name]
            if units:
                return units[0]

            return None

        self._nddata = []
        self._tables = []

        seen_refcat = False
        for unit in hdulist:
            header = unit.header
            extname = header['EXTNAME']
            if extname in self.SKIP_HEADERS:
                continue
            elif extname == 'SCI':
                self._header.append(header)
                ver = header.get('EXTVER')
                obj = NDData(unit.data, meta={'hdu': header, 'ver': ver})
                dq, var = search_for_unit('DQ', ver), search_for_unit('VAR', ver)
                if dq:
                    obj.mask = dq
                if var:
                    # TODO: set the uncertainty.
                    # obj.uncertainty = VarUncertainty(unit.data)
                    pass
                self._nddata.append(obj)
            elif isinstance(unit, fits.BinTableHDU):
                # REFCAT is the same, no matter how many copies. Have only one of them.
                is_refcat = extname != 'REFCAT'
                if not is_refcat or not seen_refcat:
                    obj = Table(unit.data, meta={'hdu': header, 'ver': header.get('EXTVER')})
                    seen_refcat = is_refcat
            elif isinstance(unit, fits.PrimaryHDU):
                continue
            else:
                print type(unit)
                raise Exception("I don't really know what to do here...")

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
