__all__ = ['AstroData', 'AstroDataError', 'factory']

from abc import abstractproperty
from types import StringTypes

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

class AstroDataError(Exception):
    pass

class AstroDataFactory(object):
    def __init__(self):
        self._registry = set()

    def addClass(self, cls):
        """
        Add a new class to the AstroDataFactory registry. It will be used when
        instantiating an AstroData class for a FITS file.
        """
        if not hasattr(cls, '_matches_data'):
            raise AstroDataError("Class '{}' contains no '_matches_data' method".format(cls.__name__))
        self._registry.add(cls)

    def _getAstroData(self, data_provider):
        """
        Searches the internal registry for an AstroData derivative matching
        the metadata in the HDUList that we're passed.

        Returns an instantiated object, or raises AstroDataError if it was
        not possible to find a match.
        """
        candidates = [x for x in self._registry if x._matches_data(data_provider)]

        # For every candidate in the list, remove the ones that are base classes
        # for other candidates. That way we keep only the more specific ones.
        final_candidates = []
        for cnd in candidates:
            if any(cnd in x.mro() for x in candidates if x != cnd):
                continue
            final_candidates.append(cnd)

        if len(final_candidates) > 1:
            raise AstroDataError("More than one class is candidate for this dataset")
        elif not final_candidates:
            raise AstroDataError("No class matches this dataset")

        return final_candidates[0](data_provider)

    def getAstroData(self, source):
        """
        Takes either a string (with the path to a file) or an HDUList as input, and
        tries to return an AstroData instance.

        It will raise exceptions if the file is not found, or if there is no match
        for the HDUList, among the registered AstroData classes.
        """

        # NOTE: This is not Python3 ready, but don't worry about it now...
        if isinstance(source, StringTypes):
            return self._getAstroData(FitsLoader.fromPath(source))
        else:
            # NOTE: This should be tested against the appropriate class.
            return self._getAstroData(FitsLoader.fromHduList(source))

class DataProvider(object):
    @abstractproperty
    def header(self):
        pass

    @abstractproperty
    def data(self):
        pass

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

class RawFitsProvider(DataProvider):
    # TODO: Implement
    def __init__(self):
        self._header = None
        self._nddata = None
        self.path = None

class ProcessedFitsProvider
    # TODO: Implement
    pass

class FitsLoader(DataProvider):
    @staticmethod
    def fromPath(path):
        fits_loader = FitsLoader()
        hdulist = fits.open(path, memmap=True, do_not_scale_image_data=True)
        fits_loader.path = path
        fits_loader._header = [x.header for x in hdulist
                                        if x.header.get('EXTNAME') not in FitsLoader.SKIP_HEADERS]

        return fits_loader

    @staticmethod
    def fromHduList(hdulist, path=None):
        fits_loader = FitsLoader()
        fits_loader.path = path
        fits_loader._reset_members(hdulist)

        return fits_loader

    def __init__(self):
        self._header = None
        self._nddata = None
        self._tables = None
        self.path = None

    def _reset_members(self, hdulist):
        def search_for_unit(name, ver):
            units = [x for x in hdulist
                       if x.header.get('EXTVER') == ver and x.header.get('EXTNAME') == name]
            if units:
                return units[0]

            return None

        self._header = []
        self._nddata = []
        self._tables = []

        seen_refcat = False
        for unit in hdulist:
            header = unit.header
            extname = header.get('EXTNAME')
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
            elif isinstance(unit, fits.ImageHDU):
                obj = NDData(unit.data, meta={'hdu': header, 'ver': header.get('EXTVER')})
                self._nddata.append(obj)
            elif isinstance(unit, fits.BinTableHDU):
                # REFCAT is the same, no matter how many copies. Have only one of them.
                is_refcat = extname != 'REFCAT'
                if not is_refcat or not seen_refcat:
                    obj = Table(unit.data, meta={'hdu': header, 'ver': header.get('EXTVER')})
                    seen_refcat = is_refcat
            elif isinstance(unit, fits.PrimaryHDU):
                self._header.append(header)
                continue
            else:
                print type(unit)
                raise Exception("I don't really know what to do here...")

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        if self._data is None:
            self._reset_members(fits.open(self.path))

        return self._data

    @property
    def phu(self):
        return self.header[0]

    @property
    def manipulator(self):
        return FitsKeywordManipulator(self.header)

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

    def __getitem__(self, slicing):
        return self.__class__(self._dataprov[slicing])

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

factory = AstroDataFactory()
# Let's make sure that there's at least one class that matches the data
# (if we're dealing with a FITS file)
factory.addClass(AstroDataFits)
