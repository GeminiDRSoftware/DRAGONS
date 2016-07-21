from types import StringTypes
from abc import abstractmethod
from collections import defaultdict
from functools import partial

from .core import *

from astropy.io import fits
# NDDataRef is still not in the stable astropy, but this should be the one
# we use in the future...
# VarUncertainty was pulled in from @csimpson's BardData code
from astropy.nddata import NDDataRef
from mynddata.nduncertainty import VarUncertainty
from astropy.table import Table

NO_DEFAULT = object()

class KeywordCallableWrapper(object):
    def __init__(self, keyword, default=NO_DEFAULT, on_ext=False, coerce_with=None):
        self.kw = keyword
        self.on_ext = on_ext
        self.default = default
        self.coercion_fn = coerce_with if coerce_with is not None else (lambda x: x)

    def __call__(self, adobj):
        def wrapper():
            manip = adobj.phu if not self.on_ext else adobj.hdr
            if self.default is NO_DEFAULT:
                ret = getattr(manip, self.kw)
            else:
                ret = manip.get(self.kw, self.default)
            return self.coercion_fn(ret)
        return wrapper

class FitsKeywordManipulator(object):
    def __init__(self, headers, on_extensions=False):
        self.__dict__.update({
            "_headers": headers,
            "_on_ext": on_extensions
        })

    @property
    def keywords(self):
        if self._on_ext:
            return [set(h.keys()) for h in self._headers]
        else:
            return set(self._headers[0].keys())

    def show(self):
        # TODO: This is only working for PHU now. Need to prepare it for extensions...
        if self._on_ext:
            for n, header in enumerate(self._headers):
                print("==== Header #{} ====".format(n))
                print(repr(header))
        else:
            print(repr(self._headers[0]))

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except KeyError as err:
            if self._on_ext:
                vals = err.values
                for n in err.missing_at:
                    vals[n] = default
                return vals
            else:
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

    def __contains__(self, key):
        if self._on_ext:
            return any(tuple(key in h for h in self._headers))
        else:
            return key in self._headers[0]

class FitsProvider(DataProvider):
    def __init__(self):
        self._sliced = False
        self._header = None
        self._nddata = None
        self._hdulist = None
        self.path = None

    def __iadd__(self, operand):
        for n in range(len(self._nddata)):
            nddata = self._nddata[n]
            self._nddata[n] = self._nddata[n].add(operand, handle_meta='first_found')
        return self

    def __isub__(self, operand):
        for n in range(len(self._nddata)):
            nddata = self._nddata[n]
            self._nddata[n] = self._nddata[n].subtract(operand, handle_meta='first_found')
        return self

    def __imul__(self, operand):
        for n in range(len(self._nddata)):
            nddata = self._nddata[n]
            self._nddata[n] = self._nddata[n].multiply(operand, handle_meta='first_found')
        return self

    def __idiv__(self, operand):
        for n in range(len(self._nddata)):
            nddata = self._nddata[n]
            self._nddata[n] = self._nddata[n].divide(operand, handle_meta='first_found')
        return self

    def info(self, tags):
        print("Filename: {}".format(self.path if self.path else "Unknown"))
        # NOTE: Right now we only support readonly, so it's fixed
        print("Mode: readonly")

        tags = sorted(tags, reverse=True)
        tag_line = "Tags: "
        while tags:
            new_tag = tags.pop() + ' '
            if len(tag_line + new_tag) > 80:
                print(tag_line)
                tag_line = "    " + new_tag
            else:
                tag_line = tag_line + new_tag
        print(tag_line)

        # Let's try to be generic. Could it be that some file contains only tables?
        if len(self.nddata) > 0:
            main_fmt = "{:6} {:24} {:16} {:14} {}"
            other_fmt = "          .{:20} {:16} {:14} {}"
            print("\nPixels Extensions")
            print(main_fmt.format("Ver", "Content", "Type", "Dimensions", "Format"))
            for pi in self._pixel_info():
                main_obj = pi['main']
                print(main_fmt.format(pi['ver'], main_obj['content'], main_obj['type'],
                                                 main_obj['dim'], main_obj['data_type']))
                for other in pi['other']:
                    print(other_fmt.format(other['attr'], other['type'], other['dim'],
                                           other['data_type']))

        additional_ext = list(self._other_info())
        if additional_ext:
            print("\nOther Extensions")
            print("               Type        Dimensions")
            for (attr, type_, dim) in additional_ext:
                print(".{:13} {:11} {}".format(attr, type_, dim))

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

    def __len__(self):
        return len(self.nddata)

    @abstractmethod
    def _set_headers(self, hdulist):
        pass

    @abstractmethod
    def _reset_members(self, hdulist):
        pass

    @property
    def header(self):
        return self._header

    def _lazy_populate_object(self):
        if self._nddata is None:
            if self.path:
                self._reset_members(fits.open(self.path))
            else:
                self._reset_members(self._hdulist)

    @property
    def nddata(self):
        self._lazy_populate_object()

        return self._nddata

    @property
    def phu(self):
        return self.header[0]

    @property
    def phu_manipulator(self):
        return FitsKeywordManipulator(self.header[:1])

    @property
    def ext_manipulator(self):
        if len(self.header) < 2:
            return None
        return FitsKeywordManipulator(self.header[1:], on_extensions=True)

class RawFitsProvider(FitsProvider):
    def _set_headers(self, hdulist):
        self._header = [x.header for x in hdulist]

    def _reset_members(self, hdulist):
        self._hdulist = hdulist
        self._nddata = []
        for unit in hdulist:
            if isinstance(unit, fits.ImageHDU):
                obj = NDDataRef(unit.data, meta={'hdu': unit.header, 'ver': -1})
                self._nddata.append(obj)

    def _pixel_info(self):
        for obj in self.nddata:
            yield dict(
                ver = '[NA]',
                main = dict(
                    content = 'raw',
                    type = type(obj).__name__,
                    dim = '({})'.format(', '.join(str(s) for s in obj.data.shape)),
                    data_type = obj.data.dtype.name
                ),
                other = ()
            )

    def _other_info(self):
        return ()

class ProcessedFitsProvider(FitsProvider):
    def __init__(self):
        super(ProcessedFitsProvider, self).__init__()
        self._tables = None
        self._exposed = []

    def _pixel_info(self):
        for obj in self.nddata:
            header = obj.meta['hdu']
            other_objects = []
            for name in ['uncertainty', 'mask'] + sorted(obj.meta['other']):
                other = getattr(obj, name)
                if other is not None:
                    if isinstance(other, Table):
                        other_objects.append(dict(
                            attr=name, type='Table',
                            dim=(len(other), len(other.columns)),
                            data_type='n/a'
                        ))
                    else:
                        if hasattr(other, 'dtype'):
                            dt = other.dtype.name
                        elif hasattr(other, 'data'):
                            dt = other.data.dtype.name
                        elif hasattr(other, 'array'):
                            dt = other.array.dtype.name
                        else:
                            dt = 'unknown'
                        other_objects.append(dict(
                            attr=name, type=type(other).__name__,
                            dim='', data_type = dt
                        ))

            yield dict(
                    ver = '[{:2}]'.format(header.get('EXTVER', -1)),
                    main = dict(
                        content = 'science',
                        type = type(obj).__name__,
                        dim = '({})'.format(', '.join(str(s) for s in obj.data.shape)),
                        data_type = obj.data.dtype.name
                    ),
                    other = other_objects
            )

    def _other_info(self):
        # TODO: This covers tables, only. Study other cases before implementing a more general solution
        if self._tables:
            for name, table in sorted(self._tables.items()):
                if type(table) is list:
                    # This is not a free floating table
                    continue
                yield (name, 'Table', (len(table), len(table.columns)))

    @property
    def exposed(self):
        self._lazy_populate_object()
        return set(self._exposed)

    def _slice(self, indices):
        scopy = super(ProcessedFitsProvider, self)._slice(indices)
        scopy._tables = {}
        if self._tables is not None:
            for name, content in self._tables.items():
                if type(content) is list:
                    scopy._tables[name] = [self.content[n] for n in indices]
                else:
                    scopy._tables[name] = self.content

        return scopy

    def _set_headers(self, hdulist):
        self._header = [hdulist[0].header] + [x.header for x in hdulist if x.header.get('EXTNAME') == 'SCI']

    def _reset_members(self, hdulist):
        self._hdulist = hdulist
        self._tables = {}
        seen = set([hdulist[0]])

        skip_names = set(['SCI', 'REFCAT', 'MDF'])

        def search_for_associated(ver):
            return [x for x in hdulist
                      if x.header.get('EXTVER') == ver and x.header['EXTNAME'] not in skip_names]

        def process_meta_unit(nd, meta, add=True):
            eheader = meta.header
            name = eheader.get('EXTNAME')
            data = meta.data
            if name == 'DQ':
                nd.mask = data
            elif name == 'VAR':
                var_un = VarUncertainty(data)
                var_un.parent_nddata = nd
                nd.uncertainty = var_un
            else:
                if isinstance(meta, fits.BinTableHDU):
                    meta_obj = Table(data, meta={'hdu': eheader})
                    if add is True:
                        if name in self._tables:
                            self._tables[name].append(meta_obj)
                        else:
                            self._tables[name] = [meta_obj]
                elif isinstance(meta, fits.ImageHDU):
                    meta_obj = NDDataRef(data, meta={'hdu': eheader})
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
            nd = NDDataRef(unit.data, meta={'hdu': header, 'ver': ver, 'other': []})
            self._nddata.append(nd)

            for extra_unit in search_for_associated(ver):
                seen.add(extra_unit)
                process_meta_unit(nd, extra_unit, add=True)

        for other in self._hdulist:
            if other in seen:
                continue
            name = other.header['EXTNAME']
            if name in self._tables:
                continue
# NOTE: This happens with GPI. Let's leave it for later...
#            if other.header.get('EXTVER', -1) >= 0:
#                raise ValueError("Extension {!r} has EXTVER, but doesn't match any of SCI".format(name))
            if isinstance(other, fits.BinTableHDU):
                self._tables[name] = Table(other.data, meta={'hdu': other.header})
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

    @property
    def hdr(self):
        return self._dataprov.ext_manipulator

    def info(self):
        self._dataprov.info(self.tags)
