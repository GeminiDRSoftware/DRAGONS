from types import StringTypes
from abc import abstractmethod
from collections import defaultdict
import os
from functools import partial, wraps

from .core import *

from astropy.io import fits
from astropy.io.fits import HDUList, Header, DELAYED
from astropy.io.fits import PrimaryHDU, ImageHDU, BinTableHDU
from astropy.io.fits import Column, FITS_rec
# NDDataRef is still not in the stable astropy, but this should be the one
# we use in the future...
from astropy.nddata import NDDataRef as NDDataObject
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
import numpy as np

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
    def __init__(self, headers, on_extensions=False, single=False):
        self.__dict__.update({
            "_headers": headers,
            "_single": single,
            "_on_ext": on_extensions
        })

    def _ret_ext(self, values):
        if self._single and len(self._headers) == 1:
            return values[0]
        else:
            return values

    @property
    def keywords(self):
        if self._on_ext:
            return self._ret_ext([set(h.keys()) for h in self._headers])
        else:
            return set(self._headers[0].keys())

    def show(self):
        if self._on_ext:
            for n, header in enumerate(self._headers):
                print("==== Header #{} ====".format(n))
                print(repr(header))
        else:
            print(repr(self._headers[0]))

    def set(self, key, value=None, comment=None):
        for header in self._headers:
            header.set(key, value=value, comment=comment)

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except KeyError as err:
            if self._on_ext:
                vals = err.values
                for n in err.missing_at:
                    vals[n] = default
                return self._ret_ext(vals)
            else:
                return default

    def remove(self, key):
        if self._on_ext:
            deleted = 0
            for header in self._headers:
                try:
                    del header[key]
                    deleted = deleted + 1
                except KeyError:
                    pass
            if not deleted:
                raise AttributeError("'{}' is not on any of the extensions".format(key))
        else:
            try:
                del self._headers[0][key]
            except KeyError:
                raise AttributeError("'{}' is not on the PHU".format(key))

    def get_comment(self, key):
        if self._on_ext:
            return self._ret_ext([header.comments[key] for header in self._headers])
        else:
            return self._headers[0].comments[key]

    def set_comment(self, key, comment):
        def _inner_set_comment(header):
            if key not in header:
                raise KeyError("Keyword {!r} not available".format(key))

            header.set(key, comment=comment)

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
            return self._ret_ext(ret)
        else:
            return self._headers[0][key]

    def __setattr__(self, key, value):
        self.set(key, value=value)

    def __delattr__(self, key):
        self.remove(key)

    def __contains__(self, key):
        if self._on_ext:
            return any(tuple(key in h for h in self._headers))
        else:
            return key in self._headers[0]

def new_imagehdu(data, header, name=None):
    i = ImageHDU(data=DELAYED, header=header.copy(), name=name)
    i.data = data
    return i

def table_to_bintablehdu(table):
    array = table.as_array()
    header = table.meta['hdu'].copy()
    coldefs = []
    for n, name in enumerate(array.dtype.names, 1):
        coldefs.append(Column(
            name   = header.get('TTYPE{}'.format(n)),
            format = header.get('TFORM{}'.format(n)),
            unit   = header.get('TUNIT{}'.format(n)),
            null   = header.get('TNULL{}'.format(n)),
            bscale = header.get('TSCAL{}'.format(n)),
            bzero  = header.get('TZERO{}'.format(n)),
            disp   = header.get('TDISP{}'.format(n)),
            start  = header.get('TBCOL{}'.format(n)),
            dim    = header.get('TDIM{}'.format(n)),
            array  = array[name]
        ))

    return BinTableHDU(data=FITS_rec.from_columns(coldefs), header=header)

def force_load(fn):
    @wraps(fn)
    def wrapper(self, *args, **kw):
        # Force the loading of data, we may need it later
        self._lazy_populate_object()
        return fn(self, *args, **kw)
    return wrapper

class FitsProvider(DataProvider):
    def __init__(self):
        # We're overloading __setattr__. This is safer than setting the
        # attributes the normal way.
        self.__dict__.update({
            '_sliced': False,
            '_single': False,
            '_header': None,
            '_nddata': None,
            '_hdulist': None,
            '_path': None,
            '_orig_filename': None,
            '_tables': {},
            '_exposed': set(),
            '_resetting': False,
            '_fixed_settable': set([
                'data',
                'uncertainty',
                'mask',
                'variance',
                'path',
                'filename'
                ])
            })

    def settable(self, attr):
        return attr in self._fixed_settable or attr.isupper()

    @force_load
    def __getattr__(self, attribute):
        # First, make sure that the object is not an exposed one. It may be that
        # we just lazy-loaded the contents for the first time...
        if attribute in self._exposed:
            return getattr(self, attribute)

        # So, the attribute hasn't been exposed. Probably an alias. Test...
        for nd in self._nddata:
            if nd.meta.get('name') == attribute:
                return nd

        # Not quite. Ok, if we're working with single slices, let's look some things up
        # in the ND object
        if self._single:
            if attribute.isupper() or attribute in ('data', 'mask', 'uncertainty'):
                try:
                    return getattr(self._nddata[0], attribute)
                except AttributeError:
                    # Not found. Will raise an exception...
                    pass
            raise AttributeError("{} not found in this object".format(attribute))
        else:
            raise AttributeError("{} not found in this object, or available only for sliced data".format(attribute))

    def __setattr__(self, attribute, value):
        def _my_attribute(attr):
            return attr in self.__dict__ or attr in self.__class__.__dict__

        # This method is meant to let the user set certain attributes of the NDData
        # objects.
        #
        # self._resetting shortcircuits the method when populating the object. In that
        # situation, we don't want to interfere. Of course, we need to check first
        # if self._resetting is there, because otherwise we enter a loop..
        if '_resetting' in self.__dict__ and not self._resetting and not _my_attribute(attribute):
            if attribute.isupper():
                if self._sliced and not self._single:
                    raise TypeError("This attribute cannot be assigned to a partial slice")
                self._lazy_populate_object()
                raise NotImplementedError("Assignment of new associated extensions is not yet implemented")
        # Fallback
        super(FitsProvider, self).__setattr__(attribute, value)

    def __oper(self, operator, operand):
        if isinstance(operand, AstroData):
            if len(operand) != len(self):
                raise ValueError("Operands are not the same size")
            for n in range(len(self)):
                try:
                    self._nddata[n] = operator(self._nddata[n], operand.nddata[n])
                except TypeError:
                    # This may happen if operand is a sliced, single AstroData object
                    self._nddata[n] = operator(self._nddata[n], operand.nddata)
            ltab, rtab = set(self._tables), set(operand.table)
            for tab in (rtab - ltab):
                self._tables[tab] = operand.table[tab]
        else:
            for n in range(len(self)):
                self._nddata[n] = operator(self._nddata[n], operand)

    def __iadd__(self, operand):
        self.__oper(partial(NDDataObject.add, handle_meta='first_found'), operand)
        return self

    def __isub__(self, operand):
        self.__oper(partial(NDDataObject.subtract, handle_meta='first_found'), operand)
        return self

    def __imul__(self, operand):
        self.__oper(partial(NDDataObject.multiply, handle_meta='first_found'), operand)
        return self

    def __idiv__(self, operand):
        self.__oper(partial(NDDataObject.divide, handle_meta='first_found'), operand)
        return self

    def info(self, tags):
        print("Filename: {}".format(self.path if self.path else "Unknown"))
        # This is fixed. We don't support opening for update
        # print("Mode: readonly")

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
        self._lazy_populate_object()
        if len(self._nddata) > 0:
            main_fmt = "{:6} {:24} {:17} {:14} {}"
            other_fmt = "          .{:20} {:17} {:14} {}"
            print("\nPixels Extensions")
            print(main_fmt.format("Index", "Content", "Type", "Dimensions", "Format"))
            for pi in self._pixel_info():
                main_obj = pi['main']
                print(main_fmt.format(pi['idx'], main_obj['content'][:24], main_obj['type'][:17],
                                                 main_obj['dim'], main_obj['data_type']))
                for other in pi['other']:
                    print(other_fmt.format(other['attr'][:20], other['type'][:17], other['dim'],
                                           other['data_type']))

        additional_ext = list(self._other_info())
        if additional_ext:
            print("\nOther Extensions")
            print("               Type        Dimensions")
            for (attr, type_, dim) in additional_ext:
                print(".{:13} {:11} {}".format(attr[:13], type_[:11], dim))

    def _pixel_info(self):
        self._lazy_populate_object()
        for idx, obj in enumerate(self._nddata):
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
                        dim = ''
                        if hasattr(other, 'dtype'):
                            dt = other.dtype.name
                            dim = str(other.shape)
                        elif hasattr(other, 'data'):
                            dt = other.data.dtype.name
                            dim = str(other.data.shape)
                        elif hasattr(other, 'array'):
                            dt = other.array.dtype.name
                            dim = str(other.array.shape)
                        else:
                            dt = 'unknown'
                        other_objects.append(dict(
                            attr=name, type=type(other).__name__,
                            dim=dim, data_type = dt
                        ))

            yield dict(
                    idx = '[{:2}]'.format(idx),
                    main = dict(
                        content = 'science',
                        type = type(obj).__name__,
                        dim = '({})'.format(', '.join(str(s) for s in obj.data.shape)),
                        data_type = obj.data.dtype.name
                    ),
                    other = other_objects
            )

    def _other_info(self):
        # NOTE: This covers tables, only. Study other cases before implementing a more general solution
        if self._tables:
            for name, table in sorted(self._tables.items()):
                if type(table) is list:
                    # This is not a free floating table
                    continue
                yield (name, 'Table', (len(table), len(table.columns)))

    @property
    def exposed(self):
        self._lazy_populate_object()
        return self._exposed.copy()

    @force_load
    def _slice(self, indices, multi=True):
        # if not self._sliced and not self._hdulist:
        #     # Force the loading of data, we may need it later
        #     self.nddata
        scopy = self.__class__()
        scopy._sliced = True
        scopy._single = not multi
        scopy.path = self.path
        scopy._header = [self._header[0]] + [self._header[n+1] for n in indices]
        if self._nddata:
            scopy._nddata = [self._nddata[n] for n in indices]

        scopy._tables = {}
        for name, content in self._tables.items():
            if type(content) is list:
                scopy._tables[name] = [content[n] for n in indices]
            else:
                scopy._tables[name] = content

        return scopy

    @force_load
    def __getitem__(self, slc):
        nitems = len(self._header) - 1 # The Primary HDU does not count
        multiple = True
        if isinstance(slc, slice):
            start, stop, step = slc.indices(nitems)
            indices = range(start, stop, step)
        else:
            if isinstance(slc, int):
                slc = (slc,)
                multiple = False
            # Normalize negative indices...
            indices = [(x if x >= 0 else nitems + x) for x in slc]
        if any(i >= nitems for i in indices):
            raise IndexError("Index out of range")

        return self._slice(indices, multi=multiple)

    @force_load
    def __delitem__(self, idx):
        if self._sliced:
            raise TypeError("Can't remove items from a sliced object")

        nitems = len(self._header) - 1 # The Primary HDU does not count
        if idx >= nitems or idx < (-nitems):
            raise IndexError("Index out of range")

        del self._header[idx + 1]
        del self._nddata[idx]

    def __len__(self):
        self._lazy_populate_object()
        return len(self._nddata)

    def _set_headers(self, hdulist):
        self._header = [hdulist[0].header] + [x.header for x in hdulist[1:] if
                                                (x.header.get('EXTNAME') in ('SCI', None))]
        tables = [unit for unit in hdulist if isinstance(unit, BinTableHDU)]
        for table in tables:
            name = table.header.get('EXTNAME')
            if name == 'OBJCAT':
                continue
            self._tables[name] = None
            self._exposed.add(name)

    def _add_table(self, table, name=None):
        if isinstance(table, BinTableHDU):
            meta_obj = Table(table.data, meta={'hdu': table.header})
            hname = table.header.get('EXTNAME')
        elif isinstance(table, Table):
            meta_obj = table
            hname = table.meta['hdu'].get('EXTNAME')

        if hname is None:
            if name is None:
                raise ValueError("Cannot add a table that has no EXTNAME")
            table.meta['hdu']['EXTNAME'] = name

        return meta_obj

    def _add_pixel_image(self, pixim, append=False, name=None):
        if not isinstance(pixim, ImageHDU):
            hdu = ImageHDU(pixim)

        header = pixim.header
        nd = NDDataObject(pixim.data, meta={'hdu': header})

        if append:
            ver = header.get('EXTVER', -1)
            nd.meta['other'] = []

            if header.get('EXTNAME') is None:
                header['EXTNAME'] = (name if name is not None else 'SCI')

            if ver == -1:
                ver = max(_nd.meta['ver'] for _nd in self._nddata)
                header['EXTVER'] = ver
            nd.meta['ver'] = ver

            self._nddata.append(nd)

        return nd

    def _reset_members(self, hdulist):
        prev_reset = self._resetting
        self._resetting = True
        try:
            self._tables = {}
            seen = set([hdulist[0]])

            skip_names = set(['SCI', 'REFCAT', 'MDF'])

            def search_for_associated(ver):
                return [x for x in hdulist
                          if x.header.get('EXTVER') == ver and x.header['EXTNAME'] not in skip_names]

            def process_unit(nd, meta):
                eheader = meta.header
                name = eheader.get('EXTNAME')
                data = meta.data
                if name == 'DQ':
                    nd.mask = data
                elif name == 'VAR':
                    std_un = StdDevUncertainty(np.sqrt(data))
                    std_un.parent_nddata = nd
                    nd.uncertainty = std_un
                else:
                    if isinstance(meta, BinTableHDU):
                        meta_obj = self._add_table(meta)
                    elif isinstance(meta, ImageHDU):
                        meta_obj = NDDataObject(data, meta={'hdu': eheader})
                    else:
                        raise ValueError("Unknown extension type: {!r}".format(name))
                    return name, meta_obj

            self._nddata = []
            sci_units = [x for x in hdulist[1:] if x.header['EXTNAME'] == 'SCI']

            for unit in sci_units:
                seen.add(unit)
                ver = unit.header.get('EXTVER', -1)
                nd = self._add_pixel_image(unit, append=True)

                for extra_unit in search_for_associated(ver):
                    seen.add(extra_unit)
                    ret = process_unit(nd, extra_unit)
                    if ret is not None:
                        name, obj = ret
                        setattr(nd, name, obj)
                        nd.meta['other'].append(name)

            for other in hdulist:
                if other in seen:
                    continue
                name = other.header['EXTNAME']
                if name in self._tables:
                    continue
    # TODO: Fix it
    # NOTE: This happens with GPI. Let's leave it for later...
    #            if other.header.get('EXTVER', -1) >= 0:
    #                raise ValueError("Extension {!r} has EXTVER, but doesn't match any of SCI".format(name))
                processed = process_unit(None, other)
                if processed is not None:
                    name, processed = processed
                setattr(self, name, processed)
                self._exposed.add(name)
                if isinstance(processed, Table):
                    self._tables[name] = processed
        finally:
            self._resetting = prev_reset

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if self._path is not None:
            self._lazy_populate_object()
        else:
            self._orig_filename = os.path.basename(value)
        self._path = value

    @property
    def filename(self):
        if self.path is not None:
            return os.path.basename(self.path)

    @filename.setter
    def filename(self, value):
        if self.path is None:
            self.path = os.path.abspath(value)
        elif os.path.isabs(value):
            raise ValueError("Cannot set the filename to an absolute path!")
        else:
            dirname = os.path.dirname(self.path)
            self.path = os.path.join(dirname, value)

    @property
    def orig_filename(self):
        return self._orig_filename

    @property
    def header(self):
        return self._header

    def _lazy_populate_object(self):
        prev_reset = self._resetting
        if self._nddata is None:
            self._resetting = True
            try:
                if self.path:
                    hdulist = FitsLoader._prepare_hdulist(fits.open(self.path))
                else:
                    hdulist = self._hdulist
                # We need to replace the headers, to make sure that we don't end
                # up with different objects in self._headers and elsewhere
                self._set_headers(hdulist)
                self._reset_members(hdulist)
                self._hdulist = None
            finally:
                self._resetting = prev_reset

    @property
    def nddata(self):
        self._lazy_populate_object()

        if not self._single:
            return self._nddata
        else:
            return self._nddata[0]

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
        return FitsKeywordManipulator(self.header[1:], on_extensions=True, single=self._single)

    @force_load
    def set_name(self, ext, name):
        self._nddata[ext].meta['name'] = name

    @force_load
    def to_hdulist(self):

        hlst = HDUList()
        hlst.append(PrimaryHDU(header=self._header[0], data=DELAYED))

        for ext in self._nddata:
            header, ver = ext.meta['hdu'], ext.meta['ver']

            hlst.append(new_imagehdu(ext.data, header))
            if ext.uncertainty is not None:
                hlst.append(new_imagehdu(ext.uncertainty.array ** 2, header, 'VAR'))
            if ext.mask is not None:
                hlst.append(new_imagehdu(ext.mask, header, 'DQ'))

            for name in ext.meta.get('other', ()):
                other = getattr(ext, name)
                if isinstance(other, Table):
                    hlst.append(table_to_bintablehdu(other))
                elif isinstance(other, NDDataObject):
                    hlst.append(new_imagehdu(other.data, other.meta['hdu']))
                else:
                    raise ValueError("I don't know how to write back an object of type {}".format(type(other)))

        if self._tables is not None:
            for name, table in sorted(self._tables.items()):
                hlst.append(table_to_bintablehdu(table))

        return hlst

    @force_load
    def table(self):
        return self._tables.copy()

    @property
    def tables(self):
        return set(self._tables.keys())

    @property
    @force_load
    def data(self):
        if self._single:
            return self._nddata[0].data
        else:
            return [nd.data for nd in self._nddata]

    @data.setter
    @force_load
    def data(self, value):
        if not self._single:
            raise ValueError("Trying to assign to a non-sliced AstroData object")
        ext = self._nddata[0]
        # Setting the ._data in the NDData is a bit kludgy, but we're all grown adults
        # and know what we're doing, isn't it?
        if hasattr(value, 'shape'):
            ext._data = value
        else:
            raise AttributeError("Trying to assign data to be something with no shape")

    @property
    @force_load
    def uncertainty(self):
        if self._single:
            return self._nddata[0].uncertainty
        else:
            return [nd.uncertainty for nd in self._nddata]

    @uncertainty.setter
    @force_load
    def uncertainty(self, value):
        if not self._single:
            raise ValueError("Trying to assign uncertainty to a non-sliced AstroData object")
        self._nddata[0].uncertainty = value

    @property
    @force_load
    def mask(self):
        if self._single:
            return self._nddata[0].mask
        else:
            return [nd.mask for nd in self._nddata]

    @mask.setter
    @force_load
    def mask(self, value):
        if not self._single:
            raise ValueError("Trying to assign mask to a non-sliced AstroData object")
        self._nddata[0].mask = value

    @property
    @force_load
    def variance(self):
        def variance_for(nd):
            if nd.uncertainty is not None:
                return nd.uncertainty.array**2

        if self._single:
            return variance_for(self._nddata[0])
        else:
            return [variance_for(nd) for nd in self._nddata]

    @variance.setter
    @force_load
    def variance(self, value):
        if not self._single:
            raise ValueError("Trying to assign variance to a non-sliced AstroData object")
        if value is None:
            self._nddata[0].uncertainty = None
        else:
            self._nddata[0].uncertainty = StdDevUncertainty(np.sqrt(value))

    def _crop_nd(self, nd, x1, y1, x2, y2):
        nd.data = nd.data[y1:y2+1, x1:x2+1]
        if nd.uncertainty:
            nd.uncertainty = nd.uncertainty[y1:y2+1, x1:x2+1]
        if nd.mask:
            nd.mask = nd.mask[y1:y2+1, x1:x2+1]

    @force_load
    def crop(self, x1, y1, x2, y2):
        # TODO: Consider cropping of objects in the meta section
        for nd in self._nddata:
            dim = nd.data.shape
            self._crop_nd(nd, x1, y1, x2, y2)
            for o in nd.meta['other']:
                if isinstance(o, NDData):
                    if o.shape == dim:
                        self._crop_nd(o)

    def append(self, ext, name=None):
        if isinstance(ext, (Table, BinTableHDU)):
            return self._add_table(ext, name=name)
        elif isinstance(ext, PrimaryHDU):
            raise ValueError("Only one Primary HDU allowed")
        else: # Assume that it is going to be something we know how to deal with...
            return self._add_pixel_image(ext, append=True, name=name)

    def extver_map(self):
        """
        Provide a mapping between the FITS EXTVER of an extension and the index
        that will be used to access it within this object.

        Returns
        -------
        A dictionary `{EXTVER:index, ...}`

        Raises
        ------
        ValueError
            If used against a single slice. It is of no use in that situation.
        """
        if self._single:
            raise ValueError("Trying to get a mapping out of a single slice")
        return dict((nd._meta['ver'], n) for (n, nd) in enumerate(self.nddata))

class FitsLoader(object):
    @staticmethod
    def provider_for_hdulist(hdulist):
        """
        Returns an instance of the appropriate DataProvider class,
        according to the HDUList object
        """

        return FitsProvider()

    @staticmethod
    def _prepare_hdulist(hdulist):
        new_list = []
        highest_ver = 0
        recognized = set()

        for n, unit in enumerate(hdulist):
            ev = unit.header.get('EXTVER')
            eh = unit.header.get('EXTNAME')
            if ev not in (-1, None) and eh is not None:
                highest_ver = max(highest_ver, unit.header['EXTVER'])
            elif not isinstance(unit, PrimaryHDU):
                continue

            new_list.append(unit)
            recognized.add(unit)

        for unit in hdulist:
            if unit in recognized:
                continue
            elif isinstance(unit, ImageHDU):
                highest_ver += 1
                if 'EXTNAME' not in unit.header:
                    unit.header['EXTNAME'] = ('SCI', 'Added by AstroData')
                if unit.header.get('EXTVER') in (-1, None):
                    unit.header['EXTVER'] = (highest_ver, 'Added by AstroData')

            new_list.append(unit)
            recognized.add(unit)

        def comp(a, b):
            ha, hb = a.header, b.header
            hav, hbv = ha.get('EXTVER'), hb.get('EXTVER')
            # A PrimaryHDU is always sorted first
            if isinstance(a, PrimaryHDU):
                return -1
            elif isinstance(b, PrimaryHDU):
                return 1
            elif hav not in (-1, None):
                # If both headers have EXTVER, compare based on EXTVER.
                # Else, b is a not a pixel image, push it to the end
                if hbv not in (-1, None):
                    ret = cmp(ha['EXTVER'], hb['EXTVER'])
                    # Break ties depending on EXTNAME. SCI goes first
                    if ret == 0:
                        if hav == 'SCI':
                            return -1
                        elif hbv == 'SCI':
                            return 1
                        else:
                            return 0
                    else:
                        return ret
                else:
                    return -1
            elif hbv not in (-1, None):
                # If b is the only one with EXTVER, push a to the end
                return 1
            else:
                # If none of them are PrimaryHDU, nor have an EXTVER
                # we don't care about the order
                return 0

        return HDUList(sorted(new_list, cmp=comp))

    @staticmethod
    def from_path(path):
        hdulist = fits.open(path, memmap=True, do_not_scale_image_data=True)
        hdulist = FitsLoader._prepare_hdulist(hdulist)
        provider = FitsLoader.provider_for_hdulist(hdulist)
        provider.path = path
        provider._set_headers(hdulist)
        # Note: we don't call _reset_members, to allow for lazy loading...

        return provider

    @staticmethod
    def from_hdulist(hdulist, path=None):
        provider = FitsLoader.provider_for_hdulist(hdulist)
        provider.path = path
        provider._hdulist = hdulist
        provider._set_headers(hdulist)
        provider._reset_members(hdulist)

        return provider

class AstroDataFits(AstroData):
    # Derived classes may provide their own __keyword_dict. Being a private
    # variable, each class will preserve its own, and there's no risk of
    # overriding the whole thing
    __keyword_dict = {
        'instrument': 'INSTRUME',
        'object': 'OBJECT',
        'telescope': 'TELESCOP',
        'ut_date': 'DATE-OBS'
    }

    def _keyword_for(self, name):
        """
        Returns the FITS keyword name associated to ``name``.

        Parameters
        ----------
        name : str
            The common "key" name for which we want to know the associated
            FITS keyword

        Returns
        -------
        str
            The desired keyword name

        Raises
        ------
        AttributeError
            If there is no keyword for the specified ``name``
        """

        for cls in self.__class__.mro():
            mangled_dict_name = '_{}__keyword_dict'.format(cls.__name__)
            try:
                return getattr(self, mangled_dict_name)[name]
            except (AttributeError, KeyError) as e:
                pass
        else:
            raise AttributeError("No match for '{}'".format(name))

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

    def write(self, filename=None, clobber=False):
        if filename is None:
            if self.path is None:
                raise ValueError("A file name needs to be specified")
            filename = self.path
        self._dataprov.to_hdulist().writeto(filename, clobber=clobber)


    @astro_data_descriptor
    def instrument(self):
        """
        Returns the name of the instrument making the observation

        Returns
        -------
        str
            instrument name
        """
        return self.phu.get(self._keyword_for('instrument'))


    @astro_data_descriptor
    def object(self):
        """
        Returns the name of the object being observed

        Returns
        -------
        str
            object name
        """
        return self.phu.get(self._keyword_for('object'))

    @astro_data_descriptor
    def telescope(self):
        """
        Returns the name of the telescope

        Returns
        -------
        str
            name of the telescope
        """
        return self.phu.get(self._keyword_for('telescope'))

    def extver(self, ver):
        """
        Get an extension using its EXTVER instead of the positional index in this
        object.

        Parameters
        ----------
        ver: int
             The EXTVER for the desired extension

        Returns
        -------
        A sliced object containing the desired extension

        Raises
        ------
        IndexError
            If the provided EXTVER doesn't exist
        """

        try:
            if isinstance(ver, int):
                return self[self._dataprov.extver_map()[ver]]
            else:
                raise ValueError("{} is not an integer EXTVER".format(ver))
        except KeyError as e:
            raise IndexError("EXTVER {} not found".format(e.args[0]))

# TODO: Remove this when we're sure that there are no external uses
def write(filename, ad_object, clobber=False):
    ad_object._dataprov.to_hdulist().writeto(filename, clobber=clobber)
