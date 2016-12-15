from types import StringTypes
from abc import abstractmethod
from copy import deepcopy
from collections import namedtuple, OrderedDict
import os
from functools import partial, wraps
from itertools import izip_longest, ifilterfalse

from .core import *

from astropy.io import fits
from astropy.io.fits import HDUList, Header, DELAYED
from astropy.io.fits import PrimaryHDU, ImageHDU, BinTableHDU
from astropy.io.fits import Column, FITS_rec
from astropy.io.fits.hdu.table import _TableBaseHDU
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
        if isinstance(value, tuple):
            self.set(key, value=value[0], comment=value[1])
        else:
            self.set(key, value=value)

    def __delattr__(self, key):
        self.remove(key)

    def __contains__(self, key):
        if self._on_ext:
            return any(tuple(key in h for h in self._headers))
        else:
            return key in self._headers[0]

def new_imagehdu(data, header, name=None):
# Assigning data in a delayed way, won't reset BZERO/BSCALE in the header,
# for some reason. Need to investigated. Maybe astropy.io.fits bug. Figure
# out WHY were we delaying in the first place.
#    i = ImageHDU(data=DELAYED, header=header.copy(), name=name)
#    i.data = data
    return ImageHDU(data=data, header=header.copy(), name=name)

def table_to_bintablehdu(table):
    array = table.as_array()
    header = table.meta['header'].copy()
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

header_type_map = {
        'bool': 'L',
        'int8': 'B',
        'int16': 'I',
        'int32': 'J',
        'int64': 'K',
        'uint8': 'B',
        'uint16': 'I',
        'uint32': 'J',
        'uint64': 'K',
        'float32': 'E',
        'float64': 'D',
        'complex64': 'C',
        'complex128': 'M'}

def header_for_table(table):
    columns = []
    for col in table.itercols():
        descr = {'name': col.name}
        typename = col.dtype.name
        if typename.startswith('string'): # Array of strings
            strlen = col.dtype.itemsize
            descr['format'] = '{}A'.format(strlen)
            descr['disp'] = 'A{}'.format(strlen)
        elif typename == 'object': # Variable length array
            raise TypeError("Variable length arrays like in column '{}' are not supported".format(col.name))
        else:
            try:
                typedesc = header_type_map[typename]
            except KeyError:
                raise TypeError("I don't know how to treat type {!r} for column {}".format(col.dtype, col.name))
            repeat = ''
            data = col.data
            shape = data.shape
            if len(shape) > 1:
                repeat = data.size / shape[0]
                if len(shape) > 2:
                    descr['dim'] = shape[1:]
            if typedesc == 'L' and len(shape) > 1:
                # Bit array
                descr['format'] = '{}X'.format(repeat)
            else:
                descr['format'] = '{}{}'.format(repeat, typedesc)
            if col.unit is not None:
                descr['unit'] = str(col.unit)

        columns.append(fits.Column(array=col.data, **descr))

    return fits.BinTableHDU.from_columns(columns).header

def add_header_to_table(table):
    header = header_for_table(table)
    table.meta['header'] = header
    return header

def card_filter(cards, include=None, exclude=None):
    for card in cards:
        if include is not None and card not in include:
            continue
        elif exclude is not None and card in exclude:
            continue
        yield card

def update_header(headera, headerb):
    cardsa = tuple(tuple(cr) for cr in headera.cards)
    cardsb = tuple(tuple(cr) for cr in headerb.cards)

    if cardsa == cardsb:
        return headera

    # Ok, headerb differs somehow. Let's try to bring the changes to
    # headera
    # Updated keywords that should be unique
    difference = set(cardsb) - set(cardsa)
    headera.update(card_filter(difference, exclude={'HISTORY', 'COMMENT', ''}))
    # Check the HISTORY and COMMENT cards, just in case
    for key in ('HISTORY', 'COMMENT'):
        fltcardsa = card_filter(cardsa, include={key})
        fltcardsb = card_filter(cardsb, include={key})
        for (ca, cb) in izip_longest(fltcardsa, fltcardsb):
            if cb is None:
                headera.update((cb,))

    return headera

def normalize_indices(slc, nitems):
    multiple = True
    if isinstance(slc, slice):
        start, stop, step = slc.indices(nitems)
        indices = range(start, stop, step)
    elif isinstance(slc, int):
        slc = (slc,)
        multiple = False
        # Normalize negative indices...
        indices = [(x if x >= 0 else nitems + x) for x in slc]
    else:
        raise ValueError("Invalid index: {}".format(slc))

    if any(i >= nitems for i in indices):
        raise IndexError("Index out of range")

    return indices, multiple

class FitsProviderProxy(DataProvider):
    # TODO: CAVEAT. Not all methods are intercepted. Some, like "info", may not make
    #       sense for slices. If a method of interest is identified, we need to
    #       implement it properly, or make it raise an exception if not valid.

    def __init__(self, provider, mapping, single):
        # We're overloading __setattr__. This is safer than setting the
        # attributes the normal way.
        self.__dict__.update({
            '_provider': provider,
            '_mapping': tuple(mapping),
            '_sliced': True,
            '_single': single
            })

    @property
    def is_sliced(self):
        return True

    @property
    def is_single(self):
        return self._single

    def __deepcopy__(self, memo):
        return self._provider._clone(mapping=self._mapping)

    def settable(self, attr):
        if attr in {'path', 'filename'}:
            return False

        return self._provider.settable(attr)

    def __len__(self):
        return len(self._mapping)

    def _mapped_nddata(self, idx=None):
        self._provider._lazy_populate_object()
        if idx is None:
            return [self._provider._nddata[idx] for idx in self._mapping]
        else:
            return self._provider._nddata[self._mapping[idx]]

    def __getattr__(self, attribute):
        if not attribute.startswith('_'):
            try:
                # Check first if this is something we can get from the main object
                # But only if it's not an internal attribute
                try:
                    return self._provider._getattr_impl(attribute, self._mapped_nddata())
                except AttributeError:
                    # Not a special attribute. Check the regular interface
                    return getattr(self._provider, attribute)
            except AttributeError:
                pass
        # Not found in the real Provider. Ok, if we're working with single
        # slices, let's look some things up in the ND object
        if self.is_single:
            if attribute.isupper():
                try:
                    return self._mapped_nddata(0).meta['other'][attribute]
                except KeyError:
                    # Not found. Will raise an exception...
                    pass
        raise AttributeError("{} not found in this object".format(attribute))

    def __setattr__(self, attribute, value):
        def _my_attribute(attr):
            return attr in self.__dict__ or attr in self.__class__.__dict__

        # This method is meant to let the user set certain attributes of the NDData
        # objects. First we check if the attribute belongs to this object's dictionary.
        # Otherwise, see if we can pass it down.

        if not _my_attribute(attribute) and self._provider.settable(attribute):
            if attribute.isupper():
                if not self.is_single:
                    raise TypeError("This attribute can only be assigned to a single-slice object")
                target = self._mapped_nddata(0)
                self._provider._append(value, name=attribute, add_to=target)
                return
            elif attribute in {'path', 'filename'}:
                raise AttributeError("Can't set path or filename on a sliced object")
            else:
                setattr(self._provider, attribute, value)

        super(FitsProviderProxy, self).__setattr__(attribute, value)

    def __delattr__(self, attribute):
        if not self.is_single:
            raise TypeError("Can't delete attributes on non-single slices")
        elif not attribute.isupper():
            raise ValueError("Can't delete non-capitalized attributes from slices")
        other, otherh = self.nddata.meta['other'], self.nddata.meta['other_header']
        if attribute in other:
            del other[attribute]
            if attribute in otherh:
                del otherh[attribute]
        else:
            raise AttributeError("'{}' does not exist in this extension".format(attribute))

    @property
    def exposed(self):
        return self._provider._exposed.copy() | set(self._mapped_nddata(0).meta['other'])

    def __getitem__(self, slc):
        if self.is_single:
            raise TypeError("Can't slice a single slice!")

        indices, multiple = normalize_indices(slc, nitems=len(self))
        mapped_indices = tuple(self._mapping[idx] for idx in indices)
        return self._provider._slice(mapped_indices, multi=multiple)

    def __delitem__(self, idx):
        raise TypeError("Can't remove items from a sliced object")

    def __iadd__(self, operand):
        self._provider._standard_nddata_op(NDDataObject.add, operand, self._mapping)
        return self

    def __isub__(self, operand):
        self._provider._standard_nddata_op(NDDataObject.subtract, operand, self._mapping)
        return self

    def __imul__(self, operand):
        self._provider._standard_nddata_op(NDDataObject.multiply, operand, self._mapping)
        return self

    def __idiv__(self, operand):
        self._provider._standard_nddata_op(NDDataObject.divide, operand, self._mapping)
        return self

    @property
    def header(self):
        return [self._provider._header[idx] for idx in [0] + [n+1 for n in self._mapping]]

    @property
    def data(self):
        if self.is_single:
            return self._mapped_nddata(0).data
        else:
            return [nd.data for nd in self._mapped_nddata()]

    @data.setter
    def data(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that is not a single slice")

        ext = self._mapped_nddata(0)
        # Setting the ._data in the NDData is a bit kludgy, but we're all grown adults
        # and know what we're doing, isn't it?
        if hasattr(value, 'shape'):
            ext._data = value
        else:
            raise AttributeError("Trying to assign data to be something with no shape")

    @property
    def uncertainty(self):
        if self.is_single:
            return self._mapped_nddata(0).uncertainty
        else:
            return [nd.uncertainty for nd in self._mapped_nddata()]

    @uncertainty.setter
    def uncertainty(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that is not a single slice")
        self._mapped_nddata(0).uncertainty = value

    @property
    def mask(self):
        if self.is_single:
            return self._mapped_nddata(0).mask
        else:
            return [nd.mask for nd in self._mapped_nddata()]

    @mask.setter
    def mask(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that is not a single slice")
        self._mapped_nddata(0).mask = value

    @property
    def variance(self):
        def variance_for(un):
            if un is not None:
                return un.array**2

        if self.is_single:
            return variance_for(self.uncertainty)
        else:
            return [variance_for(un) for un in self.uncertainty]

    @variance.setter
    def variance(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that is not a single slice")
        nd = self._mapped_nddata(0)
        if value is None:
            nd.uncertainty = None
        else:
            nd.uncertainty = StdDevUncertainty(np.sqrt(value))

    @property
    def nddata(self):
        if not self.is_single:
            return self._mapped_nddata()
        else:
            return self._mapped_nddata(0)

    @property
    def ext_manipulator(self):
        return FitsKeywordManipulator(self.header[1:], on_extensions=True, single=self.is_single)

    def set_name(self, ext, name):
        self._provider.set_name(self._mapping[ext], name)

    def crop(self, x1, y1, x2, y2):
        self._crop_impl(x1, y1, x2, y2, self._mapped_nddata)

    def append(self, ext, name):
        if not self.is_single:
            # TODO: We could rethink this one, but leave it like that at the moment
            raise TypeError("Can't append pixel planes to non-single slices")
        elif name is None:
            raise TypeError("Can't append pixel planes to a slice without an extension name")
        target = self._mapped_nddata(0)

        return self._provider._append(ext, name=name, add_to=target)

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
        if self.is_single:
            raise ValueError("Trying to get a mapping out of a single slice")

        return self._provider._extver_impl(self._mapped_nddata)

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

    @force_load
    def _clone(self, mapping=None):
        if mapping is None:
            mapping = range(len(self))

        dp = FitsProvider()
        dp._header = [deepcopy(self._header[0])]
        for n in mapping:
            dp.append(self._nddata[n])
        for t in self._tables.values():
            dp.append(t)

        return dp

    def settable(self, attr):
        return attr in self._fixed_settable or attr.isupper()

    @force_load
    def _getattr_impl(self, attribute, nds):
        # Exposed objects are part of the normal object interface. We may have
        # just lazy-loaded them, and that's why we get here...
        if attribute in self._exposed:
            return getattr(self, attribute)

        # Check if it's an aliased object
        for nd in nds:
            if nd.meta.get('name') == attribute:
                return nd

        raise AttributeError("Not found")

    @force_load
    def __getattr__(self, attribute):
        try:
            return self._getattr_impl(attribute, self._nddata)
        except AttributeError:
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
                self._lazy_populate_object()
                self._append(value, name=attribute, add_to=None)
                return

        # Fallback
        super(FitsProvider, self).__setattr__(attribute, value)

    @force_load
    def _oper(self, operator, operand, indices=None):
        if indices is None:
            indices = tuple(range(len(self._nddata)))
        if isinstance(operand, AstroData):
            if len(operand) != len(indices):
                raise ValueError("Operands are not the same size")
            for n in indices:
                try:
                    self._nddata[n] = operator(self._nddata[n], operand.nddata[n])
                except TypeError:
                    # This may happen if operand is a sliced, single AstroData object
                    self._nddata[n] = operator(self._nddata[n], operand.nddata)
            op_table = operand.table()
            ltab, rtab = set(self._tables), set(op_table)
            for tab in (rtab - ltab):
                self._tables[tab] = op_table[tab]
        else:
            for n in indices:
                self._nddata[n] = operator(self._nddata[n], operand)

    def _standard_nddata_op(self, fn, operand, indices=None):
        return self._oper(partial(fn, handle_mask=np.bitwise_or, handle_meta='first_found'),
                          operand, indices)

    def __iadd__(self, operand):
        self._standard_nddata_op(NDDataObject.add, operand)
        return self

    def __isub__(self, operand):
        self._standard_nddata_op(NDDataObject.subtract, operand)
        return self

    def __imul__(self, operand):
        self._standard_nddata_op(NDDataObject.multiply, operand)
        return self

    def __idiv__(self, operand):
        self._standard_nddata_op(NDDataObject.divide, operand)
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
            header = obj.meta['header']
            other_objects = []
            fixed = (('uncertainty', obj.uncertainty), ('mask', obj.mask))
            for name, other in fixed + tuple(sorted(obj.meta['other'].items())):
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
        return FitsProviderProxy(self, indices, single=not multi)

    @force_load
    def __getitem__(self, slc):
        nitems = len(self._header) - 1 # The Primary HDU does not count
        indices, multiple = normalize_indices(slc, nitems=nitems)
        return self._slice(indices, multi=multiple)

    @force_load
    def __delitem__(self, idx):
        nitems = len(self._header) - 1 # The Primary HDU does not count
        if idx >= nitems or idx < (-nitems):
            raise IndexError("Index out of range")

        del self._header[idx + 1]
        del self._nddata[idx]

    def __len__(self):
        self._lazy_populate_object()
        return len(self._nddata)

    def _set_headers(self, hdulist, update=True):
        new_headers = [hdulist[0].header] + [x.header for x in hdulist[1:] if
                                                (x.header.get('EXTNAME') in ('SCI', None))]
        # When update is True, self._header should NEVER be None, but check anyway
        if update and self._header is not None:
            assert len(self._header) == len(new_headers)
            self._header = [update_header(ha, hb) for (ha, hb) in zip(new_headers, self._header)]
        else:
            self._header = new_headers
        tables = [unit for unit in hdulist if isinstance(unit, BinTableHDU)]
        for table in tables:
            name = table.header.get('EXTNAME')
            if name == 'OBJCAT':
                continue
            self._tables[name] = None
            self._exposed.add(name)

    # NOTE: This one does not make reference to self at all. May as well
    #       move it out
    def _process_table(self, table, name=None):
        if isinstance(table, BinTableHDU):
            obj = Table(table.data, meta={'header': table.header})
        elif isinstance(table, Table):
            obj = Table(table)
            if 'header' not in obj.meta:
                obj.meta['header'] = header_for_table(obj)
        else:
            raise ValueError("{} is not a recognized table type".format(table.__class__))

        if name is not None:
            obj.meta['header']['EXTNAME'] = name

        return obj

    def _process_pixel_plane(self, pixim, name=None, top_level=False, reset_ver=False):
        if not isinstance(pixim, NDDataObject):
            # Assume that we get an ImageHDU or something that can be
            # turned into one
            if not isinstance(pixim, ImageHDU):
                pixim = ImageHDU(pixim)

            header = pixim.header
            nd = NDDataObject(pixim.data, meta={'header': header})

            currname = header.get('EXTNAME')
            ver = header.get('EXTVER', -1)
        else:
            nd = pixim
            header = nd.meta['header']
            currname = header.get('EXTNAME')
            ver = header.get('EXTVER', -1)

        if name and (currname is None):
            header['EXTNAME'] = (name if name is not None else 'SCI')

        if top_level:
            if 'other' not in nd.meta:
                nd.meta['other'] = OrderedDict()
                nd.meta['other_header'] = {}

            if reset_ver or ver == -1:
                try:
                    ver = max(_nd.meta['ver'] for _nd in self._nddata) + 1
                except ValueError:
                    # Got an empty sequence. This is the first extension!
                    ver = 1
                header['EXTVER'] = ver
                oheaders = nd.meta['other_header']
                for extname, ext in nd.meta['other'].items():
                    try:
                        oheaders[extname]['EXTVER'] = ver
                    except KeyError:
                        # This must be a table. Assume that it has meta
                        ext.meta['header']['EXTVER'] = ver

            nd.meta['ver'] = ver

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

            self._nddata = []
            sci_units = [x for x in hdulist[1:] if x.header['EXTNAME'] == 'SCI']

            for unit in sci_units:
                seen.add(unit)
                ver = unit.header.get('EXTVER', -1)
                nd = self._append(unit, name='SCI')

                for extra_unit in search_for_associated(ver):
                    seen.add(extra_unit)
                    name = extra_unit.header.get('EXTNAME')
                    self._append(extra_unit, name=name, add_to=nd)

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
                added = self._append(other, name=name)
        finally:
            self._resetting = prev_reset

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if self._path is not None:
            self._lazy_populate_object()
        elif value is not None:
            self._orig_filename = os.path.basename(value)
        self._path = value

    @property
    def filename(self):
        if self.path is not None:
            return os.path.basename(self.path)

    @filename.setter
    def filename(self, value):
        if os.path.isabs(value):
            raise ValueError("Cannot set the filename to an absolute path!")
        elif self.path is None:
            self.path = os.path.abspath(value)
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
                # Make sure that we have an HDUList to work with. Maybe we're creating
                # an object from scratch
                if hdulist is not None:
                    # We need to replace the headers, to make sure that we don't end
                    # up with different objects in self._headers and elsewhere
                    self._set_headers(hdulist, update=True)
                    self._reset_members(hdulist)
                    self._hdulist = None
                else:
                    self._nddata = []
            finally:
                self._resetting = prev_reset

    @property
    @force_load
    def nddata(self):
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

    @force_load
    def set_name(self, ext, name):
        self._nddata[ext].meta['name'] = name

    @force_load
    def to_hdulist(self):

        hlst = HDUList()
        hlst.append(PrimaryHDU(header=self._header[0], data=DELAYED))

        for ext in self._nddata:
            header, ver = ext.meta['header'], ext.meta['ver']

            hlst.append(new_imagehdu(ext.data, header))
            if ext.uncertainty is not None:
                hlst.append(new_imagehdu(ext.uncertainty.array ** 2, header, 'VAR'))
            if ext.mask is not None:
                hlst.append(new_imagehdu(ext.mask, header, 'DQ'))

            for name, other in ext.meta.get('other', {}).items():
                if isinstance(other, Table):
                    hlst.append(table_to_bintablehdu(other))
                elif isinstance(other, np.ndarray):
                    hlst.append(new_imagehdu(other, ext.meta['other_header'].get(name)))
                elif isinstance(other, NDDataObject):
                    hlst.append(new_imagehdu(other.data, other.meta['header']))
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
        return [nd.data for nd in self._nddata]

    @data.setter
    def data(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    @property
    @force_load
    def uncertainty(self):
        return [nd.uncertainty for nd in self._nddata]

    @uncertainty.setter
    def uncertainty(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    @property
    @force_load
    def mask(self):
        return [nd.mask for nd in self._nddata]

    @mask.setter
    def mask(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    @property
    @force_load
    def variance(self):
        def variance_for(nd):
            if nd.uncertainty is not None:
                return nd.uncertainty.array**2

        return [variance_for(nd) for nd in self._nddata]

    @variance.setter
    def variance(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    def _crop_nd(self, nd, x1, y1, x2, y2):
        nd.data = nd.data[y1:y2+1, x1:x2+1]
        if nd.uncertainty:
            nd.uncertainty = nd.uncertainty[y1:y2+1, x1:x2+1]
        if nd.mask:
            nd.mask = nd.mask[y1:y2+1, x1:x2+1]

    def _crop_impl(self, x1, y1, x2, y2, nds=None):
        if nds is None:
            nds = self.nddata
        # TODO: Consider cropping of objects in the meta section
        for nd in nds:
            dim = nd.data.shape
            self._crop_nd(nd, x1, y1, x2, y2)
            for o in nd.meta['other'].items():
                try:
                    if o.shape == dim:
                        self._crop_nd(o)
                except AttributeError:
                    # No 'shape' attribute in the object. It's probably not array-like
                    pass

    def crop(self, x1, y1, x2, y2):
        self._crop_impl(x1, y1, x2, y2)

    def _append(self, ext, name=None, add_to=None, reset_ver=False):
        self._lazy_populate_object()
        top = add_to is None
        if isinstance(ext, NDDataObject):
            ext = deepcopy(ext)
            self._header.append(ext.meta['header'])
            self._nddata.append(self._process_pixel_plane(ext, top_level=True, reset_ver=reset_ver))
            return ext
        else:
            add_to_other = None
            if isinstance(ext, (Table, _TableBaseHDU)):
                tb = self._process_table(ext, name)
                hname = tb.meta['header'].get('EXTNAME') if name is None else name
                if hname is None:
                    raise ValueError("Cannot add a table that has no EXTNAME")
                if top:
                    # Don't use setattr, which is overloaded and may case problems
                    self.__dict__[hname] = tb
                    self._tables[hname] = tb
                    self._exposed.add(hname)
                else:
                    setattr(add_to, hname, tb)
                    add_to_other = (hname, tb, tb.meta['header'])
                    add_to.meta['other'][hname] = tb
                ret = tb
            else: # Assume that this is a pixel plane
                # Special cases for Gemini
                if name in {'DQ', 'VAR'}:
                    if add_to is None:
                        raise ValueError("'{}' need to be associated to a 'SCI' one".format(name))
                    if name == 'DQ':
                        add_to.mask = ext.data
                        ret = ext.data
                    elif name == 'VAR':
                        std_un = StdDevUncertainty(np.sqrt(ext.data))
                        std_un.parent_nddata = add_to
                        add_to.uncertainty = std_un
                        ret = std_un
                elif top and name != 'SCI':
                    # Don't use setattr, which is overloaded and may case problems
                    self.__dict__[name] = ext
                    self._exposed.add(name)
                    ret = ext
                else:
                    nd = self._process_pixel_plane(ext, name=name, top_level=top)
                    if top:
                        self._nddata.append(nd)
                    else:
                        header = nd.meta['header']
                        hname = header.get('EXTNAME') if name is None else name
                        if hname is None:
                            raise TypeError("Can't append pixel planes to other objects without a name")
                        add_to_other = (hname, nd.data, header)

                    ret = nd
            try:
                oname, data, header = add_to_other
                header['EXTVER'] = add_to.meta.get('ver', -1)
                meta = add_to.meta
                meta['other'][oname] = data
                meta['other_header'][oname] = header
            except TypeError:
                pass

            return ret

    def append(self, ext, name=None, reset_ver=False):
        if isinstance(ext, PrimaryHDU):
            raise ValueError("Only one Primary HDU allowed")

        self._lazy_populate_object()

        return self._append(ext, name=name, add_to=None, reset_ver=reset_ver)

    def _extver_impl(self, nds=None):
        if nds is None:
            nds = self.nddata
        return dict((nd._meta['ver'], n) for (n, nd) in enumerate(nds))

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
        return self._extver_impl()

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

    def write(self, fileobj=None, clobber=False):
        if fileobj is None:
            if self.path is None:
                raise ValueError("A file name needs to be specified")
            fileobj = self.path
        self._dataprov.to_hdulist().writeto(fileobj, clobber=clobber)


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
