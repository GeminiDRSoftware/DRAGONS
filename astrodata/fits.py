from __future__ import print_function

from datetime import datetime

from future.utils import PY3

from builtins import object
from copy import deepcopy
from collections import OrderedDict
import os
import re
from functools import partial, wraps
import logging
import warnings
import gc
import inspect
import traceback

try:
    # Python 3
    from itertools import zip_longest, product as cart_product
except ImportError:
    from itertools import izip_longest as zip_longest, product as cart_product

from .core import AstroData, DataProvider, astro_data_descriptor
from .nddata import NDAstroData as NDDataObject, new_variance_uncertainty_instance

import astropy
from astropy.io import fits
from astropy.io.fits import HDUList, DELAYED
from astropy.io.fits import PrimaryHDU, ImageHDU, BinTableHDU
from astropy.io.fits import Column, FITS_rec
from astropy.io.fits.hdu.table import _TableBaseHDU
from astropy import units as u
# NDDataRef is still not in the stable astropy, but this should be the one
# we use in the future...
# from astropy.nddata import NDData, NDDataRef as NDDataObject
from astropy.nddata import NDData
from astropy.table import Table
import numpy as np

INTEGER_TYPES = (int, np.integer)
NO_DEFAULT = object()
LOGGER = logging.getLogger('AstroData FITS')

class AstroDataFitsDeprecationWarning(DeprecationWarning):
    pass

warnings.simplefilter("always", AstroDataFitsDeprecationWarning)

def deprecated(reason):
    def decorator_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kw):
            current_source = '|'.join(traceback.format_stack(inspect.currentframe()))
            if current_source not in wrapper.seen:
                wrapper.seen.add(current_source)
                warnings.warn(reason, AstroDataFitsDeprecationWarning)
            return fn(*args, **kw)
        wrapper.seen = set()
        return wrapper
    return decorator_wrapper

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

class FitsHeaderCollection(object):
    """
    FitsHeaderCollection(headers)

    This class provides group access to a list of PyFITS Header-like objects.
    It exposes a number of methods (`set`, `get`, etc.) that operate over all
    the headers at the same time.

    It can also be iterated.
    """
    def __init__(self, headers):
        self.__headers = list(headers)

    def _insert(self, idx, header):
        self.__headers.insert(idx, header)

    def __iter__(self):
        for h in self.__headers:
            yield h

#    @property
#    def keywords(self):
#        if self._on_ext:
#            return self._ret_ext([set(h.keys()) for h in self.__headers])
#        else:
#            return set(self.__headers[0].keys())
#
#    def show(self):
#        if self._on_ext:
#            for n, header in enumerate(self.__headers):
#                print("==== Header #{} ====".format(n))
#                print(repr(header))
#        else:
#            print(repr(self.__headers[0]))

    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            self.set(key, value=value[0], comment=value[1])
        else:
            self.set(key, value=value)

    def set(self, key, value=None, comment=None):
        for header in self.__headers:
            header.set(key, value=value, comment=comment)

    def __getitem__(self, key):
        raised = False
        missing_at = []
        ret = []
        for n, header in enumerate(self.__headers):
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

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError as err:
            vals = err.values
            for n in err.missing_at:
                vals[n] = default
            return vals

    def __delitem__(self, key):
        self.remove(key)

    def remove(self, key):
        deleted = 0
        for header in self.__headers:
            try:
                del header[key]
                deleted = deleted + 1
            except KeyError:
                pass
        if not deleted:
            raise KeyError("'{}' is not on any of the extensions".format(key))

    def get_comment(self, key):
        return [header.comments[key] for header in self.__headers]

    def set_comment(self, key, comment):
        def _inner_set_comment(header):
            if key not in header:
                raise KeyError("Keyword {!r} not available".format(key))

            header.set(key, comment=comment)

        for n, header in enumerate(self.__headers):
            try:
                _inner_set_comment(header)
            except KeyError as err:
                err.message = err.message + " at header {}".format(n)
                raise

    def __contains__(self, key):
        return any(tuple(key in h for h in self.__headers))

def new_imagehdu(data, header, name=None):
# Assigning data in a delayed way, won't reset BZERO/BSCALE in the header,
# for some reason. Need to investigated. Maybe astropy.io.fits bug. Figure
# out WHY were we delaying in the first place.
#    i = ImageHDU(data=DELAYED, header=header.copy(), name=name)
#    i.data = data
    return ImageHDU(data=data, header=header.copy(), name=name)

def table_to_bintablehdu(table, extname=None):
    add_header_to_table(table)
    array = table.as_array()
    header = table.meta['header'].copy()
    if extname:
        header['EXTNAME'] = (extname, 'added by AstroData')
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
    'complex128': 'M'
}


def header_for_table(table):
    columns = []
    for col in table.itercols():
        descr = {'name': col.name}
        typekind = col.dtype.kind
        typename = col.dtype.name
        if typekind in {'S', 'U'}: # Array of strings
            strlen = col.dtype.itemsize // col.dtype.alignment
            descr['format'] = '{}A'.format(strlen)
            descr['disp'] = 'A{}'.format(strlen)
        elif typekind == 'O': # Variable length array
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
                repeat = data.size // shape[0]
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

    fits_header = fits.BinTableHDU.from_columns(columns).header
    if 'header' in table.meta:
        fits_header = update_header(table.meta['header'], fits_header)
    return fits_header

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
        for (ca, cb) in zip_longest(fltcardsa, fltcardsb):
            if cb is None:
                headera.update((cb,))

    return headera

def normalize_indices(slc, nitems):
    multiple = True
    if isinstance(slc, slice):
        start, stop, step = slc.indices(nitems)
        indices = list(range(start, stop, step))
    elif isinstance(slc, INTEGER_TYPES) or (isinstance(slc, tuple) and all(isinstance(i, INTEGER_TYPES) for i in slc)):
        if isinstance(slc, INTEGER_TYPES):
            slc = (int(slc),)   # slc's type m
            multiple = False
        else:
            multiple = True
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

    def is_settable(self, attr):
        if attr in {'path', 'filename'}:
            return False

        return self._provider.is_settable(attr)

    def __len__(self):
        return len(self._mapping)

    def _mapped_nddata(self, idx=None):
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

        if not _my_attribute(attribute) and self._provider.is_settable(attribute):
            if attribute.isupper():
                if not self.is_single:
                    raise TypeError("This attribute can only be assigned to a single-slice object")
                target = self._mapped_nddata(0)
                self._provider.append(value, name=attribute, add_to=target)
                return
            elif attribute in {'path', 'filename'}:
                raise AttributeError("Can't set path or filename on a sliced object")
            else:
                setattr(self._provider, attribute, value)

        super(FitsProviderProxy, self).__setattr__(attribute, value)

    def __delattr__(self, attribute):
        if not attribute.isupper():
            raise ValueError("Can't delete non-capitalized attributes from slices")
        if not self.is_single:
            raise TypeError("Can't delete attributes on non-single slices")
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

    def __iter__(self):
        if self._single:
            yield self
        else:
            for n in self._mapping:
                yield self._provider._slice((n,), multi=False)

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

    __itruediv__ = __idiv__

    def __rdiv__(self, operand):
        self._provider._oper(self._provider._rdiv, operand, self._mapping)
        return self

    @property
    @deprecated("Access to headers through this property is deprecated and will be removed in the future. "
                "Use '.hdr' instead.")
    def header(self):
        return self._provider._get_raw_headers(with_phu=True, indices=self._mapping)

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
        if self.is_single:
            return self._mapped_nddata(0).variance
        else:
            return [nd.variance for nd in self._mapped_nddata()]

    @variance.setter
    def variance(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that is not a single slice")
        nd = self._mapped_nddata(0)
        if value is None:
            nd.uncertainty = None
        else:
            nd.uncertainty = new_variance_uncertainty_instance(value)

    @property
    def nddata(self):
        if not self.is_single:
            return self._mapped_nddata()
        else:
            return self._mapped_nddata(0)

    @property
    def shape(self):
        if not self.is_single:
            return [nd.shape for nd in self._mapped_nddata()]
        else:
            return self._mapped_nddata(0).shape

    def hdr(self):
        headers = self._provider._get_raw_headers(indices=self._mapping)

        return headers[0] if self.is_single else FitsHeaderCollection(headers)

    def set_name(self, ext, name):
        self._provider.set_name(self._mapping[ext], name)

    def crop(self, x1, y1, x2, y2):
        self._crop_impl(x1, y1, x2, y2, self._mapped_nddata())

    def append(self, ext, name):
        if not self.is_single:
            # TODO: We could rethink this one, but leave it like that at the moment
            raise TypeError("Can't append objects to non-single slices")
        elif name is None:
            raise TypeError("Can't append objects to a slice without an extension name")
        target = self._mapped_nddata(0)

        return self._provider.append(ext, name=name, add_to=target)

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

        return self._provider._extver_impl(self._mapped_nddata())

    def info(self, tags):
        self._provider.info(tags, indices=self._mapping)

class FitsProvider(DataProvider):
    default_extension = 'SCI'
    def __init__(self):
        # We're overloading __setattr__. This is safer than setting the
        # attributes the normal way.
        self.__dict__.update({
            '_sliced': False,
            '_single': False,
            '_phu': None,
            '_nddata': [],
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

    def __deepcopy__(self, memo):
        nfp = FitsProvider()
        to_copy = ('_sliced', '_phu', '_single', '_nddata',
                   '_path', '_orig_filename', '_tables', '_exposed',
                   '_resetting')
        for attr in to_copy:
            nfp.__dict__[attr] = deepcopy(self.__dict__[attr])

        # Top-level tables
        for key in set(self.__dict__) - set(nfp.__dict__):
            nfp.__dict__[key] = nfp.__dict__['_tables'][key]

        return nfp

    def _clone(self, mapping=None):
        if mapping is None:
            mapping = range(len(self))

        dp = FitsProvider()
        dp._phu = deepcopy(self._phu)
        for n in mapping:
            dp.append(deepcopy(self._nddata[n]))
        for t in self._tables.values():
            dp.append(deepcopy(t))

        return dp

    def is_settable(self, attr):
        return attr in self._fixed_settable or attr.isupper()

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
        # CJS 20200131: if the attribute is "exposed" then we should set it via the
        # append method I think (it's a Table or something)
        if ('_resetting' in self.__dict__ and not self._resetting and
                (not _my_attribute(attribute) or attribute in self._exposed)):
            if attribute.isupper():
                self.append(value, name=attribute, add_to=None)
                return

        # Fallback
        super(FitsProvider, self).__setattr__(attribute, value)

    def __delattr__(self, attribute):
        # TODO: So far we're only deleting tables by name.
        #       Figure out what to do with aliases
        if not attribute.isupper():
            raise ValueError("Can't delete non-capitalized attributes")
        try:
            del self._tables[attribute]
            del self.__dict__[attribute]
        except KeyError:
            raise AttributeError("'{}' is not a global table for this instance".format(attribute))

    def _oper(self, operator, operand, indices=None):
        if indices is None:
            indices = tuple(range(len(self._nddata)))
        if isinstance(operand, AstroData):
            if len(operand) != len(indices):
                raise ValueError("Operands are not the same size")
            for n in indices:
                try:
                    self._set_nddata(n, operator(self._nddata[n],
                                                (operand.nddata if operand.is_single else operand.nddata[n])))
                except TypeError:
                    # This may happen if operand is a sliced, single AstroData object
                    self._set_nddata(n, operator(self._nddata[n], operand.nddata))
            op_table = operand.table()
            ltab, rtab = set(self._tables), set(op_table)
            for tab in (rtab - ltab):
                self._tables[tab] = op_table[tab]
        else:
            for n in indices:
                self._set_nddata(n, operator(self._nddata[n], operand))

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

    __itruediv__ = __idiv__

    def __rdiv__(self, operand):
        self._oper(self._rdiv, operand)
        return self

    def _rdiv(self, ndd, operand):
        # Divide method works with the operand first
        return NDDataObject.divide(operand, ndd)

    def set_phu(self, phu):
        self._phu = phu

    def info(self, tags, indices=None):
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
        if indices is None:
            indices = tuple(range(len(self._nddata)))
        if indices:
            main_fmt = "{:6} {:24} {:17} {:14} {}"
            other_fmt = "          .{:20} {:17} {:14} {}"
            print("\nPixels Extensions")
            print(main_fmt.format("Index", "Content", "Type", "Dimensions", "Format"))
            for pi in self._pixel_info(indices):
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

    def _pixel_info(self, indices):
        for idx, obj in ((n, self._nddata[k]) for (n, k) in enumerate(indices)):
            header = obj.meta['header']
            other_objects = []
            uncer = obj.uncertainty
            fixed = (('variance', None if uncer is None else uncer.as_variance()), ('mask', obj.mask))
            for name, other in fixed + tuple(sorted(obj.meta['other'].items())):
                if other is not None:
                    if isinstance(other, Table):
                        other_objects.append(dict(
                            attr=name, type='Table',
                            dim=str((len(other), len(other.columns))),
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
        return self._exposed.copy()

    def _slice(self, indices, multi=True):
        return FitsProviderProxy(self, indices, single=not multi)

    def __iter__(self):
        for n in range(len(self)):
            yield self._slice((n,), multi=False)

    def __getitem__(self, slc):
        nitems = len(self._nddata)
        indices, multiple = normalize_indices(slc, nitems=nitems)
        return self._slice(indices, multi=multiple)

    def __delitem__(self, idx):
        del self._nddata[idx]

    def __len__(self):
        return len(self._nddata)

    # NOTE: This one does not make reference to self at all. May as well
    #       move it out
    def _process_table(self, table, name=None, header=None):
        if isinstance(table, BinTableHDU):
            obj = Table(table.data, meta={'header': header or table.header})
            for i, col in enumerate(obj.columns, start=1):
                try:
                    obj[col].unit = u.Unit(obj.meta['header']['TUNIT{}'.format(i)])
                except (KeyError, TypeError):
                    pass
        elif isinstance(table, Table):
            obj = Table(table)
            if header is not None:
                obj.meta['header'] = deepcopy(header)
            elif 'header' not in obj.meta:
                obj.meta['header'] = header_for_table(obj)
        else:
            raise ValueError("{} is not a recognized table type".format(table.__class__))

        if name is not None:
            obj.meta['header']['EXTNAME'] = name

        return obj

    def _get_max_ver(self):
        try:
            return max(_nd.meta['ver'] for _nd in self._nddata) + 1
        except ValueError:
            # This seems to be the first extension!
            return 1

    def _reset_ver(self, nd):
        ver = self._get_max_ver()
        nd.meta['header']['EXTVER'] = ver
        nd.meta['ver'] = ver

        try:
            oheaders = nd.meta['other_header']
            for extname, ext in nd.meta['other'].items():
                try:
                    oheaders[extname]['EXTVER'] = ver
                except KeyError:
                    pass
                try:
                    # The object may keep the header on its own structure
                    ext.meta['header']['EXTVER'] = ver
                except AttributeError:
                    pass
        except KeyError:
            pass

        return ver

    def _process_pixel_plane(self, pixim, name=None, top_level=False, reset_ver=True, custom_header=None):
        if not isinstance(pixim, NDDataObject):
            # Assume that we get an ImageHDU or something that can be
            # turned into one
            if isinstance(pixim, ImageHDU):
                header = pixim.header
                nd = NDDataObject(pixim.data, meta={'header': header})
            elif custom_header is not None:
                header = custom_header
                nd = NDDataObject(pixim, meta={'header': custom_header})
            else:
                header = {}
                nd = NDDataObject(pixim, meta={})


            currname = header.get('EXTNAME')
            ver = header.get('EXTVER', -1)
        else:
            if custom_header is not None:
                pixim.meta['header'] = custom_header
            header = pixim.meta['header']
            nd = pixim
            currname = header.get('EXTNAME')
            ver = header.get('EXTVER', -1)

        # TODO: Review the logic. This one seems bogus
        if name and (currname is None):
            header['EXTNAME'] = (name if name is not None else FitsProvider.default_extension)

        if top_level:
            if 'other' not in nd.meta:
                nd.meta['other'] = OrderedDict()
                nd.meta['other_header'] = {}

            if reset_ver or ver == -1:
                self._reset_ver(nd)
            else:
                nd.meta['ver'] = ver

        return nd

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if self._path is None and value is not None:
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

    def _ext_header(self, obj):
        if isinstance(obj, int):
            # Assume that 'obj' is an index
            obj = self.nddata[obj]
        return obj.meta['header']

    def _get_raw_headers(self, with_phu=False, indices=None):
        if indices is None:
            indices = range(len(self.nddata))
        extensions = [self._ext_header(self.nddata[n]) for n in indices]

        if with_phu:
            return [self._phu] + extensions

        return extensions

    @property
    @deprecated("Access to headers through this property is deprecated and will be removed in the future")
    def header(self):
        return self._get_raw_headers(with_phu=True)

    @property
    def nddata(self):
        return self._nddata

    def phu(self):
        return self._phu

    def hdr(self):
        if not self.nddata:
            return None
        return FitsHeaderCollection(self._get_raw_headers())

    def set_name(self, ext, name):
        self._nddata[ext].meta['name'] = name

    def to_hdulist(self):

        hlst = HDUList()
        hlst.append(PrimaryHDU(header=self.phu(), data=DELAYED))

        for ext in self._nddata:
            meta = ext.meta
            header, ver = meta['header'], meta['ver']

            hlst.append(new_imagehdu(ext.data, header))
            if ext.uncertainty is not None:
                hlst.append(new_imagehdu(ext.uncertainty.array ** 2, header, 'VAR'))
            if ext.mask is not None:
                hlst.append(new_imagehdu(ext.mask, header, 'DQ'))

            for name, other in meta.get('other', {}).items():
                if isinstance(other, Table):
                    hlst.append(table_to_bintablehdu(other))
                elif isinstance(other, np.ndarray):
                    hlst.append(new_imagehdu(other, meta['other_header'].get(name, meta['header']), name=name))
                elif isinstance(other, NDDataObject):
                    hlst.append(new_imagehdu(other.data, meta['header']))
                else:
                    raise ValueError("I don't know how to write back an object of type {}".format(type(other)))

        if self._tables is not None:
            for name, table in sorted(self._tables.items()):
                hlst.append(table_to_bintablehdu(table, extname=name))

        return hlst

    def table(self):
        return self._tables.copy()

    @property
    def tables(self):
        return set(self._tables.keys())

    @property
    def shape(self):
        return [nd.shape for nd in self._nddata]

    @property
    def data(self):
        return [nd.data for nd in self._nddata]

    @data.setter
    def data(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    @property
    def uncertainty(self):
        return [nd.uncertainty for nd in self._nddata]

    @uncertainty.setter
    def uncertainty(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    @property
    def mask(self):
        return [nd.mask for nd in self._nddata]

    @mask.setter
    def mask(self, value):
        raise ValueError("Trying to assign to a non-sliced AstroData object")

    @property
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

    def _add_to_other(self, add_to, name, data, header=None):
        meta = add_to.meta
        meta['other'][name] = data
        if header:
            header['EXTVER'] = meta.get('ver', -1)
            meta['other_header'][name] = header

    def _append_array(self, data, name=None, header=None, add_to=None):
        def_ext = FitsProvider.default_extension
        # Top level:
        if add_to is None:
            # Special cases for Gemini
            if name is None:
                name = def_ext

            if name in {'DQ', 'VAR'}:
                raise ValueError("'{}' need to be associated to a '{}' one".format(name, def_ext))
            else:
                if name is not None:
                    hname = name
                elif header is not None:
                    hname = header.get('EXTNAME', def_ext)
                else:
                    hname = def_ext

                hdu = ImageHDU(data, header = header)
                hdu.header['EXTNAME'] = hname
                ret = self._append_imagehdu(hdu, name=hname, header=None, add_to=None)
        # Attaching to another extension
        else:
            if header is not None and name in {'DQ', 'VAR'}:
                LOGGER.warning("The header is ignored for '{}' extensions".format(name))
            if name is None:
                raise ValueError("Can't append pixel planes to other objects without a name")
            elif name is def_ext:
                raise ValueError("Can't attach '{}' arrays to other objects".format(def_ext))
            elif name == 'DQ':
                add_to.mask = data
                ret = data
            elif name == 'VAR':
                std_un = new_variance_uncertainty_instance(data)
                std_un.parent_nddata = add_to
                add_to.uncertainty = std_un
                ret = std_un
            else:
                self._add_to_other(add_to, name, data, header=header)
                ret = data

        return ret

    def _append_imagehdu(self, unit, name, header, add_to, reset_ver=True):
        if name in {'DQ', 'VAR'} or add_to is not None:
            return self._append_array(unit.data, name=name, add_to=add_to)
        else:
            nd = self._process_pixel_plane(unit, name=name, top_level=True, reset_ver=reset_ver,
                                           custom_header=header)
            return self._append_nddata(nd, name, add_to=None)

    def _append_raw_nddata(self, raw_nddata, name, header, add_to, reset_ver=True):
        # We want to make sure that the instance we add is whatever we specify as
        # `NDDataObject`, instead of the random one that the user may pass
        top_level = add_to is None
        if not isinstance(raw_nddata, NDDataObject):
            raw_nddata = NDDataObject(raw_nddata)
        processed_nddata = self._process_pixel_plane(raw_nddata, top_level=top_level,
                custom_header=header, reset_ver=reset_ver)
        return self._append_nddata(processed_nddata, name=name, add_to=add_to)

    # NOTE: This method is only used by others that have constructed NDData according
    #       to our internal format. We don't accept new headers at this point, and
    #       that's why it's missing from the signature
    def _append_nddata(self, new_nddata, name, add_to, reset_ver=True):
        # 'name' is ignored. It's there just to comply with the
        # _append_XXX signature
        def_ext = FitsProvider.default_extension
        if add_to is not None:
            raise TypeError("You can only append NDData derived instances at the top level")

        hd = new_nddata.meta['header']
        hname = hd.get('EXTNAME', def_ext)
        if hname == def_ext:
            if reset_ver:
                self._reset_ver(new_nddata)
            self._nddata.append(new_nddata)
        else:
            raise ValueError("Arbitrary image extensions can only be added in association to a '{}'".format(def_ext))

        return new_nddata

    def _set_nddata(self, n, new_nddata):
        self._nddata[n] = new_nddata

    def _append_table(self, new_table, name, header, add_to, reset_ver=True):
        tb = self._process_table(new_table, name, header)
        hname = tb.meta['header'].get('EXTNAME') if name is None else name
        #if hname is None:
        #    raise ValueError("Can't attach a table without a name!")
        if add_to is None:
            if hname is None:
                table_num = 1
                while 'TABLE{}'.format(table_num) in self._tables:
                    table_num += 1
                hname = 'TABLE{}'.format(table_num)
            # Don't use setattr, which is overloaded and may case problems
            self.__dict__[hname] = tb
            self._tables[hname] = tb
            self._exposed.add(hname)
        else:
            if hname is None:
                table_num = 1
                while getattr(add_to, 'TABLE{}'.format(table_num), None):
                    table_num += 1
                hname = 'TABLE{}'.format(table_num)
            setattr(add_to, hname, tb)
            self._add_to_other(add_to, hname, tb, tb.meta['header'])
            add_to.meta['other'][hname] = tb
        return tb

    def _append_astrodata(self, astrod, name, header, add_to, reset_ver=True):
        if not astrod.is_single:
            raise ValueError("Cannot append AstroData instances that are not single slices")
        elif add_to is not None:
            raise ValueError("Cannot append an AstroData slice to another slice")

        new_nddata = deepcopy(astrod.nddata)
        if header is not None:
            new_nddata.meta['header'] = deepcopy(header)

        return self._append_nddata(new_nddata, name=None, add_to=None, reset_ver=True)

    def append(self, ext, name=None, header=None, reset_ver=True, add_to=None):
        # NOTE: Most probably, if we want to copy the input argument, we
        #       should do it here...
        if isinstance(ext, PrimaryHDU):
            raise ValueError("Only one Primary HDU allowed. Use set_phu if you really need to set one")

        dispatcher = (
                (NDData, self._append_raw_nddata),
                ((Table, _TableBaseHDU), self._append_table),
                (ImageHDU, self._append_imagehdu),
                (AstroData, self._append_astrodata),
                )

        for bases, method in dispatcher:
            if isinstance(ext, bases):
                return method(ext, name=name, header=header, add_to=add_to, reset_ver=reset_ver)
        else:
            # Assume that this is an array for a pixel plane
            return self._append_array(ext, name=name, header=header, add_to=add_to)

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

def fits_ext_comp_key(ext):
    """
    Returns a pair (integer, string) that will be used to sort extensions
    """
    if isinstance(ext, PrimaryHDU):
        # This will guarantee that the primary HDU goes first
        ret = (-1, "")
    else:
        header = ext.header
        ver = header.get('EXTVER')

        # When two extensions share version number, we'll use their names
        # to sort them out. Choose a suitable key so that:
        #
        #  - SCI extensions come first
        #  - unnamed extensions come last
        #
        # We'll resort to add 'z' in front of the usual name to force
        # SCI to be the "smallest"
        name = header.get('EXTNAME') # Make sure that the name is a string
        if name is None:
            name = "zzzz"
        elif name != FitsProvider.default_extension:
            name = "z" + name

        if ver in (-1, None):
            # In practice, this number should be larger than any
            # EXTVER found in real life HDUs, pushing unnumbered
            # HDUs to the end
            ret = (2**32-1, name)
        else:
            # For the general case, just return version and name, to let them
            # be sorted naturally
            ret = (ver, name)

    return ret


class FitsLazyLoadable(object):

    def __init__(self, obj):
        self._obj = obj
        self.lazy = True

    def _create_result(self, shape):
        return np.empty(shape, dtype=self.dtype)

    def _scale(self, data):
        bscale = self._obj._orig_bscale
        bzero = self._obj._orig_bzero
        if bscale == 1 and bzero == 0:
            return data
        return (bscale * data + bzero).astype(self.dtype)

    def __getitem__(self, sl):
        # TODO: We may want (read: should) create an empty result array before scaling
        return self._scale(self._obj.section[sl])

    @property
    def header(self):
        return self._obj.header

    @property
    def data(self):
        res = self._create_result(self.shape)
        res[:] = self._scale(self._obj.data)
        return res

    @property
    def shape(self):
        return self._obj.shape

    @property
    def dtype(self):
        """
        Need to to some overriding of astropy.io.fits since it doesn't
        know about BITPIX=8
        """
        dtype = self._obj._dtype_for_bitpix()
        bitpix = self._obj._orig_bitpix
        if dtype is None:
            if bitpix < 0:
                dtype = np.dtype('float{}'.format(abs(bitpix)))
        if (self._obj.header['EXTNAME'] == 'DQ' or self._obj._uint and
                self._obj._orig_bscale == 1 and bitpix == 8):
            dtype = np.uint16
        return dtype


class FitsLoader(object):

    def __init__(self, cls = FitsProvider):
        self._cls = cls

    @staticmethod
    def _prepare_hdulist(hdulist, default_extension='SCI', extname_parser=None):
        new_list = []
        highest_ver = 0
        recognized = set()

        if len(hdulist) > 1 or (len(hdulist) == 1 and hdulist[0].data is None):
            # MEF file
            for n, unit in enumerate(hdulist):
                if extname_parser:
                    extname_parser(unit)
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
                        unit.header['EXTNAME'] = (default_extension, 'Added by AstroData')
                    if unit.header.get('EXTVER') in (-1, None):
                        unit.header['EXTVER'] = (highest_ver, 'Added by AstroData')

                new_list.append(unit)
                recognized.add(unit)
        else:
            # Uh-oh, a single image FITS file
            new_list.append(PrimaryHDU(header=hdulist[0].header))
            image = ImageHDU(header=hdulist[0].header, data=hdulist[0].data)
            # Fudge due to apparent issues with assigning ImageHDU from data
            image._orig_bscale = hdulist[0]._orig_bscale
            image._orig_bzero = hdulist[0]._orig_bzero

            for keyw in ('SIMPLE', 'EXTEND'):
                if keyw in image.header:
                    del image.header[keyw]
            # TODO: Remove self here (static method)
            image.header['EXTNAME'] = (default_extension, 'Added by AstroData')
            image.header['EXTVER'] = (1, 'Added by AstroData')
            new_list.append(image)

        return HDUList(sorted(new_list, key=fits_ext_comp_key))

    def load(self, source, extname_parser=None):
        """
        Takes either a string (with the path to a file) or an HDUList as input, and
        tries to return a populated FitsProvider (or descendant) instance.

        It will raise exceptions if the file is not found, or if there is no match
        for the HDUList, among the registered AstroData classes.
        """

        provider = self._cls()

        if isinstance(source, (str if PY3 else basestring)):
            hdulist = fits.open(source, memmap=True,
                                do_not_scale_image_data=True, mode='readonly')
            provider.path = source
        else:
            hdulist = source
            try:
                provider.path = source[0].header.get('ORIGNAME')
            except AttributeError:
                provider.path = None

        def_ext = self._cls.default_extension
        _file = hdulist._file
        hdulist = self._prepare_hdulist(hdulist, default_extension=def_ext,
                                        extname_parser=extname_parser)
        if _file is not None:
            hdulist._file = _file

        # Initialize the object containers to a bare minimum
        if 'ORIGNAME' not in hdulist[0].header and provider.orig_filename is not None:
            hdulist[0].header.set('ORIGNAME', provider.orig_filename,
                                  'Original filename prior to processing')
        provider.set_phu(hdulist[0].header)

        seen = set([hdulist[0]])

        skip_names = set([def_ext, 'REFCAT', 'MDF'])

        def associated_extensions(ver):
            for unit in hdulist:
                header = unit.header
                if header.get('EXTVER') == ver and header['EXTNAME'] not in skip_names:
                    yield unit

        sci_units = [x for x in hdulist[1:] if x.header.get('EXTNAME') == def_ext]

        for idx, unit in enumerate(sci_units):
            seen.add(unit)
            ver = unit.header.get('EXTVER', -1)
            parts = {'data': unit, 'uncertainty': None, 'mask': None, 'other': []}

            for extra_unit in associated_extensions(ver):
                seen.add(extra_unit)
                name = extra_unit.header.get('EXTNAME')
                if name == 'DQ':
                    parts['mask'] = extra_unit
                elif name == 'VAR':
                    parts['uncertainty'] = extra_unit
                else:
                    parts['other'].append(extra_unit)

            if hdulist._file is not None and hdulist._file.memmap:
                nd = NDDataObject(
                        data = FitsLazyLoadable(parts['data']),
                        uncertainty = None if parts['uncertainty'] is None else FitsLazyLoadable(parts['uncertainty']),
                        mask = None if parts['mask'] is None else FitsLazyLoadable(parts['mask'])
                        )
                provider.append(nd, name=def_ext, reset_ver=False)
            else:
                nd = provider.append(parts['data'], name=def_ext, reset_ver=False)
                for item_name in ('mask', 'uncertainty'):
                    item = parts[item_name]
                    if item is not None:
                        provider.append(item, name=item.header['EXTNAME'], add_to=nd)

            for other in parts['other']:
                provider.append(other, name=other.header['EXTNAME'], add_to=nd)

        for other in hdulist:
            if other in seen:
                continue
            name = other.header.get('EXTNAME')
            try:
                added = provider.append(other, name=name, reset_ver=False)
            except ValueError as e:
                print(str(e)+". Discarding "+name)

        return provider

def windowedOp(fn, sequence, kernel, shape=None, dtype=None, with_uncertainty=False, with_mask=False):
    def generate_boxes(shape, kernel):
        if len(shape) != len(kernel):
            raise AssertionError("Incompatible shape ({}) and kernel ({})".format(shape, kernel))
        ticks = [[(x, x+step) for x in range(0, axis, step)]
                 for axis, step in zip(shape, kernel)]
        return cart_product(*ticks)

    if shape is None:
        if len(set(x.shape for x in sequence)) > 1:
            raise ValueError("Can't calculate final shape: sequence elements disagree on shape, and none was provided")
        shape = sequence[0].shape

    if dtype is None:
        dtype = sequence[0].window[:1,:1].data.dtype

    result = NDDataObject(np.empty(shape, dtype=dtype),
                          uncertainty=(new_variance_uncertainty_instance(np.zeros(shape, dtype=dtype))
                                       if with_uncertainty else None),
                          mask=(np.empty(shape, dtype=np.uint16) if with_mask else None),
                          meta=sequence[0].meta)
    # Delete other extensions because we don't know what to do with them
    result.meta['other'] = OrderedDict()
    result.meta['other_header'] = {}

    # The Astropy logger's "INFO" messages aren't warnings, so have to fudge
    log_level = astropy.logger.conf.log_level
    astropy.log.setLevel(astropy.logger.WARNING)
    for coords in generate_boxes(shape, kernel):
        # The coordinates come as ((x1, x2, ..., xn), (y1, y2, ..., yn), ...)
        # Zipping them will get us a more desirable ((x1, y1, ...), (x2, y2, ...), ..., (xn, yn, ...))
        # box = list(zip(*coords))
        section = tuple([slice(start, end) for (start, end) in coords])
        result.set_section(section, fn((element.window[section] for element in sequence)))
        gc.collect()
    astropy.log.setLevel(log_level)  # and reset

    return result

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

    @classmethod
    def load(cls, source):
        """
        Implementation of the abstract method `load`.

        It takes an `HDUList` and returns a fully instantiated `AstroData` instance.
        """

        return cls(FitsLoader(FitsProvider).load(source))

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
        return self._dataprov.phu()

    @property
    def hdr(self):
        return self._dataprov.hdr()

    def info(self):
        self._dataprov.info(self.tags)

    def write(self, filename=None, overwrite=False):
        if filename is None:
            if self.path is None:
                raise ValueError("A filename needs to be specified")
            filename = self.path

        # Cope with astropy v1 and v2; can't use inspect because the
        # writeto() method is decorated in astropy v2+
        hdulist = self._dataprov.to_hdulist()
        try:
            hdulist.writeto(filename, overwrite=overwrite)
        except TypeError:
            hdulist.writeto(filename, clobber=overwrite)

    def update_filename(self, prefix=None, suffix=None, strip=False):
        """
        This method updates the "filename" attribute of the AstroData object.
        A prefix and/or suffix can be specified. If strip=True, these will
        replace the existing prefix/suffix; if strip=False, they will simply
        be prepended/appended.

        The current filename is broken down into its existing prefix, root,
        and suffix using the ORIGNAME phu keyword, if it exists and is
        contained within the current filename. Otherwise, the filename is
        split at the last underscore and the part before is assigned as the
        root and the underscore and part after the suffix. No prefix is
        assigned.

        Note that, if strip=True, a prefix or suffix will only be stripped
        if '' is specified.

        Parameters
        ----------
        prefix: str/None
            new prefix (None => leave alone)
        suffix: str/None
            new suffix (None => leave alone)
        strip: bool
            Strip existing prefixes and suffixes if new ones are given?
        """
        if self.filename is None:
            if 'ORIGNAME' in self.phu:
                self.filename = self.phu['ORIGNAME']
            else:
                raise ValueError("A filename needs to be set before it "
                                 "can be updated")

        # Set the ORIGNAME keyword if it's not there
        if 'ORIGNAME' not in self.phu:
            self.phu.set('ORIGNAME', self.orig_filename,
                         'Original filename prior to processing')

        if strip:
            root, filetype = os.path.splitext(self.phu['ORIGNAME'])
            filename, filetype = os.path.splitext(self.filename)
            m = re.match('(.*){}(.*)'.format(re.escape(root)), filename)
            # Do not strip a prefix/suffix unless a new one is provided
            if m:
                if prefix is None:
                    prefix = m.groups()[0]
                existing_suffix = m.groups()[1]
            else:
                try:
                    root, existing_suffix = filename.rsplit("_", 1)
                    existing_suffix = "_" + existing_suffix
                except ValueError:
                    root, existing_suffix = filename, ''
            if suffix is None:
                suffix = existing_suffix
        else:
            root, filetype = os.path.splitext(self.filename)

        # Cope with prefix or suffix as None
        self.filename = (prefix or '') + root + (suffix or '') + filetype
        return

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
