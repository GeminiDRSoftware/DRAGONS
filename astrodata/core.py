import inspect
import logging
import os
import re
import textwrap
from collections import OrderedDict
from contextlib import suppress
from copy import deepcopy
from functools import partial

import numpy as np

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.utils import format_doc

from .fits import (DEFAULT_EXTENSION, FitsHeaderCollection, _process_table,
                   read_fits, write_fits)
from .nddata import ADVarianceUncertainty
from .nddata import NDAstroData as NDDataObject
from .utils import (assign_only_single_slice, astro_data_descriptor,
                    deprecated, normalize_indices, returns_list)

NO_DEFAULT = object()


_arit_doc = """
    Performs {name} by evaluating ``self {op} operand``.

    Parameters
    ----------
    oper : number or object
        The operand to perform the operation  ``self {op} operand``.

    Returns
    --------
    `AstroData` instance
"""


class AstroData:
    """
    Base class for the AstroData software package. It provides an interface
    to manipulate astronomical data sets.

    Parameters
    ----------
    nddata : `astrodata.NDAstroData` or list of `astrodata.NDAstroData`
        List of NDAstroData objects.
    tables : dict[name, `astropy.table.Table`]
        Dict of table objects.
    phu : `astropy.io.fits.Header`
        Primary header.
    indices : list of int
        List of indices mapping the `astrodata.NDAstroData` objects that this
        object will access to. This is used when slicing an object, then the
        sliced AstroData will have the ``.nddata`` list from its parent and
        access the sliced NDAstroData through this list of indices.

    """

    # Derived classes may provide their own __keyword_dict. Being a private
    # variable, each class will preserve its own, and there's no risk of
    # overriding the whole thing
    __keyword_dict = {
        'instrument': 'INSTRUME',
        'object': 'OBJECT',
        'telescope': 'TELESCOP',
        'ut_date': 'DATE-OBS'
    }

    def __init__(self, nddata=None, tables=None, phu=None, indices=None):
        if nddata is None:
            nddata = []
        if not isinstance(nddata, (list, tuple)):
            nddata = list(nddata)

        # _all_nddatas contains all the extensions from the original file or
        # object.  And _indices is used to map extensions for sliced objects.
        self._all_nddatas = nddata
        self._indices = indices

        if tables is not None and not isinstance(tables, dict):
            raise ValueError('tables must be a dict')
        self._tables = tables or {}

        self._phu = phu or fits.Header()
        self._exposed = set()
        self._fixed_settable = {'data', 'uncertainty', 'mask', 'variance',
                                'wcs', 'path', 'filename'}
        self._logger = logging.getLogger(__name__)
        self._orig_filename = None
        self._path = None

    def __deepcopy__(self, memo):
        """
        Returns a new instance of this class.

        Parameters
        ----------
        memo : dict
            See the documentation on `deepcopy` for an explanation on how
            this works.

        """
        # Force the data provider to load data, if needed.
        # FIXME: probably no more needed
        len(self)

        obj = self.__class__()

        for attr in ('_phu', '_path', '_orig_filename', '_tables', '_exposed'):
            obj.__dict__[attr] = deepcopy(self.__dict__[attr])

        obj.__dict__['_all_nddatas'] = [deepcopy(nd) for nd in self._nddata]

        # Top-level tables
        for key in set(self.__dict__) - set(obj.__dict__):
            obj.__dict__[key] = obj.__dict__['_tables'][key]

        # FIXME: this was used by FitsProviderProxy, not sure which way is best
        # for nd in self.nddata:
        #     obj.append(deepcopy(nd))
        # for t in self._tables.values():
        #     obj.append(deepcopy(t))

        return obj

    def _keyword_for(self, name):
        """
        Returns the FITS keyword name associated to ``name``.

        Parameters
        ----------
        name : str
            The common "key" name for which we want to know the associated
            FITS keyword.

        Returns
        -------
        str
            The desired keyword name.

        Raises
        ------
        AttributeError
            If there is no keyword for the specified ``name``.

        """
        for cls in self.__class__.mro():
            with suppress(AttributeError, KeyError):
                mangled_dict_name = '_{}__keyword_dict'.format(cls.__name__)
                return getattr(self, mangled_dict_name)[name]
        else:
            raise AttributeError("No match for '{}'".format(name))

    def _process_tags(self):
        """
        Determines the tag set for the current instance.

        Returns
        -------
        set of str

        """
        results = []
        # Calling inspect.getmembers on `self` would trigger all the
        # properties (tags, phu, hdr, etc.), and that's undesirable. To
        # prevent that, we'll inspect the *class*.
        filt = lambda x: hasattr(x, 'tag_method')
        for _, method in inspect.getmembers(self.__class__, filt):
            ts = method(self)
            if ts.add or ts.remove or ts.blocks:
                results.append(ts)

        # Sort by the length of substractions... those that substract
        # from others go first
        results = sorted(results, key=lambda x: len(x.remove) + len(x.blocks),
                         reverse=True)

        # Sort by length of blocked_by, those that are never disabled go first
        results = sorted(results, key=lambda x: len(x.blocked_by))

        # Sort by length of if_present... those that need other tags to
        # be present go last
        results = sorted(results, key=lambda x: len(x.if_present))

        tags = set()
        removals = set()
        blocked = set()
        for plus, minus, blocked_by, blocks, is_present in results:
            if is_present:
                # If this TagSet requires other tags to be present, make
                # sure that all of them are. Otherwise, skip...
                if len(tags & is_present) != len(is_present):
                    continue
            allowed = (len(tags & blocked_by) + len(plus & blocked)) == 0
            if allowed:
                # This set is not being blocked by others...
                removals.update(minus)
                tags.update(plus - removals)
                blocked.update(blocks)

        return tags

    @staticmethod
    def _matches_data(source):
        # This one is trivial. Will be more specific for subclasses.
        return True

    @property
    def path(self):
        """Return the file path."""
        return self._path

    @path.setter
    def path(self, value):
        if self._path is None and value is not None:
            self._orig_filename = os.path.basename(value)
        self._path = value

    @property
    def filename(self):
        """Return the file name."""
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
        """Return the original file name (before it was modified)."""
        return self._orig_filename

    @orig_filename.setter
    def orig_filename(self, value):
        self._orig_filename = value

    @property
    def phu(self):
        """Return the primary header."""
        return self._phu

    @phu.setter
    def phu(self, phu):
        self._phu = phu

    @property
    def hdr(self):
        """Return all headers, as a `astrodata.fits.FitsHeaderCollection`."""
        if not self.nddata:
            return None
        headers = [nd.meta['header'] for nd in self._nddata]
        return headers[0] if self.is_single else FitsHeaderCollection(headers)

    @property
    @deprecated("Access to headers through this property is deprecated and "
                "will be removed in the future. Use '.hdr' instead.")
    def header(self):
        return [self.phu] + [ndd.meta['header'] for ndd in self._nddata]

    @property
    def tags(self):
        """A set of strings that represent the tags defining this instance."""
        return self._process_tags()

    @property
    def descriptors(self):
        """
        Returns a sequence of names for the methods that have been
        decorated as descriptors.

        Returns
        --------
        tuple of str
        """
        members = inspect.getmembers(self.__class__,
                                     lambda x: hasattr(x, 'descriptor_method'))
        return tuple(mname for (mname, method) in members)

    @property
    def indices(self):
        """Returns the extensions indices for sliced objects."""
        return self._indices if self._indices else list(range(len(self)))

    @property
    def is_sliced(self):
        """
        If this data provider instance represents the whole dataset, return
        False. If it represents a slice out of the whole, return True.
        """
        return self._indices is not None

    @property
    def is_single(self):
        """
        If this data provider represents a single slice out of a whole dataset,
        return True. Otherwise, return False.
        """
        return self._indices is not None and len(self._indices) == 1

    def is_settable(self, attr):
        """Return True if the attribute is meant to be modified."""
        if self.is_sliced and attr in {'path', 'filename'}:
            return False
        return attr in self._fixed_settable or attr.isupper()

    @property
    def _nddata(self):
        """Return the list of `astrodata.NDAstroData` objects. Contrary to
        ``self.nddata`` this always returns a list.
        """
        if self._indices is not None:
            return [self._all_nddatas[i] for i in self._indices]
        else:
            return self._all_nddatas

    @property
    def nddata(self):
        """Return the list of `astrodata.NDAstroData` objects.

        If the `AstroData` object is sliced, this returns only the NDData
        objects of the sliced extensions. And if this is a single extension
        object, the NDData object is returned directly (i.e. not a list).

        """
        return self._nddata[0] if self.is_single else self._nddata

    def table(self):
        # FIXME: do we need this in addition to .tables ?
        return self._tables.copy()

    @property
    def tables(self):
        """Return the names of the `astropy.table.Table` objects."""
        return set(self._tables.keys())

    @property
    @returns_list
    def shape(self):
        return [nd.shape for nd in self._nddata]

    @property
    @returns_list
    def data(self):
        """
        A list of the arrays (or single array, if this is a single slice)
        corresponding to the science data attached to each extension.
        """
        return [nd.data for nd in self._nddata]

    @data.setter
    @assign_only_single_slice
    def data(self, value):
        # Setting the ._data in the NDData is a bit kludgy, but we're all
        # grown adults and know what we're doing, isn't it?
        if hasattr(value, 'shape'):
            self.nddata._data = value
        else:
            raise AttributeError("Trying to assign data to be something "
                                 "with no shape")

    @property
    @returns_list
    def uncertainty(self):
        """
        A list of the uncertainty objects (or a single object, if this is
        a single slice) attached to the science data, for each extension.

        The objects are instances of AstroPy's `astropy.nddata.NDUncertainty`,
        or `None` where no information is available.

        See also
        --------
        variance : The actual array supporting the uncertainty object.

        """
        return [nd.uncertainty for nd in self._nddata]

    @uncertainty.setter
    @assign_only_single_slice
    def uncertainty(self, value):
        self.nddata.uncertainty = value

    @property
    @returns_list
    def mask(self):
        """
        A list of the mask arrays (or a single array, if this is a single
        slice) attached to the science data, for each extension.

        For objects that miss a mask, `None` will be provided instead.
        """
        return [nd.mask for nd in self._nddata]

    @mask.setter
    @assign_only_single_slice
    def mask(self, value):
        self.nddata.mask = value

    @property
    @returns_list
    def variance(self):
        """
        A list of the variance arrays (or a single array, if this is a single
        slice) attached to the science data, for each extension.

        For objects that miss uncertainty information, `None` will be provided
        instead.

        See also
        ---------
        uncertainty : The uncertainty objects used under the hood.

        """
        return [nd.variance for nd in self._nddata]

    @variance.setter
    @assign_only_single_slice
    def variance(self, value):
        if value is None:
            self.nddata.uncertainty = None
        else:
            self.nddata.uncertainty = ADVarianceUncertainty(value)

    @property
    def wcs(self):
        """Returns the list of WCS objects for each extension."""
        if self.is_single:
            return self.nddata.wcs
        else:
            raise ValueError("Cannot return WCS for an AstroData object "
                             "that is not a single slice")

    @wcs.setter
    @assign_only_single_slice
    def wcs(self, value):
        self.nddata.wcs = value

    def __iter__(self):
        if self.is_single:
            yield self
        else:
            for n in range(len(self)):
                yield self[n]

    def __getitem__(self, idx):
        """
        Returns a sliced view of the instance. It supports the standard
        Python indexing syntax.

        Parameters
        ----------
        slice : int, `slice`
            An integer or an instance of a Python standard `slice` object

        Raises
        -------
        TypeError
            If trying to slice an object when it doesn't make sense (e.g.
            slicing a single slice)
        ValueError
            If `slice` does not belong to one of the recognized types
        IndexError
            If an index is out of range

        """
        if self.is_single:
            raise TypeError("Can't slice a single slice!")

        indices, multiple = normalize_indices(idx, nitems=len(self))
        if self._indices:
            indices = [self._indices[i] for i in indices]

        obj = self.__class__(self._all_nddatas, tables=self._tables,
                             phu=self.phu, indices=indices)
        obj._path = self.path
        obj._orig_filename = self.orig_filename
        obj._exposed = self._exposed
        # FIXME: tables are stored both in _tables and __dict__, not sure why,
        # we could probably get rid of this (and exposed as well?)
        for k in self._exposed:
            obj.__dict__[k] = self.__dict__[k]
        return obj

    def __delitem__(self, idx):
        """
        Called to implement deletion of ``self[idx]``.  Supports standard
        Python syntax (including negative indices).

        Parameters
        ----------
        idx : int
            This index represents the order of the element that you want
            to remove.

        Raises
        -------
        IndexError
            If `idx` is out of range.

        """
        if self.is_sliced:
            raise TypeError("Can't remove items from a sliced object")
        del self._all_nddatas[idx]

    def __getattr__(self, attribute):
        """
        Called when an attribute lookup has not found the attribute in the
        usual places (not an instance attribute, and not in the class tree
        for ``self``).

        Parameters
        ----------
        attribute : str
            The attribute's name.

        Raises
        -------
        AttributeError
            If the attribute could not be found/computed.

        """
        # Exposed objects are part of the normal object interface. We may have
        # just lazy-loaded them, and that's why we get here...
        if attribute in self._exposed:
            return self.__dict__[attribute]

        # Check if it's an aliased object
        for nd in self._nddata:
            if nd.meta.get('name') == attribute:
                return nd

        # I we're working with single slices, let's look some things up
        # in the ND object
        if self.is_single and attribute.isupper():
            try:
                return self.nddata.meta['other'][attribute]
            except KeyError:
                pass

        raise AttributeError("{!r} object has no attribute {!r}"
                             .format(self.__class__.__name__, attribute))

    def __setattr__(self, attribute, value):
        """
        Called when an attribute assignment is attempted, instead of the
        normal mechanism.

        Parameters
        ----------
        attribute : str
            The attribute's name.
        value : object
            The value to be assigned to the attribute.

        """

        def _my_attribute(attr):
            return attr in self.__dict__ or attr in self.__class__.__dict__

        if attribute.isupper() and not _my_attribute(attribute):
            # This method is meant to let the user set certain attributes of
            # the NDData objects. First we check if the attribute belongs to
            # this object's dictionary.  Otherwise, see if we can pass it down.
            #
            # CJS 20200131: if the attribute is "exposed" then we should set
            # it via the append method I think (it's a Table or something)
            if (self.is_settable(attribute) and
                    (not _my_attribute(attribute) or
                     attribute in self._exposed)):
                if self.is_sliced and not self.is_single:
                    raise TypeError("This attribute can only be "
                                    "assigned to a single-slice object")
                add_to = self.nddata[0] if self.is_sliced else None
                self.append(value, name=attribute, add_to=add_to)
                return

        super().__setattr__(attribute, value)

    def __delattr__(self, attribute):
        """Implements attribute removal."""
        if not attribute.isupper():
            super().__delattr__(attribute)
            return

        if self.is_sliced:
            if not self.is_single:
                raise TypeError("Can't delete attributes on non-single slices")

            other = self.nddata.meta['other']
            if attribute in other:
                del other[attribute]
                otherh = self.nddata.meta['other_header']
                if attribute in otherh:
                    del otherh[attribute]
            else:
                raise AttributeError(
                    "{!r} sliced object has no attribute {!r}"
                    .format(self.__class__.__name__, attribute))
        else:
            # TODO: So far we're only deleting tables by name.
            #       Figure out what to do with aliases
            if attribute in self._tables:
                del self._tables[attribute]
            else:
                raise AttributeError(
                    "'{}' is not a global table for this instance"
                    .format(attribute))

    def __contains__(self, attribute):
        """
        Implements the ability to use the ``in`` operator with an
        `AstroData` object.

        Parameters
        ----------
        attribute : str
            An attribute name.

        Returns
        --------
        bool
        """
        return attribute in self.exposed

    def __len__(self):
        """Return the number of independent extensions stored by the object.
        """
        if self._indices is not None:
            return len(self._indices)
        else:
            return len(self._all_nddatas)

    @property
    def exposed(self):
        """
        A collection of strings with the names of objects that can be accessed
        directly by name as attributes of this instance, and that are not part
        of its standard interface (i.e. data objects that have been added
        dynamically).

        Examples
        ---------
        >>> ad[0].exposed  # doctest: +SKIP
        set(['OBJMASK', 'OBJCAT'])

        """
        exposed = self._exposed.copy()
        if self.is_sliced:
            nd = self.nddata if self.is_single else self.nddata[0]
            exposed |= set(nd.meta['other'])
        return exposed

    def _pixel_info(self):
        for idx, nd in enumerate(self._nddata):
            other_objects = []
            uncer = nd.uncertainty
            fixed = (('variance', None if uncer is None else uncer),
                     ('mask', nd.mask))
            for name, other in fixed + tuple(sorted(nd.meta['other'].items())):
                if other is None:
                    continue
                if isinstance(other, Table):
                    other_objects.append(dict(
                        attr=name,
                        type='Table',
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
                        attr=name,
                        type=type(other).__name__,
                        dim=dim,
                        data_type=dt
                    ))

            yield dict(
                idx='[{:2}]'.format(idx),
                main=dict(
                    content='science',
                    type=type(nd).__name__,
                    dim=str(nd.data.shape),
                    data_type=nd.data.dtype.name
                ),
                other=other_objects
            )

    def info(self):
        """Prints out information about the contents of this instance."""

        print("Filename: {}".format(self.path if self.path else "Unknown"))
        # This is fixed. We don't support opening for update
        # print("Mode: readonly")

        text = 'Tags: ' + ' '.join(sorted(self.tags))
        textwrapper = textwrap.TextWrapper(width=80, subsequent_indent='    ')
        for line in textwrapper.wrap(text):
            print(line)

        if len(self) > 0:
            main_fmt = "{:6} {:24} {:17} {:14} {}"
            other_fmt = "          .{:20} {:17} {:14} {}"
            print("\nPixels Extensions")
            print(main_fmt.format("Index", "Content", "Type", "Dimensions",
                                  "Format"))
            for pi in self._pixel_info():
                main_obj = pi['main']
                print(main_fmt.format(
                    pi['idx'], main_obj['content'][:24], main_obj['type'][:17],
                    main_obj['dim'], main_obj['data_type']))
                for other in pi['other']:
                    print(other_fmt.format(
                        other['attr'][:20], other['type'][:17], other['dim'],
                        other['data_type']))

        # NOTE: This covers tables, only. Study other cases before
        # implementing a more general solution
        if self._tables:
            print("\nOther Extensions")
            print("               Type        Dimensions")
            for name, table in sorted(self._tables.items()):
                if type(table) is list:
                    # This is not a free floating table
                    continue
                print(".{:13} {:11} {}".format(
                    name[:13], 'Table', (len(table), len(table.columns))))

    def _oper(self, operator, operand):
        ind = self.indices
        ndd = self._all_nddatas
        if isinstance(operand, AstroData):
            if len(operand) != len(self):
                raise ValueError("Operands are not the same size")
            for n in range(len(self)):
                try:
                    data = (operand.nddata if operand.is_single
                            else operand.nddata[n])
                    ndd[ind[n]] = operator(ndd[ind[n]], data)
                except TypeError:
                    # This may happen if operand is a sliced, single
                    # AstroData object
                    ndd[ind[n]] = operator(ndd[ind[n]], operand.nddata)
            op_table = operand.table()
            ltab, rtab = set(self._tables), set(op_table)
            for tab in (rtab - ltab):
                self._tables[tab] = op_table[tab]
        else:
            for n in range(len(self)):
                ndd[ind[n]] = operator(ndd[ind[n]], operand)

    def _standard_nddata_op(self, fn, operand):
        return self._oper(partial(fn, handle_mask=np.bitwise_or,
                                  handle_meta='first_found'), operand)

    @format_doc(_arit_doc, name='addition', op='+')
    def __add__(self, oper):
        copy = deepcopy(self)
        copy += oper
        return copy

    @format_doc(_arit_doc, name='subtraction', op='-')
    def __sub__(self, oper):
        copy = deepcopy(self)
        copy -= oper
        return copy

    @format_doc(_arit_doc, name='multiplication', op='*')
    def __mul__(self, oper):
        copy = deepcopy(self)
        copy *= oper
        return copy

    @format_doc(_arit_doc, name='division', op='/')
    def __truediv__(self, oper):
        copy = deepcopy(self)
        copy /= oper
        return copy

    @format_doc(_arit_doc, name='inplace addition', op='+=')
    def __iadd__(self, oper):
        self._standard_nddata_op(NDDataObject.add, oper)
        return self

    @format_doc(_arit_doc, name='inplace subtraction', op='-=')
    def __isub__(self, oper):
        self._standard_nddata_op(NDDataObject.subtract, oper)
        return self

    @format_doc(_arit_doc, name='inplace multiplication', op='*=')
    def __imul__(self, oper):
        self._standard_nddata_op(NDDataObject.multiply, oper)
        return self

    @format_doc(_arit_doc, name='inplace division', op='/=')
    def __itruediv__(self, oper):
        self._standard_nddata_op(NDDataObject.divide, oper)
        return self

    add = __iadd__
    subtract = __isub__
    multiply = __imul__
    divide = __itruediv__

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, oper):
        copy = (deepcopy(self) - oper) * -1
        return copy

    def _rdiv(self, ndd, operand):
        # Divide method works with the operand first
        return NDDataObject.divide(operand, ndd)

    def __rtruediv__(self, oper):
        obj = deepcopy(self)
        obj._oper(obj._rdiv, oper)
        return obj

    def _reset_ver(self, nd):
        try:
            ver = max(nd.meta['ver'] for nd in self._all_nddatas) + 1
        except ValueError:
            # This seems to be the first extension!
            ver = 1

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

    def _process_pixel_plane(self, pixim, name=None, top_level=False,
                             reset_ver=True, custom_header=None):
        if not isinstance(pixim, NDDataObject):
            # Assume that we get an ImageHDU or something that can be
            # turned into one
            if isinstance(pixim, fits.ImageHDU):
                nd = NDDataObject(pixim.data, meta={'header': pixim.header})
            elif custom_header is not None:
                nd = NDDataObject(pixim, meta={'header': custom_header})
            else:
                nd = NDDataObject(pixim, meta={'header': {}})
        else:
            nd = pixim
            if custom_header is not None:
                nd.meta['header'] = custom_header

        header = nd.meta['header']
        currname = header.get('EXTNAME')
        ver = header.get('EXTVER', -1)

        # TODO: Review the logic. This one seems bogus
        if name and (currname is None):
            header['EXTNAME'] = name if name is not None else DEFAULT_EXTENSION

        if top_level:
            if 'other' not in nd.meta:
                nd.meta['other'] = OrderedDict()
                nd.meta['other_header'] = {}

            if reset_ver or ver == -1:
                self._reset_ver(nd)
            else:
                nd.meta['ver'] = ver

        return nd

    def _add_to_other(self, add_to, name, data, header=None):
        meta = add_to.meta
        meta['other'][name] = data
        if header:
            header['EXTVER'] = meta.get('ver', -1)
            meta['other_header'][name] = header

    def _append_array(self, data, name=None, header=None, add_to=None):
        if add_to is None:
            # Top level extension

            # Special cases for Gemini
            if name is None:
                name = DEFAULT_EXTENSION

            if name in {'DQ', 'VAR'}:
                raise ValueError("'{}' need to be associated to a '{}' one"
                                 .format(name, DEFAULT_EXTENSION))
            else:
                # FIXME: the logic here is broken since name is
                # always set to somehing above with DEFAULT_EXTENSION
                if name is not None:
                    hname = name
                elif header is not None:
                    hname = header.get('EXTNAME', DEFAULT_EXTENSION)
                else:
                    hname = DEFAULT_EXTENSION

                hdu = fits.ImageHDU(data, header=header)
                hdu.header['EXTNAME'] = hname
                ret = self._append_imagehdu(hdu, name=hname, header=None,
                                            add_to=None)
        else:
            # Attaching to another extension
            if header is not None and name in {'DQ', 'VAR'}:
                self._logger.warning(
                    "The header is ignored for '{}' extensions".format(name))
            if name is None:
                # FIXME: both should raise the same exception
                if self.is_sliced:
                    raise TypeError("Can't append objects to a slice "
                                    "without an extension name")
                else:
                    raise ValueError("Can't append pixel planes to other "
                                     "objects without a name")
            elif name is DEFAULT_EXTENSION:
                raise ValueError("Can't attach '{}' arrays to other objects"
                                 .format(DEFAULT_EXTENSION))
            elif name == 'DQ':
                add_to.mask = data
                ret = data
            elif name == 'VAR':
                std_un = ADVarianceUncertainty(data)
                std_un.parent_nddata = add_to
                add_to.uncertainty = std_un
                ret = std_un
            else:
                self._add_to_other(add_to, name, data, header=header)
                ret = data

        return ret

    def _append_imagehdu(self, hdu, name, header, add_to, reset_ver=True):
        if name in {'DQ', 'VAR'} or add_to is not None:
            return self._append_array(hdu.data, name=name, add_to=add_to)
        else:
            nd = self._process_pixel_plane(hdu, name=name, top_level=True,
                                           reset_ver=reset_ver,
                                           custom_header=header)
            return self._append_nddata(nd, name, add_to=None)

    def _append_raw_nddata(self, raw_nddata, name, header, add_to,
                           reset_ver=True):
        # We want to make sure that the instance we add is whatever we specify
        # as NDDataObject, instead of the random one that the user may pass
        top_level = add_to is None
        if not isinstance(raw_nddata, NDDataObject):
            raw_nddata = NDDataObject(raw_nddata)
        processed_nddata = self._process_pixel_plane(raw_nddata,
                                                     top_level=top_level,
                                                     custom_header=header,
                                                     reset_ver=reset_ver)
        return self._append_nddata(processed_nddata, name=name, add_to=add_to)

    def _append_nddata(self, new_nddata, name, add_to, reset_ver=True):
        # NOTE: This method is only used by others that have constructed NDData
        # according to our internal format. We don't accept new headers at this
        # point, and that's why it's missing from the signature.  'name' is
        # ignored. It's there just to comply with the _append_XXX signature.
        if add_to is not None:
            raise TypeError("You can only append NDData derived instances "
                            "at the top level")

        hd = new_nddata.meta['header']
        hname = hd.get('EXTNAME', DEFAULT_EXTENSION)
        if hname == DEFAULT_EXTENSION:
            if reset_ver:
                self._reset_ver(new_nddata)
            self._all_nddatas.append(new_nddata)
        else:
            raise ValueError("Arbitrary image extensions can only be added "
                             "in association to a '{}'"
                             .format(DEFAULT_EXTENSION))

        return new_nddata

    def _append_table(self, new_table, name, header, add_to, reset_ver=True):
        tb = _process_table(new_table, name, header)
        hname = tb.meta['header'].get('EXTNAME') if name is None else name
        # if hname is None:
        #     raise ValueError("Can't attach a table without a name!")
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

    def _append_astrodata(self, ad, name, header, add_to, reset_ver=True):
        if not ad.is_single:
            raise ValueError("Cannot append AstroData instances that are "
                             "not single slices")
        elif add_to is not None:
            raise ValueError("Cannot append an AstroData slice to "
                             "another slice")

        new_nddata = deepcopy(ad.nddata)
        if header is not None:
            new_nddata.meta['header'] = deepcopy(header)

        return self._append_nddata(new_nddata, name=None, add_to=None,
                                   reset_ver=True)

    def append(self, ext, name=None, header=None, reset_ver=True, add_to=None):
        """
        Adds a new top-level extension. Objects appended to a single
        slice will actually be made hierarchically dependent of the science
        object represented by that slice. If appended to the provider as
        a whole, the new member will be independent (eg. global table, new
        science object).

        Parameters
        ----------
        ext : array, `astropy.nddata.NDData`, `astropy.table.Table`, other
            The contents for the new extension. The exact accepted types depend
            on the class implementing this interface. Implementations specific
            to certain data formats may accept specialized types (eg. a FITS
            provider will accept an `astropy.io.fits.ImageHDU` and extract the
            array out of it).
        name : str, optional
            A name that may be used to access the new object, as an attribute
            of the provider. The name is typically ignored for top-level
            (global) objects, and required for the others. If the name cannot
            be derived from the metadata associated to ``ext``, you will
            have to provider one.
            It can consist in a combination of numbers and letters, with the
            restriction that the letters have to be all capital, and the first
            character cannot be a number ("[A-Z][A-Z0-9]*").

        Returns
        --------
        The same object, or a new one, if it was necessary to convert it to
        a more suitable format for internal use.

        Raises
        -------
        TypeError
            If adding the object in an invalid situation (eg. ``name`` is
            `None` when adding to a single slice).
        ValueError
            Raised if the extension is of a proper type, but its value is
            illegal somehow.

        """
        if self.is_sliced and not self.is_single:
            # TODO: We could rethink this one, but leave it like that at
            # the moment
            raise TypeError("Can't append objects to non-single slices")

        # FIXME: this was used on FitsProviderProxy:
        # elif name is None:
        #     raise TypeError("Can't append objects to a slice without an "
        #                     "extension name")

        # NOTE: Most probably, if we want to copy the input argument, we
        #       should do it here...
        if isinstance(ext, fits.PrimaryHDU):
            raise ValueError("Only one Primary HDU allowed. "
                             "Use .phu if you really need to set one")

        if self.is_sliced:
            add_to = self.nddata

        dispatcher = (
            (NDData, self._append_raw_nddata),
            ((Table, fits.TableHDU, fits.BinTableHDU), self._append_table),
            (fits.ImageHDU, self._append_imagehdu),
            (AstroData, self._append_astrodata),
        )

        for bases, method in dispatcher:
            if isinstance(ext, bases):
                return method(ext, name=name, header=header, add_to=add_to,
                              reset_ver=reset_ver)
        else:
            # Assume that this is an array for a pixel plane
            return self._append_array(ext, name=name, header=header,
                                      add_to=add_to)

    @classmethod
    def read(cls, source, extname_parser=None):
        """Read from a file, file object, HDUList, etc."""
        return read_fits(cls, source, extname_parser=extname_parser)

    load = read  # for backward compatibility

    def write(self, filename=None, overwrite=False):
        """
        Write the object to disk.

        Parameters
        ----------
        filename : str, optional
            If the filename is not given, ``self.path`` is used.
        overwrite : bool
            If True, overwrites existing file.

        """
        if filename is None:
            if self.path is None:
                raise ValueError("A filename needs to be specified")
            filename = self.path

        write_fits(self, filename, overwrite=overwrite)

    def extver_map(self):
        """
        Provide a mapping between the FITS EXTVER of an extension and the index
        that will be used to access it within this object.

        Returns
        -------
        A dictionary ``{EXTVER:index, ...}``

        Raises
        ------
        ValueError
            If used against a single slice. It is of no use in that situation.

        """
        if self.is_single:
            raise ValueError("Trying to get a mapping out of a single slice")
        return {nd.meta['ver']: n for (n, nd) in enumerate(self._nddata)}

    def extver(self, ver):
        """
        Get an extension using its EXTVER instead of the positional index
        in this object.

        Parameters
        ----------
        ver : int
            The EXTVER for the desired extension.

        Returns
        -------
        A sliced object containing the desired extension.

        Raises
        ------
        IndexError
            If the provided EXTVER doesn't exist.

        """
        try:
            if isinstance(ver, int):
                return self[self.extver_map()[ver]]
            else:
                raise ValueError("{} is not an integer EXTVER".format(ver))
        except KeyError as e:
            raise IndexError("EXTVER {} not found".format(e.args[0]))

    def operate(self, operator, *args, **kwargs):
        """
        Applies a function to the main data array on each extension, replacing
        the data with the result. The data will be passed as the first argument
        to the function.

        It will be applied to the mask and variance of each extension, too, if
        they exist.

        This is a convenience method, which is equivalent to::

            for ext in ad:
                ad.ext.data = operator(ad.ext.data, *args, **kwargs)
                if ad.ext.mask is not None:
                    ad.ext.mask = operator(ad.ext.mask, *args, **kwargs)
                if ad.ext.variance is not None:
                    ad.ext.variance = operator(ad.ext.variance, *args, **kwargs)

        with the additional advantage that it will work on single slices, too.

        Parameters
        ----------
        operator : callable
            A function that takes an array (and, maybe, other arguments)
            and returns an array.
        args, kwargs : optional
            Additional arguments to be passed to the ``operator``.

        Examples
        ---------
        >>> import numpy as np
        >>> ad.operate(np.squeeze)  # doctest: +SKIP

        """
        # Ensure we can iterate, even on a single slice
        for ext in [self] if self.is_single else self:
            ext.data = operator(ext.data, *args, **kwargs)
            if ext.mask is not None:
                ext.mask = operator(ext.mask, *args, **kwargs)
            if ext.variance is not None:
                ext.variance = operator(ext.variance, *args, **kwargs)

    def reset(self, data, mask=NO_DEFAULT, variance=NO_DEFAULT, check=True):
        """
        Sets the ``.data``, and optionally ``.mask`` and ``.variance``
        attributes of a single-extension AstroData slice. This function will
        optionally check whether these attributes have the same shape.

        Parameters
        ----------
        data : ndarray
            The array to assign to the ``.data`` attribute ("SCI").
        mask : ndarray, optional
            The array to assign to the ``.mask`` attribute ("DQ").
        variance: ndarray, optional
            The array to assign to the ``.variance`` attribute ("VAR").
        check: bool
            If set, then the function will check that the mask and variance
            arrays have the same shape as the data array.

        Raises
        -------
        TypeError
            if an attempt is made to set the .mask or .variance attributes
            with something other than an array
        ValueError
            if the .mask or .variance attributes don't have the same shape as
            .data, OR if this is called on an AD instance that isn't a single
            extension slice

        """
        if not self.is_single:
            raise ValueError("Trying to reset a non-sliced AstroData object")

        # In case data is an NDData object
        try:
            self.data = data.data
        except AttributeError:
            self.data = data
        # Set mask, with checking if required
        try:
            if mask.shape != self.data.shape and check:
                raise ValueError("Mask shape incompatible with data shape")
        except AttributeError:
            if mask is None:
                self.mask = mask
            elif mask == NO_DEFAULT:
                if hasattr(data, 'mask'):
                    self.mask = data.mask
            else:
                raise TypeError("Attempt to set mask inappropriately")
        else:
            self.mask = mask
        # Set variance, with checking if required
        try:
            if variance.shape != self.data.shape and check:
                raise ValueError("Variance shape incompatible with data shape")
        except AttributeError:
            if variance is None:
                self.uncertainty = None
            elif variance == NO_DEFAULT:
                if hasattr(data, 'uncertainty'):
                    self.uncertainty = data.uncertainty
            else:
                raise TypeError("Attempt to set variance inappropriately")
        else:
            self.variance = variance

        if hasattr(data, 'wcs'):
            self.wcs = data.wcs

    def update_filename(self, prefix=None, suffix=None, strip=False):
        """
        Update the "filename" attribute of the AstroData object.

        A prefix and/or suffix can be specified. If ``strip=True``, these will
        replace the existing prefix/suffix; if ``strip=False``, they will
        simply be prepended/appended.

        The current filename is broken down into its existing prefix, root,
        and suffix using the ``ORIGNAME`` phu keyword, if it exists and is
        contained within the current filename. Otherwise, the filename is
        split at the last underscore and the part before is assigned as the
        root and the underscore and part after the suffix. No prefix is
        assigned.

        Note that, if ``strip=True``, a prefix or suffix will only be stripped
        if '' is specified.

        Parameters
        ----------
        prefix: str, optional
            New prefix (None => leave alone)
        suffix: str, optional
            New suffix (None => leave alone)
        strip: bool, optional
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

    def _crop_nd(self, nd, x1, y1, x2, y2):
        nd.data = nd.data[y1:y2+1, x1:x2+1]
        if nd.uncertainty is not None:
            nd.uncertainty = nd.uncertainty[y1:y2+1, x1:x2+1]
        if nd.mask is not None:
            nd.mask = nd.mask[y1:y2+1, x1:x2+1]

    def crop(self, x1, y1, x2, y2):
        """Crop the NDData objects given indices.

        Parameters
        ----------
        x1, y1, x2, y2 : int
            Minimum and maximum indices for the x and y axis.

        """
        # TODO: Consider cropping of objects in the meta section
        for nd in self._nddata:
            orig_shape = nd.data.shape
            self._crop_nd(nd, x1, y1, x2, y2)
            for o in nd.meta['other'].values():
                try:
                    if o.shape == orig_shape:
                        self._crop_nd(o, x1, y1, x2, y2)
                except AttributeError:
                    # No 'shape' attribute in the object. It's probably
                    # not array-like
                    pass

    @astro_data_descriptor
    def instrument(self):
        """Returns the name of the instrument making the observation."""
        return self.phu.get(self._keyword_for('instrument'))

    @astro_data_descriptor
    def object(self):
        """Returns the name of the object being observed."""
        return self.phu.get(self._keyword_for('object'))

    @astro_data_descriptor
    def telescope(self):
        """Returns the name of the telescope."""
        return self.phu.get(self._keyword_for('telescope'))
