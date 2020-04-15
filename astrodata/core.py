import inspect
import logging
import os
import re
from collections import namedtuple, OrderedDict
from contextlib import suppress
from copy import deepcopy
from functools import partial, wraps

import numpy as np

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

from .fits import DEFAULT_EXTENSION, _process_table, read_fits, write_fits
from .nddata import ADVarianceUncertainty
from .nddata import NDAstroData as NDDataObject
from .utils import deprecated, normalize_indices

NO_DEFAULT = object()


class TagSet(namedtuple('TagSet', 'add remove blocked_by blocks if_present')):
    """
    TagSet(add=None, remove=None, blocked_by=None, blocks=None, if_present=None)

    Named tuple that is used by tag methods to return which actions should be
    performed on a tag set. All the attributes are optional, and any
    combination of them can be used, allowing to create complex tag structures.
    Read the documentation on the tag-generating algorithm if you want to
    better understand the interactions.

    The simplest TagSet, though, tends to just add tags to the global set.

    It can be initialized by position, like any other tuple (the order of the
    arguments is the one in which the attributes are listed below). It can
    also be initialized by name.

    Attributes
    ----------
    add : set of str, or None
        Tags to be added to the global set
    remove : set of str, or None
        Tags to be removed from the global set
    blocked_by : set of str, or None
        Tags that will prevent this TagSet from being applied
    blocks : set of str, or None
        Other TagSets containing these won't be applied
    if_present : set of str, or None
        This TagSet will be applied only *all* of these tags are present

    Examples
    ---------
    >>> TagSet()
    TagSet(add=set(), remove=set(), blocked_by=set(), blocks=set(), if_present=set())
    >>> TagSet({'BIAS', 'CAL'})
    TagSet(add={'BIAS', 'CAL'}, remove=set(), blocked_by=set(), blocks=set(), if_present=set())
    >>> TagSet(remove={'BIAS', 'CAL'})
    TagSet(add=set(), remove={'BIAS', 'CAL'}, blocked_by=set(), blocks=set(), if_present=set())

    """
    def __new__(cls, add=None, remove=None, blocked_by=None, blocks=None, if_present=None):
        return super().__new__(cls, add or set(),
                               remove or set(),
                               blocked_by or set(),
                               blocks or set(),
                               if_present or set())


def astro_data_descriptor(fn):
    """
    Decorator that will mark a class method as an AstroData descriptor.
    Useful to produce list of descriptors, for example.

    If used in combination with other decorators, this one *must* be the
    one on the top (ie. the last one applying). It doesn't modify the
    method in any other way.

    Args
    -----
    fn : method
        The method to be decorated

    Returns
    --------
    The tagged method (not a wrapper)
    """
    fn.descriptor_method = True
    return fn


def returns_list(fn):
    """
    Decorator to ensure that descriptors that should return a list (of one
    value per extension) only returns single values when operating on
    single slices; and vice versa.

    This is a common case, and you can use the decorator to simplify the
    logic of your descriptors.

    Args
    -----
    fn : method
        The method to be decorated

    Returns
    --------
    A function
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        ret = fn(self, *args, **kwargs)
        if self.is_single:
            if isinstance(ret, list):
                # TODO: log a warning if the list is >1 element
                if len(ret) > 1:
                    pass
                return ret[0]
            else:
                return ret
        else:
            if isinstance(ret, list):
                if len(ret) == len(self):
                    return ret
                else:
                    raise IndexError(
                        "Incompatible numbers of extensions and elements in {}"
                        .format(fn.__name__))
            else:
                return [ret] * len(self)
    return wrapper


def astro_data_tag(fn):
    """
    Decorator that marks methods of an `AstroData` derived class as part of the
    tag-producing system.

    It wraps the method around a function that will ensure a consistent return
    value: the wrapped method can return any sequence of sequences of strings,
    and they will be converted to a TagSet. If the wrapped method
    returns None, it will be turned into an empty TagSet.

    Args
    -----
    fn : method
        The method to be decorated

    Returns
    --------
    A wrapper function
    """
    @wraps(fn)
    def wrapper(self):
        try:
            ret = fn(self)
            if ret is not None:
                if not isinstance(ret, TagSet):
                    raise TypeError("Tag function {} didn't return a TagSet"
                                    .format(fn.__name__))

                return TagSet(*tuple(set(s) for s in ret))
        except KeyError:
            pass

        # Return empty TagSet for the "doesn't apply" case
        return TagSet()

    wrapper.tag_method = True
    return wrapper


class AstroDataError(Exception):
    pass


class AstroData:
    """
    AstroData(provider)

    Base class for the AstroData software package. It provides an interface
    to manipulate astronomical data sets.

    Parameters
    -----------
    provider : DataProvider
        The data that will be manipulated through the `AstroData` instance.
    """

    # Derived classes may provide their own _keyword_dict. Being a private
    # variable, each class will preserve its own, and there's no risk of
    # overriding the whole thing
    _keyword_dict = {
        'instrument': 'INSTRUME',
        'object': 'OBJECT',
        'telescope': 'TELESCOP',
        'ut_date': 'DATE-OBS'
    }

    def __init__(self, nddatas=None, other=None, phu=None, indices=None):
        if nddatas is None:
            nddatas = []
        if not isinstance(nddatas, (list, tuple)):
            nddatas = list(nddatas)

        self._nddata = nddatas
        self._other = other
        self._phu = phu or fits.Header()
        self._indices = indices

        self._exposed = set()
        self._fixed_settable = {'data', 'uncertainty', 'mask', 'variance',
                                'wcs', 'path', 'filename'}
        self._logger = logging.getLogger(__name__)
        self._orig_filename = None
        self._path = None
        self._processing_tags = False
        self._resetting = False
        self._tables = {}

    def _clone(self):
        # FIXME: this was used by FitsProviderProxy
        obj = self.__class__()
        obj._phu = deepcopy(self._phu)
        for nd in self.nddata:
            obj.append(deepcopy(nd))
        for t in self._tables.values():
            obj.append(deepcopy(t))

        return obj

    def __deepcopy__(self, memo):
        """
        Returns a new instance of this class, initialized with a deep copy
        of the associted `DataProvider`.

        Args
        -----
        memo : dict
            See the documentation on `deepcopy` for an explanation on how
            this works.

        Returns
        --------
        A deep copy of this instance
        """
        # Force the data provider to load data, if needed
        len(self)

        obj = self.__class__()
        to_copy = ('_phu', '_nddata', '_path', '_orig_filename',
                   '_tables', '_exposed', '_resetting', '_indices')
        for attr in to_copy:
            obj.__dict__[attr] = deepcopy(self.__dict__[attr])

        # Top-level tables
        for key in set(self.__dict__) - set(obj.__dict__):
            obj.__dict__[key] = obj.__dict__['_tables'][key]

        return obj

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
            with suppress(AttributeError, KeyError):
                return self._keyword_dict[name]
        else:
            raise AttributeError("No match for '{}'".format(name))

    def _process_tags(self):
        """
        Determines the tag set for the current instance

        Returns
        --------
        A set of strings
        """
        # This prevents infinite recursion
        if self._processing_tags:
            return set()
        self._processing_tags = True
        try:
            results = []
            # Calling inspect.getmembers on `self` would trigger all the properties (tags,
            # phu, hdr, etc.), and that's undesirable. To prevent that, we'll inspect the
            # *class*. But that returns us unbound methods. We use `method.__get__(self)` to
            # get a bound version.
            #
            # It's a bit of a roundabout way to get to what we want, but it's better than
            # the option...
            for mname, method in inspect.getmembers(self.__class__, lambda x: hasattr(x, 'tag_method')):
                ts = method.__get__(self)()
                plus, minus, blocked_by, blocks, if_present = ts
                if plus or minus or blocks:
                    results.append(ts)

            # Sort by the length of substractions... those that substract from others go first
            results = sorted(results, key=lambda x: len(x.remove) + len(x.blocks), reverse=True)
            # Sort by length of blocked_by... those that are never disabled go first
            results = sorted(results, key=lambda x: len(x.blocked_by))
            # Sort by length of if_present... those that need other tags to be present go last
            results = sorted(results, key=lambda x: len(x.if_present))

            tags = set()
            removals = set()
            blocked = set()
            for plus, minus, blocked_by, blocks, is_present in results:
                if is_present:
                    # If this TagSet requires other tags to be present, make sure that all of
                    # them are. Otherwise, skip...
                    if len(tags & is_present) != len(is_present):
                        continue
                allowed = (len(tags & blocked_by) + len(plus & blocked)) == 0
                if allowed:
                    # This set is not being blocked by others...
                    removals.update(minus)
                    tags.update(plus - removals)
                    blocked.update(blocks)
        finally:
            self._processing_tags = False

        return tags

    @staticmethod
    def _matches_data(dataprov):
        # This one is trivial. As long as we get a FITS file...
        return True

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

    @property
    def phu(self):
        return self._phu

    def set_phu(self, phu):
        self._phu = phu

    def _get_raw_headers(self, with_phu=False):
        extensions = [ndd.meta['header'] for ndd in self.nddata]
        if with_phu:
            return [self._phu] + extensions
        return extensions

    @property
    def hdr(self):
        if not self.nddata:
            return None
        from .fits import FitsHeaderCollection
        headers = self._get_raw_headers()
        return headers[0] if self.is_single else FitsHeaderCollection(headers)

    @property
    @deprecated("Access to headers through this property is deprecated and "
                "will be removed in the future. Use '.hdr' instead.")
    def header(self):
        return self._get_raw_headers(with_phu=True)

    @property
    def tags(self):
        """
        A set of strings that represent the tags defining this instance
        """
        return self._process_tags()

    @property
    def descriptors(self):
        """
        Returns a sequence of names for the methods that have been
        decorated as descriptors.

        Returns
        --------
        A tuple of str
        """
        members = inspect.getmembers(self.__class__,
                                     lambda x: hasattr(x, 'descriptor_method'))
        return tuple(mname for (mname, method) in members)

    @property
    def is_sliced(self):
        """
        If this data provider instance represents the whole dataset, return
        False. If it represents a slice out of the whole, return True.

        Returns
        --------
        A boolean
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
        """
        Predicate that can be used to figure out if certain attribute of the
        `DataProvider` is meant to be modified by an external object.

        This is used mostly by `AstroData`, which acts as a proxy exposing
        attributes of its assigned provider, to decide if it should set a value
        on the provider or on itself.

        Args
        -----
        attribute : str

        Returns
        --------
        A boolean
        """
        if self.is_sliced and attr in {'path', 'filename'}:
            return False
        return attr in self._fixed_settable or attr.isupper()

    @property
    def nddata(self):
        if self._indices is not None:
            ndd = [self._nddata[i] for i in self._indices]
        else:
            ndd = self._nddata
        return ndd[0] if self.is_single else ndd

    def table(self):
        # FIXME: do we need this in addition to .tables ?
        return self._tables.copy()

    @property
    def tables(self):
        return set(self._tables.keys())

    @property
    def shape(self):
        if self.is_single:
            return self.nddata.shape
        else:
            return [nd.shape for nd in self.nddata]

    @property
    def data(self):
        """
        A list of the the arrays (or single array, if this is a single slice)
        corresponding to the science data attached to each extension, in
        loading/appending order.
        """
        if self.is_single:
            return self.nddata.data
        else:
            return [nd.data for nd in self.nddata]

    @data.setter
    def data(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that "
                             "is not a single slice")

        # Setting the ._data in the NDData is a bit kludgy, but we're all
        # grown adults and know what we're doing, isn't it?
        if hasattr(value, 'shape'):
            self.nddata._data = value
        else:
            raise AttributeError("Trying to assign data to be something "
                                 "with no shape")

    @property
    def uncertainty(self):
        """
        A list of the uncertainty objects (or a single object, if this is
        a single slice) attached to the science data, for each extension, in
        loading/appending order.

        The objects are instances of AstroPy's `NDUncertainty`, or `None` where
        no information is available.

        See also
        ---------
        variance: The actual array supporting the uncertainty object
        """
        if self.is_single:
            return self.nddata.uncertainty
        else:
            return [nd.uncertainty for nd in self.nddata]

    @uncertainty.setter
    def uncertainty(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that "
                             "is not a single slice")
        self.nddata.uncertainty = value

    @property
    def mask(self):
        """
        A list of the mask arrays (or a single array, if this is a single
        slice) attached to the science data, for each extension, in
        loading/appending order.

        For objects that miss a mask, `None` will be provided instead.
        """
        if self.is_single:
            return self.nddata.mask
        else:
            return [nd.mask for nd in self.nddata]

    @mask.setter
    def mask(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that "
                             "is not a single slice")
        self.nddata.mask = value

    @property
    def variance(self):
        """
        A list of the variance arrays (or a single array, if this is a single
        slice) attached to the science data, for each read_fitsextension, in
        loading/appending order.

        For objects that miss uncertainty information, `None` will be provided
        instead.

        See also
        ---------
        uncertainty: The `NDUncertainty` object used under the hood to
        propagate uncertainty when operating with the data
        """
        if self.is_single:
            return self.nddata.variance
        else:
            return [nd.variance for nd in self.nddata]

    @variance.setter
    def variance(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object that "
                             "is not a single slice")
        if value is None:
            self.nddata.uncertainty = None
        else:
            self.nddata.uncertainty = ADVarianceUncertainty(value)

    @property
    def wcs(self):
        if self.is_single:
            return self.nddata.wcs
        else:
            raise ValueError("Cannot return WCS for an AstroData object "
                             "that is not a single slice")

    @wcs.setter
    def wcs(self, value):
        if not self.is_single:
            raise ValueError("Trying to assign to an AstroData object "
                             "that is not a single slice")
        self.nddata.wcs = value

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    def __getitem__(self, idx):
        """
        Returns a sliced view of the instance. It supports the standard
        Python indexing syntax.

        Args
        -----
        slice : int, `slice`
            An integer or an instance of a Python standard `slice` object

        Raises
        -------
        TypeError
            If trying to slice an object when it doesn't make sense (eg.
            slicing a single slice)
        ValueError
            If `slice` does not belong to one of the recognized types
        IndexError
            If an index is out of range

        """
        if self.is_single:
            raise TypeError("Can't slice a single slice!")

        indices, multiple = normalize_indices(idx, nitems=len(self))
        # FIXME: propagate other attributes ? path, orig_filename, etc.
        if self._indices:
            # FIXME: slice _indices
            pass
        return self.__class__(self.nddatas, other=self.other, phu=self.phu,
                              indices=indices)

    def __delitem__(self, idx):
        """
        Called to implement deletion of `self[idx]`.  Supports standard
        Python syntax (including negative indices).

        Args
        -----
        idx : integer
            This index represents the order of the element that you want
            to remove.

        Raises
        -------
        IndexError
            If `idx` is out of range
        """
        if self._indices:
            raise TypeError("Can't remove items from a sliced object")
        # FIXME: what happens with indices/slices ?
        del self._nddata[idx]

    def __getattr__(self, attribute):
        """
        Called when an attribute lookup has not found the attribute in the
        usual places (not an instance attribute, and not in the class tree
        for `self`).

        This is implemented to provide access to objects exposed by
        the `DataProvider`.

        Args
        -----
        attribute : string
            The attribute's name

        Raises
        -------
        AttributeError
            If the attribute could not be found/computed.
        """
        try:
            return getattr(self._dataprov, attribute)
        except AttributeError:
            raise AttributeError("{!r} object has no attribute {!r}"
                                 .format(self.__class__.__name__, attribute))

    def __setattr__(self, attribute, value):
        """
        Called when an attribute assignment is attempted, instead of the normal
        mechanism.  This method will check first with the `DataProvider`: if
        the DP says it will contain this attribute, or that it will accept it
        for setting, then the value will be stored at the DP level. Otherwise,
        the regular attribute assignment mechanisme takes over and the value
        will be store as an instance attribute of `self`.

        Args
        -----
        attribute : string
            The attribute's name

        value : object
            The value to be assigned to the attribute

        Returns
        --------
        If the value is passed to the `DataProvider`, and it is not of an
        acceptable type, a `ValueError` (or other exception) may be rised.
        Please, check the appropriate documentation for this.

        """
        if attribute != '_dataprov' and '_dataprov' in self.__dict__:
            if self._dataprov.is_settable(attribute):
                setattr(self._dataprov, attribute, value)
                return
        super().__setattr__(attribute, value)

    def __delattr__(self, attribute):
        """
        Implements attribute removal. If `self` represents a single slice, the
        """
        try:
            try:
                self._dataprov.__delattr__(attribute)
            except (ValueError, AttributeError):
                super().__delattr__(attribute)
        except AttributeError:
            if self.is_sliced:
                raise AttributeError(
                    "{!r} sliced object has no attribute {!r}"
                    .format(self.__class__.__name__, attribute))
            else:
                raise

    def __contains__(self, attribute):
        """
        Implements the ability to use the `in` operator with an `AstroData`
        object.  It will look up the specified attribute name within the
        exposed members of the internal `DataProvider` object. Refer to the
        concrete `DataProvider` implementation's documentation to know what
        members are exposed.

        Args
        -----
        attribute : string
            An attribute name

        Returns
        --------
        A boolean
        """
        return attribute in self.exposed

    def __len__(self):
        """
        Number of independent extensions stored by the `DataProvider`

        Returns
        --------
        A non-negative integer.
        """
        if self._indices is not None:
            return len(self._indices)
        else:
            return len(self._nddata)

    @property
    def exposed(self):
        """
        A collection of strings with the names of objects that can be accessed
        directly by name as attributes of this instance, and that are not part
        of its standard interface (ie. data objects that have been added
        dynamically).

        Examples
        ---------
        >>> ad[0].exposed  # doctest: +SKIP
        set(['OBJMASK', 'OBJCAT'])

        """
        return self._exposed.copy()

    def _pixel_info(self, indices):
        for idx, obj in ((n, self._nddata[k]) for n, k in enumerate(indices)):
            other_objects = []
            uncer = obj.uncertainty
            fixed = (('variance', None if uncer is None else uncer),
                     ('mask', obj.mask))
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
                            attr=name,
                            type=type(other).__name__,
                            dim=dim,
                            data_type=dt
                        ))

            yield dict(
                idx='[{:2}]'.format(idx),
                main=dict(
                    content='science',
                    type=type(obj).__name__,
                    dim='({})'.format(', '.join(str(s) for s in obj.data.shape)),
                    data_type=obj.data.dtype.name
                ),
                other=other_objects
            )

    def _other_info(self):
        # NOTE: This covers tables, only. Study other cases before
        # implementing a more general solution
        if self._tables:
            for name, table in sorted(self._tables.items()):
                if type(table) is list:
                    # This is not a free floating table
                    continue
                yield (name, 'Table', (len(table), len(table.columns)))

    def info(self):
        """
        Prints out information about the contents of this instance.
        """
        print("Filename: {}".format(self.path if self.path else "Unknown"))
        # This is fixed. We don't support opening for update
        # print("Mode: readonly")

        tags = sorted(self.tags, reverse=True)
        tag_line = "Tags: "
        while tags:
            new_tag = tags.pop() + ' '
            if len(tag_line + new_tag) > 80:
                print(tag_line)
                tag_line = "    " + new_tag
            else:
                tag_line = tag_line + new_tag
        print(tag_line)

        # Let's try to be generic. Could it be that some file contains
        # only tables?
        indices = (tuple(range(len(self))) if self._indices is None
                   else self._indices)

        if indices:
            main_fmt = "{:6} {:24} {:17} {:14} {}"
            other_fmt = "          .{:20} {:17} {:14} {}"
            print("\nPixels Extensions")
            print(main_fmt.format("Index", "Content", "Type", "Dimensions",
                                  "Format"))
            for pi in self._pixel_info(indices):
                main_obj = pi['main']
                print(main_fmt.format(
                    pi['idx'], main_obj['content'][:24], main_obj['type'][:17],
                    main_obj['dim'], main_obj['data_type']))
                for other in pi['other']:
                    print(other_fmt.format(
                        other['attr'][:20], other['type'][:17], other['dim'],
                        other['data_type']))

        additional_ext = list(self._other_info())
        if additional_ext:
            print("\nOther Extensions")
            print("               Type        Dimensions")
            for (attr, type_, dim) in additional_ext:
                print(".{:13} {:11} {}".format(attr[:13], type_[:11], dim))

    def _oper(self, operator, operand):
        if isinstance(operand, AstroData):
            if len(operand) != len(self):
                raise ValueError("Operands are not the same size")
            for n in range(len(self)):
                try:
                    data = operand.nddata if operand.is_single else operand.nddata[n]
                    self._nddata[n] = operator(self._nddata[n], data)
                except TypeError:
                    # This may happen if operand is a sliced, single
                    # AstroData object
                    self._nddata[n] = operator(self._nddata[n], operand.nddata)
            op_table = operand.table()
            ltab, rtab = set(self._tables), set(op_table)
            for tab in (rtab - ltab):
                self._tables[tab] = op_table[tab]
        else:
            for n in range(len(self)):
                self._nddata[n] = operator(self._nddata[n], operand)

    def _standard_nddata_op(self, fn, operand):
        return self._oper(partial(fn, handle_mask=np.bitwise_or,
                                  handle_meta='first_found'), operand)

    def __add__(self, oper):
        """
        Implements the binary arithmetic operation `+` with `AstroData` as
        the left operand.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        A new `AstroData` instance
        """
        copy = deepcopy(self)
        copy += oper
        return copy

    def __sub__(self, oper):
        """
        Implements the binary arithmetic operation `-` with `AstroData` as
        the left operand.

        Args
        -----
        oper : number or object
            The operand to be subtracted to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        A new `AstroData` instance
        """
        copy = deepcopy(self)
        copy -= oper
        return copy

    def __mul__(self, oper):
        """
        Implements the binary arithmetic operation `*` with `AstroData` as
        the left operand.

        Args
        -----
        oper : number or object
            The operand to be multiplied to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        A new `AstroData` instance
        """
        copy = deepcopy(self)
        copy *= oper
        return copy

    def __truediv__(self, oper):
        """
        Implements the binary arithmetic operation `/` with `AstroData` as
        the left operand.

        Args
        -----
        oper : number or object
            The operand to be divided to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        A new `AstroData` instance
        """
        copy = deepcopy(self)
        copy /= oper
        return copy

    def __iadd__(self, oper):
        """
        Implements the augmented arithmetic assignment `+=`.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        `self`
        """
        self._standard_nddata_op(NDDataObject.add, oper)
        return self

    def __isub__(self, oper):
        """
        Implements the augmented arithmetic assignment `-=`.

        Args
        -----
        oper : number or object
            The operand to be subtracted to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        `self`
        """
        self._standard_nddata_op(NDDataObject.subtract, oper)
        return self

    def __imul__(self, oper):
        """
        Implements the augmented arithmetic assignment `*=`.

        Args
        -----
        oper : number or object
            The operand to be multiplied to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        `self`
        """
        self._standard_nddata_op(NDDataObject.multiply, oper)
        return self

    def __itruediv__(self, oper):
        """
        Implements the augmented arithmetic assignment `/=`.

        Args
        -----
        oper : number or other
            The operand to be divided to this instance. The accepted types
            depend on the `DataProvider`.

        Returns
        --------
        `self`
        """
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
        copy = deepcopy(self)
        # copy._dataprov.__rtruediv__(oper)
        copy._oper(copy._rdiv, oper)
        return copy

    def _reset_ver(self, nd):
        try:
            ver = max(_nd.meta['ver'] for _nd in self._nddata) + 1
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
        # as `NDDataObject`, instead of the random one that the user may pass
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
            self._nddata.append(new_nddata)
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

        Args
        -----
        ext : array, `NDData`, `Table`, other
            The contents for the new extension. The exact accepted types depend
            on the class implementing this interface. Implementations specific
            to certain data formats may accept specialized types (eg. a FITS
            provider will accept an `ImageHDU` and extract the array out of it)

        name : str, optional
            A name that may be used to access the new object, as an attribute
            of the provider. The name is typically ignored for top-level
            (global) objects, and required for the others. If the name cannot
            be derived from the metadata associated to `extension`, you will
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
            If adding the object in an invalid situation (eg. `name` is `None`
            when adding to a single slice)
        ValueError
            Raised if the extension is of a proper type, but its value is
            illegal somehow.

        """
        if not self.is_single:
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
                             "Use set_phu if you really need to set one")

        if self.is_sliced:
            add_to = self.nddata[0]  # FIXME: check this

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
    def read(cls, source):
        """Read from a file, file object, HDUList, etc."""
        return read_fits(cls, source)

    load = read  # for backward compatibility

    def write(self, filename=None, overwrite=False):
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
        A dictionary `{EXTVER:index, ...}`

        Raises
        ------
        ValueError
            If used against a single slice. It is of no use in that situation.
        """
        if self.is_single:
            raise ValueError("Trying to get a mapping out of a single slice")
        return {nd._meta['ver']: n for (n, nd) in enumerate(self.nddata)}

    def extver(self, ver):
        """
        Get an extension using its EXTVER instead of the positional index
        in this object.

        Parameters
        ----------
        ver : int
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

        Args
        -----
        operator : function, or bound method
            A function that takes an array (and, maybe, other arguments)
            and returns an array

        args : optional
            Additional arguments to be passed positionally to the `operator`

        kwargs : optional
            Additional arguments to be passed by name to the `operator`

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
        Sets the .data, and optionally .mask and .variance attributes of a
        single-extension AstroData slice. This function will optionally
        check whether these attributes have the same shape.

        Parameters
        ----------
        data : ndarray
            The array to assign to the .data attribute ("SCI")

        mask : ndarray, optional
            The array to assign to the .mask attribute ("DQ")

        variance: ndarray, optional
            The array to assign to the .variance attribute ("VAR")

        check: bool
            If set, then the function will check that the mask and variance
            arrays have the same shape as the data array

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


class AstroDataFits(AstroData):
    """Keep this for now as other classes inherit from it."""
