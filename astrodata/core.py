import inspect
import os
import re
from collections import namedtuple
from contextlib import suppress
from copy import deepcopy
from functools import partial, wraps

import numpy as np

from astropy.io import fits

from .nddata import NDAstroData as NDDataObject, ADVarianceUncertainty

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

    def __init__(self, nddatas, other=None, phu=None):
        if isinstance(nddatas, (list, tuple)):
            nddatas = dict(zip(range(len(nddatas)), nddatas))
        self._nddata = nddatas
        self._other = other
        self._phu = phu or fits.Header()
        self._processing_tags = False

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
            '_fixed_settable': {
                'data',
                'uncertainty',
                'mask',
                'variance',
                'path',
                'filename'
                }
            })

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
        len(self._dataprov)

        obj = self.__class__()
        to_copy = ('_sliced', '_phu', '_single', '_nddata',
                   '_path', '_orig_filename', '_tables', '_exposed',
                   '_resetting')
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
    def phu(self):
        return self._dataprov.phu()

    @property
    def hdr(self):
        return self._dataprov.hdr()

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
        return False

    @property
    def is_single(self):
        """
        If this data provider represents a single slice out of a whole dataset,
        return True. Otherwise, return False.

        Returns
        --------
        A boolean
        """
        return False

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
        return attr in self._fixed_settable or attr.isupper()

    def __iter__(self):
        for single in self._dataprov:
            yield self.__class__(single)

    def __getitem__(self, slicing):
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
        return self.__class__(self._dataprov[slicing])

    def __delitem__(self, idx):
        """
        Called to implement deletion of `self[idx]`.  Supports standard
        Python syntax (including negative indices).

        Args
        -----
        idx : integer
            This index represents the order of the element that you want to remove.

        Raises
        -------
        IndexError
            If `idx` is out of range
        """
        del self._dataprov[idx]

    def __getattr__(self, attribute):
        """
        Called when an attribute lookup has not found the attribute in the
        usual places (not an instance attribute, and not in the class tree
        for `self`).

        This is implemented to provide access to objects exposed by the `DataProvider`

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
            if self._dataprov.is_sliced:
                raise AttributeError("{!r} sliced object has no attribute {!r}"
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
        return attribute in self._dataprov.exposed

    def __len__(self):
        """
        Number of independent extensions stored by the `DataProvider`

        Returns
        --------
        A non-negative integer.
        """
        return len(self._dataprov)

    def append(self, ext, name=None, **kwargs):
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
        return ()

    def data(self):
        """
        A list of the the arrays (or single array, if this is a single slice)
        corresponding to the science data attached to each extension, in
        loading/appending order.
        """

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

    def mask(self):
        """
        A list of the mask arrays (or a single array, if this is a single
        slice) attached to the science data, for each extension, in
        loading/appending order.

        For objects that miss a mask, `None` will be provided instead.
        """

    def variance(self):
        """
        A list of the variance arrays (or a single array, if this is a single
        slice) attached to the science data, for each extension, in
        loading/appending order.

        For objects that miss uncertainty information, `None` will be provided
        instead.

        See also
        ---------
        uncertainty: The `NDUncertainty` object used under the hood to
        propagate uncertainty when operating with the data
        """

    def info(self):
        """
        Prints out information about the contents of this instance.
        """
        self._dataprov.info(self.tags)

    def _oper(self, operator, operand, indices=None):
        if indices is None:
            indices = tuple(range(len(self._nddata)))
        if isinstance(operand, AstroData):
            if len(operand) != len(indices):
                raise ValueError("Operands are not the same size")
            for n in indices:
                try:
                    self._set_nddata(n, operator(
                        self._nddata[n],
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
        return self._oper(partial(fn, handle_mask=np.bitwise_or,
                                  handle_meta='first_found'),
                          operand, indices)

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

    @classmethod
    def read(cls, source):
        """
        Read from a file, file object, HDUList, etc.
        """

    load = read  # for backward compatibility

    def write(self, filename=None, overwrite=False):
        if filename is None:
            if self.path is None:
                raise ValueError("A filename needs to be specified")
            filename = self.path

        hdulist = self._dataprov.to_hdulist()
        hdulist.writeto(filename, overwrite=overwrite)

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
                return self[self._dataprov.extver_map()[ver]]
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
