

try:
    from builtins import object
    from future.utils import with_metaclass
except ImportError:
    raise ImportError("AstroData requires the 'future' package for Python 2/3 compatibility")

from abc import ABCMeta, abstractmethod, abstractproperty
from functools import wraps
import inspect
from collections import namedtuple
from copy import deepcopy

class TagSet(namedtuple('TagSet', 'add remove blocked_by blocks if_present')):
    """
    TagSet(add=None, remove=None, blocked_by=None, blocks=None, if_present=None)

    Named tuple that is used by tag methods to return the actions that should be
    performed on a tag set. All the attributes are optional, and any combination
    of them can be used, allowing to create complex tag structures. Read the
    :ref:`documentation on the tag-generating algorithm <ad_tags>` if you want
    to better understand the interactions.

    The simplest ``TagSet``, though, tends to just add tags to the global set.

    It can be initialized by position, like any other tuple (the order of the
    arguments is the one in which the attributes are listed below). It can
    also be initialized by name.

    Attributes
    ----------
    add : set of strings, or ``None``
        Tags to be added to the global set
    remove : set of strings, or ``None``
        Tags to be removed from the global set
    blocked_by : set of strings, or ``None``
        Tags that will prevent this ``TagSet`` to be applied
    blocks : set of strings, or ``None``
        Other ``TagSet``\ s containing these won't be applied
    if_present : set of strings, or ``None``
        This ``TagSet`` will be applied only **all** of these tags are present

    Examples
    ---------
    >>> TagSet()
    TagSet(add=set([]), remove=set([]), blocked_by=set([]), blocks=set([]), if_present=set([]))
    >>> TagSet(set(['BIAS', 'CAL']))
    TagSet(add=set(['BIAS', 'CAL']), remove=set([]), blocked_by=set([]), blocks=set([]), if_present=set([]))
    >>> TagSet(remove=set(['BIAS', 'CAL']))
    TagSet(add=set([]), remove=set(['BIAS', 'CAL']), blocked_by=set([]), blocks=set([]), if_present=set([]))
    """
    def __new__(cls, add=None, remove=None, blocked_by=None, blocks=None, if_present=None):
        return super(TagSet, cls).__new__(cls, add or set(),
                                               remove or set(),
                                               blocked_by or set(),
                                               blocks or set(),
                                               if_present or set())

def astro_data_descriptor(fn):
    """
    Decorator that will mark a class method as an AstroData descriptor.
    Useful to produce list of descriptors, for example.

    If used in combination with other decorators, this one **must** be the
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

# Ensures descriptors coded to return a list (one value per extension)
# only return a value if sent a slice with a single extension
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
                    raise IndexError("Incompatible numbers of extensions and elements in {}".format(fn.__name__))
            else:
                return [ret] * len(self)
    return wrapper


def astro_data_tag(fn):
    """
    Decorator that marks methods of an :py:class:`AstroData` derived class as part of the
    tag-producing system.

    It wraps the method around a function that will ensure a consistent return
    value: the wrapped method can return any sequence of sequences of strings,
    and they will be converted to a ``TagSet`` of ``set``. If the wrapped method
    returns ``None``, it will be turned into an empty ``TagSet``.

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
                    raise TypeError("Tag function {} didn't return a TagSet".format(fn.__name__))

                return TagSet(*tuple(set(s) for s in ret))
        except KeyError:
            pass

        # Return empty TagSet for the "doesn't apply" case
        return TagSet()

    wrapper.tag_method = True
    return wrapper

class AstroDataError(Exception):
    pass

class DataProvider(with_metaclass(ABCMeta, object)):
    """
    Abstract class describing the minimal interface that :py:class:`DataProvider` derivative
    classes need to implement.
    """

    @property
    def is_sliced(self):
        """
        If this data provider instance represents the whole dataset, return
        ``False``. If it represents a slice out of the whole, return ``True``.

        Returns
        --------
        A boolean
        """
        return False

    @property
    def is_single(self):
        """
        If this data provider represents a single slice out of a whole dataset,
        return ``True``. Otherwise, return ``False``.

        Returns
        --------
        A boolean
        """
        return False

    @abstractmethod
    def is_settable(self, attribute):
        """
        Predicate that can be used to figure out if certain attribute of the
        :py:class:`DataProvider` is meant to be modified by an external object.

        This is used mostly by :py:class:`AstroData`, which acts as a proxy exposing
        attributes of its assigned provider, to decide if it should set a value
        on the provider or on itself.

        Args
        -----
        attribute : str

        Returns
        --------
        A boolean
        """
        pass

    @abstractmethod
    def append(self, ext, name=None):
        """
        Adds a new component to the provider. Objects appended to a single slice will
        actually be made hierarchically dependent of the science object represented by
        that slice. If appended to the provider as a whole, the new member will be
        independent (eg. global table, new science object).

        Args
        -----
        ext : array, ``NDData``, ``Table``, etc
            The component to be added. The exact accepted types depend on the class
            implementing this interface. Implementations specific to certain data formats
            may accept specialized types (eg. a FITS provider will accept an ``ImageHDU``
            and extract the array out of it)

        name : str, optional
            A name that may be used to access the new object, as an attribute of the
            provider. The name is typically ignored for top-level (global) objects,
            and required for the others.

            It can consist in a combination of numbers and letters, with the restriction
            that the letters have to be all capital, and the first character cannot be
            a number ("[A-Z][A-Z0-9]*").

        Returns
        --------
        The same object, or a new one, if it was necessary to convert it to a more
        suitable format for internal use.

        Raises
        -------
        TypeError
            If adding the object in an invalid situation (eg. ``name`` is ``None`` when
            adding to a single slice)

        ValueError
            If adding an object that is not acceptable
        """
        pass

    @abstractmethod
    def __getitem__(self, slice):
        """
        Returns a sliced view of the provider. It supports the standard Python indexing
        syntax, including negative indices.

        Args
        -----
        slice : int, ``slice``
            An integer or an instance of a Python standard ``slice`` object

        Raises
        -------
        TypeError
            If trying to slice an object when it doesn't make sense (eg. slicing a single
            slice)

        ValueError
            If ``slice`` does not belong to one of the recognized types

        IndexError
            If an index is out of range

        Examples
        ---------
        >>> single = provider[0]
        >>> multiple = provider[:5]
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        "Length" of the object. This method will typically return the number of science
        objects contained by this provider, but this may change depending on the
        implementation.

        Returns
        --------
        An integer
        """
        pass

    @abstractmethod
    def __iadd__(self, oper):
        """
        This method should attempt to do an in-place (modifying self) addition of each
        internal science object and the oper.

        Args
        -----
        oper : object
            An operand to add to the internal science objects. The actual accepted type
            depends on the implementation

        Returns
        --------
        Generally, it should return ``self``. The implementations may decide to return
        something else instead.
        """
        pass

    @abstractmethod
    def __isub__(self, oper):
        """
        This method should attempt to do an in-place (modifying self) subtraction of each
        internal science object and the oper.

        Args
        -----
        oper : object
            An operand to subtract from the internal science objects. The actual accepted type
            depends on the implementation

        Returns
        --------
        Generally, it should return ``self``. The implementations may decide to return
        something else instead.
        """
        pass

    @abstractmethod
    def __imul__(self, oper):
        """
        This method should attempt to do an in-place (modifying self) multiplication of each
        internal science object and the oper.

        Args
        -----
        oper : object
            An operand to multiply the internal science objects by. The actual accepted type
            depends on the implementation

        Returns
        --------
        Generally, it should return ``self``. The implementations may decide to return
        something else instead.
        """
        pass

    @abstractmethod
    def __idiv__(self, oper):
        """
        This method should attempt to do an in-place (modifying self) division of each
        internal science object and the oper.

        Args
        -----
        oper : object
            An operand to divide the internal science objects by. The actual accepted type
            depends on the implementation

        Returns
        --------
        Generally, it should return ``self``. The implementations may decide to return
        something else instead.
        """
        pass

    @property
    def exposed(self):
        """
        A collection of strings with the names of objects that can be accessed directly
        by name as attributes of this instance, which are not part of its standard
        interface (ie. data objects that have been added dynamically).

        Examples
        ---------
        >>> ad[0].exposed
        set(['OBJMASK', 'OBJCAT'])
        >>> ad[0].OBJCAT
        ...
        """
        return ()

    @abstractproperty
    def data(self):
        """
        A list of the the arrays (or single array, if this is a single slice) corresponding
        to the science data attached to each extension, in loading/appending order.
        """
        pass

    @abstractproperty
    def uncertainty(self):
        """
        A list of the uncertainty objects (or a single object, if this is a single slice)
        attached to the science data, for each extension, in loading/appending order.

        The objects are instances of AstroPy's ``NDUncertainty``, or ``None`` where no information
        is available.

        See also
        ---------
        variance
        """
        pass

    @abstractproperty
    def mask(self):
        """
        A list of the mask arrays (or a single array, if this is a single slice) attached to the
        science data, for each extension, in loading/appending order.

        For objects that miss a mask, ``None`` will be provided instead.
        """
        pass

    @abstractproperty
    def variance(self):
        """
        A list of the variance arrays (or a single array, if this is a single slice) attached to
        the science data, for each extension, in loading/appending order.

        For objects that miss uncertainty information, ``None`` will be provided instead.

        See also
        ---------
        uncertainty: The ``NDUncertainty`` object used under the hood to propagate uncertainty when
                     operating with the data
        """
        pass

# NOTE: This is not being used at all. Maybe it would be better to remove it altogether for the time
#       being, and reimplement it if it's ever needed
#
# def simple_descriptor_mapping(**kw):
#     def decorator(cls):
#         for descriptor, descriptor_def in kw.items():
#             setattr(cls, descriptor, property(descriptor_def))
#         return cls
#     return decorator

class AstroData(object):
    """
    AstroData(provider)

    Base class for the AstroData software package. It provides an interface to manipulate
    astronomical data sets.

    Parameters:
       provider (DataProvider):  The data that will be manipulated through the
          :py:class:`AstroData` instance.
    """

    # Simply a value that nobody is going to try to set an NDData attribute to
    _IGNORE = -23

    def __init__(self, provider):
        if not isinstance(provider, DataProvider):
            raise ValueError("AstroData is initialized with a DataProvider object. You may want to use ad.open('...') instead")
        self._dataprov = provider
        self._processing_tags = False

    def __deepcopy__(self, memo):
        """
        Returns a new instance of this class, initialized with a deep copy of the associted :py:class:`DataProvider`

        Parameters:
           memo (dict): See the documentation on :py:class:`copy.deepcopy` for an explanation on how this works

        Returns:
           AstroData: A deep copy of this instance
        """
        # Force the data provider to load data, if needed
        len(self._dataprov)
        dp = deepcopy(self._dataprov, memo)
        ad = self.__class__(dp)
        return ad

    def __process_tags(self):
        """
        Determines the tag set for the current instance

        Returns:
           set: A set of strings
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

    @property
    def tags(self):
        """
        A :py:class:`set` of strings that represent the tags defining this instance
        """
        return self.__process_tags()

    @property
    def descriptors(self):
        """
        Returns a sequence of names for the methods that have been
        decorated as descriptors.

        Returns:
           tuple: A tuple of str
        """
        members = inspect.getmembers(self.__class__,
                                     lambda x: hasattr(x, 'descriptor_method'))
        return tuple(mname for (mname, method) in members)


    def __iter__(self):
        for single in self._dataprov:
            yield self.__class__(single)

    def __getitem__(self, slicing):
        """
        Returns a sliced view of the instance. It supports the standard Python indexing
        syntax.

        Args
        -----
        slice : ``int``, ``slice``
            An integer or an instance of a Python standard :py:class:`slice` object

        Raises
        -------
        TypeError
            If trying to slice an object when it doesn't make sense (eg. slicing a single
            slice)

        ValueError
            If ``slice`` does not belong to one of the recognized types

        IndexError
            If an index is out of range

        Examples
        ---------
        >>> single = ad[0]
        >>> multiple = ad[:5]
        """
        return self.__class__(self._dataprov[slicing])

    def __delitem__(self, idx):
        """
        Called to implement deletion of ``self[idx]``.  Supports standard Python syntax
        (including negative indices).

        Args
        -----
        idx : integer
            This index represents the order of the element that you want to remove.

        Raises
        -------
        IndexError
            If ``idx`` is out of range
        """
        del self._dataprov[idx]

    def __getattr__(self, attribute):
        """
        Called when an attribute lookup has not found the attribute in the usual places
        (not an instance attribute, and not in the class tree for ``self``).

        This is implemented to provide access to objects exposed by the :py:class:`DataProvider`

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
            clsname = self.__class__.__name__
            raise AttributeError("{!r} object has no attribute {!r}".format(clsname, attribute))

    def __setattr__(self, attribute, value):
        """
        Called when an attribute assignment is attempted, instead of the normal mechanism.
        This method will check first with the :py:class:`DataProvider`: if the DP says it will contain
        this attribute, or that it will accept it for setting, then the value will be stored
        at the DP level. Otherwise, the regular attribute assignment mechanisme takes over
        and the value will be store as an instance attribute of ``self``.

        Args
        -----
        attribute : string
            The attribute's name

        value : object
            The value to be assigned to the attribute

        Raises
        -------
        ValueError
           If the value is passed to the :py:class:`DataProvider`, and it is not of an acceptable type.
           Please, check the appropriate documentation for this.
        """
        if attribute != '_dataprov' and '_dataprov' in self.__dict__:
            if self._dataprov.is_settable(attribute):
                setattr(self._dataprov, attribute, value)
                return
        super(AstroData, self).__setattr__(attribute, value)

    def __delattr__(self, attribute):
        """
        Implements attribute removal.

        Raises
        -------
        AttributeError
            if ``self`` represents a slice
        """
        try:
            try:
                self._dataprov.__delattr__(attribute)
            except (ValueError, AttributeError):
                super(AstroData, self).__delattr__(attribute)
        except AttributeError:
            if self._dataprov.is_sliced:
                raise AttributeError("{!r} sliced object has no attribute {!r}".format(self.__class__.__name__, attribute))
            else:
                raise

    def __contains__(self, attribute):
        """
        Implements the ability to use the ``in`` operator with an :py:class:`AstroData` object.
        It will look up the specified attribute name within the exposed members of
        the internal :py:class:`DataProvider` object. Refer to the concrete ``DataProvider``
        implementation's documentation to know what members are exposed.

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
        Number of independent extensions stored by the :py:class:`DataProvider`

        Returns
        --------
        A non-negative integer.
        """
        return len(self._dataprov)

    @abstractmethod
    def info(self):
        """
        Prints out information about the contents of this instance. Implemented
        by the derived classes.
        """
        pass

    def __add__(self, oper):
        """
        Implements the binary arithmetic operation ``+`` with :py:class:`AstroData` as the left operand.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        A new :py:class:`AstroData` instance
        """
        copy = deepcopy(self)
        copy += oper
        return copy

    def __sub__(self, oper):
        """
        Implements the binary arithmetic operation ``-`` with :py:class:`AstroData` as the left operand.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        A new :py:class:`AstroData` instance
        """
        copy = deepcopy(self)
        copy -= oper
        return copy

    def __mul__(self, oper):
        """
        Implements the binary arithmetic operation ``*`` with :py:class:`AstroData` as the left operand.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        A new :py:class:`AstroData` instance
        """
        copy = deepcopy(self)
        copy *= oper
        return copy

    def __div__(self, oper):
        """
        Implements the binary arithmetic operation ``/`` with :py:class:`AstroData` as the left operand.

        Args:
            oper (number or object): The operand to be added to this instance. The accepted
               types depend on the :py:class:`DataProvider`

        Returns:
            AstroData: a new :py:class:`AstroData`  instance
        """
        copy = deepcopy(self)
        copy /= oper
        return copy

    def __iadd__(self, oper):
        """
        Implements the augmented arithmetic assignment ``+=``.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        ``self``
        """
        self._dataprov += oper
        return self

    def __isub__(self, oper):
        """
        Implements the augmented arithmetic assignment ``-=``.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        ``self``
        """
        self._dataprov -= oper
        return self

    def __imul__(self, oper):
        """
        Implements the augmented arithmetic assignment ``*=``.

        Args
        -----
        oper : number or object
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        ``self``
        """
        self._dataprov *= oper
        return self

    def __idiv__(self, oper):
        """
        Implements the augmented arithmetic assignment ``/=``.

        Args
        -----
        oper : number or other
            The operand to be added to this instance. The accepted types depend on the
            :py:class:`DataProvider`

        Returns
        --------
        ``self``
        """
        self._dataprov /= oper
        return self

    __itruediv__ = __idiv__
    add = __iadd__
    subtract = __isub__
    multiply = __imul__
    divide = __idiv__

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, oper):
        # TODO
        copy = (deepcopy(self) - oper) * -1
        return copy

    def __rdiv__(self, oper):
        # TODO
        copy = deepcopy(self)
        copy._dataprov.__rdiv__(oper)
        return copy

    # This method needs to be implemented as classmethod
    @abstractmethod
    def load(cls, source):
        """
        Class method that returns an instance of this same class, properly initialized
        with a :py:class:`DataProvider` that can deal with the object passed as ``source``

        This method is abstract and has to be implemented by derived classes.
        """
        pass

    def append(self, extension, name=None, *args, **kw):
        """
        Adds a new top-level extension to the provider. Please, read the the concrete
        :py:class:`DataProvider` documentation that is being used to know the exact behavior and
        additional accepted arguments.

        Args
        -----
        extension : array, Table, or other
            The contents for the new extension. Usually the underlying :py:class:`DataProvider`
            will understand how to deal with regular NumPy arrays and with AstroData
            ``Table`` instances, but it may also accept other types.

        name : string, optional
            A :py:class:`DataProvider` will usually require a name for extensions. If the name
            cannot be derived from the metadata associated to ``extension``, you will
            have to provide one.

        args : optional
            The :py:class:`DataProvider` may accept additional arguments. Please, refer to its
            documentation.

        kw : optional
            The :py:class:`DataProvider` may accept additional arguments. Please, refer to its
            documentation.

        Returns
        --------
        The instance that has been added internally (potentially **not** the same that
        was passed as ``extension``)

        Raises
        -------
        TypeError
            Will be raised if the :py:class:`DataProvider` doesn't know how to deal with the
            data that has been passed.

        ValueError
            Raised if the extension is of a proper type, but its value is illegal
            somehow.
        """
        return self._dataprov.append(extension, name=name, *args, **kw)

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
                ad.ext.mask = operator(ad.ext.mask, *args, **kwargs) if ad.ext.mask is not None else None
                ad.ext.variance = operator(ad.ext.variance, *args, **kwargs) if ad.ext.variance is not None else None

        with the additional advantage that it will work on single slices, too.

        Args
        -----
        operator : function, or bound method
            A function that takes an array (and, maybe, other arguments)
            and returns an array

        args : optional
            Additional arguments to be passed positionally to the ``operator``

        kwargs : optional
            Additional arguments to be passed by name to the ``operator``

        Examples
        ---------
        >>> import numpy as np
        >>> ad.operate(np.squeeze)
        """
        # Ensure we can iterate, even on a single slice
        for ext in [self] if self.is_single else self:
            ext.data = operator(ext.data, *args, **kwargs)
            if ext.mask is not None:
                ext.mask = operator(ext.mask, *args, **kwargs)
            if ext.variance is not None:
                ext.variance = operator(ext.variance, *args, **kwargs)

    def reset(self, data, mask=_IGNORE, variance=_IGNORE, check=True):
        """
        Sets the .data, and optionally .mask and .variance attributes of a
        single-extension :py:class:`AstroData` slice. This function will optionally
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
            elif mask == self._IGNORE:
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
            elif variance == self._IGNORE:
                if hasattr(data, 'uncertainty'):
                    self.uncertainty = data.uncertainty
            else:
                raise TypeError("Attempt to set variance inappropriately")
        else:
            self.variance = variance
