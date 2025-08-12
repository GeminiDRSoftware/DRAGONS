"""
This module implements a derivative class based on NDData with some Mixins,
implementing windowing and on-the-fly data scaling.
"""


import warnings
from copy import deepcopy
from functools import reduce

import numpy as np

from astropy.io.fits import ImageHDU
from astropy.modeling import Model, models
from astropy.nddata import (NDArithmeticMixin, NDData, NDSlicingMixin,
                            VarianceUncertainty)
from gwcs.wcs import WCS as gWCS
from .wcs import remove_axis_from_frame

INTEGER_TYPES = (int, np.integer)

__all__ = ['NDAstroData']


class ADVarianceUncertainty(VarianceUncertainty):
    """
    Subclass VarianceUncertainty to check for negative values.
    """
    @VarianceUncertainty.array.setter
    def array(self, value):
        if value is not None and np.any(value < 0):
            warnings.warn("Negative variance values found. Setting to zero.",
                          RuntimeWarning)
            value = np.where(value >= 0., value, 0.)
        VarianceUncertainty.array.fset(self, value)


class AstroDataMixin:
    """
    A Mixin for ``NDData``-like classes (such as ``Spectrum1D``) to enable
    them to behave similarly to ``AstroData`` objects.

    These behaviors are:
        1.  ``mask`` attributes are combined with bitwise, not logical, or,
            since the individual bits are important.
        2.  The WCS must be a ``gwcs.WCS`` object and slicing results in
            the model being modified.
        3.  There is a settable ``variance`` attribute.
        4.  Additional attributes such as OBJMASK can be extracted from
            the .meta['other'] dict
    """
    def __getattr__(self, attribute):
        """
        Allow access to attributes stored in self.meta['other'], as we do
        with AstroData objects.
        """
        if attribute.isupper():
            try:
                return self.meta['other'][attribute]
            except KeyError:
                pass
        raise AttributeError(f"{self.__class__.__name__!r} object has no "
                             f"attribute {attribute!r}")

    def _arithmetic(self, operation, operand, propagate_uncertainties=True,
                    handle_mask=np.bitwise_or, handle_meta=None,
                    uncertainty_correlation=0, compare_wcs='first_found',
                    **kwds):
        """
        Override the NDData method so that "bitwise_or" becomes the default
        operation to combine masks, rather than "logical_or"
        """
        return super()._arithmetic(
            operation, operand, propagate_uncertainties=propagate_uncertainties,
            handle_mask=handle_mask, handle_meta=handle_meta,
            uncertainty_correlation=uncertainty_correlation,
            compare_wcs=compare_wcs, **kwds)

    def _slice_wcs(self, slices):
        """
        The ``__call__()`` method of gWCS doesn't appear to conform to the
        APE 14 interface for WCS implementations, and doesn't react to
        slicing properly. We override NDSlicing's method to do what we want.
        """
        if not isinstance(self.wcs, gWCS):
            return self.wcs

        # Sanitize the slices, catching some errors early
        if not isinstance(slices, (tuple, list)):
            slices = (slices,)
        slices = list(slices)
        ndim = len(self.shape)
        if len(slices) > ndim:
            raise ValueError(f"Too many dimensions specified in slice {slices}")

        if Ellipsis in slices:
            if slices.count(Ellipsis) > 1:
                raise IndexError("Only one ellipsis can be specified in a slice")
            ell_index = slices.index(Ellipsis)
            slices[ell_index:ell_index+1] = [slice(None)] * (ndim - len(slices) + 1)
        slices.extend([slice(None)] * (ndim-len(slices)))

        mods = []
        mapped_axes = []
        for i, (slice_, length) in enumerate(zip(slices[::-1], self.shape[::-1])):
            model = []
            if isinstance(slice_, slice):
                if slice_.step and abs(slice_.step) > 1:
                    raise IndexError("Cannot slice with a step")
                if slice_.step == -1:
                    model.append(models.Scale(-1))
                if slice_.start:
                    start = (length + slice_.start) if slice_.start < 0 else slice_.start
                    if start > 0:
                        model.append(models.Shift(start))
                elif slice_.start is None and slice_.step == -1:
                    model.append(models.Shift(length - 1))
                mapped_axes.append(max(mapped_axes) + 1 if mapped_axes else 0)
            elif isinstance(slice_, INTEGER_TYPES):
                model.append(models.Const1D((length + slice_) if slice_ < 0 else slice_))
                mapped_axes.append(-1)
            elif slice_ is None:  # equivalent to slice(None, None, None)
                mapped_axes.append(max(mapped_axes) + 1 if mapped_axes else 0)
            else:
                raise IndexError("Slice not an integer or range")
            if model:
                mods.append(reduce(Model.__or__, model))
            else:
                # If the previous model was an Identity, we can hang this
                # one onto that without needing to append a new Identity
                if i > 0 and isinstance(mods[-1], models.Identity):
                    mods[-1] = models.Identity(mods[-1].n_inputs + 1)
                else:
                    mods.append(models.Identity(1))

        slicing_model = reduce(Model.__and__, mods)
        if mapped_axes != list(np.arange(ndim)):
            slicing_model = models.Mapping(
                tuple(max(ax, 0) for ax in mapped_axes)) | slicing_model
            slicing_model.inverse = models.Mapping(
                tuple(ax for ax in mapped_axes if ax != -1), n_inputs=ndim)

        if isinstance(slicing_model, models.Identity) and slicing_model.n_inputs == ndim:
            return self.wcs  # Unchanged!
        new_wcs = deepcopy(self.wcs)
        input_frame = new_wcs.input_frame
        for axis, mapped_axis in reversed(list(enumerate(mapped_axes))):
            if mapped_axis == -1:
                input_frame = remove_axis_from_frame(input_frame, axis)
        new_wcs.pipeline[0].frame = input_frame
        new_wcs.insert_transform(new_wcs.input_frame, slicing_model, after=True)
        return new_wcs

    @property
    def variance(self):
        """
        A convenience property to access the contents of ``uncertainty``.
        """
        arr = self.uncertainty
        if arr is not None:
            return arr.array

    @variance.setter
    def variance(self, value):
        self.uncertainty = (ADVarianceUncertainty(value) if value is not None
                            else None)

    @property
    def wcs(self):
        return super().wcs

    @wcs.setter
    def wcs(self, value):
        if value is not None and not isinstance(value, gWCS):
            raise TypeError("wcs value must be None or a gWCS object")
        self._wcs = value

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size


class FakeArray:

    def __init__(self, very_faked):
        self.data = very_faked
        self.shape = (100, 100)  # Won't matter. This is just to fool NDData
        self.dtype = np.float32  # Same here

    def __getitem__(self, index):
        # FAKE NEWS!
        return None

    def __array__(self):
        return self.data


class NDWindowing:

    def __init__(self, target):
        self._target = target

    def __getitem__(self, slice):
        return NDWindowingAstroData(self._target, window=slice)


class NDWindowingAstroData(AstroDataMixin, NDArithmeticMixin, NDSlicingMixin, NDData):
    """
    Allows "windowed" access to some properties of an ``NDAstroData`` instance.
    In particular, ``data``, ``uncertainty``, ``variance``, and ``mask`` return
    clipped data.
    """
    def __init__(self, target, window):
        self._target = target
        self._window = window

    def __getattr__(self, attribute):
        """
        Allow access to attributes stored in self.meta['other'], as we do
        with AstroData objects.
        """
        if attribute.isupper():
            try:
                return self._target._get_simple(attribute, section=self._window)
            except KeyError:
                pass
        raise AttributeError(f"{self.__class__.__name__!r} object has no "
                             f"attribute {attribute!r}")

    @property
    def unit(self):
        return self._target.unit

    @property
    def wcs(self):
        return self._target._slice_wcs(self._window)

    @property
    def data(self):
        return self._target._get_simple('_data', section=self._window)

    @property
    def uncertainty(self):
        return self._target._get_uncertainty(section=self._window)

    @property
    def variance(self):
        if self.uncertainty is not None:
            return self.uncertainty.array

    @property
    def mask(self):
        return self._target._get_simple('_mask', section=self._window)


def is_lazy(item):
    return isinstance(item, ImageHDU) or (hasattr(item, 'lazy') and item.lazy)


class NDAstroData(AstroDataMixin, NDArithmeticMixin, NDSlicingMixin, NDData):
    """
    Implements ``NDData`` with all Mixins, plus some ``AstroData`` specifics.

    This class implements an ``NDData``-like container that supports reading
    and writing as implemented in the ``astropy.io.registry`` and also slicing
    (indexing) and simple arithmetics (add, subtract, divide and multiply).

    A very important difference between ``NDAstroData`` and ``NDData`` is that
    the former attempts to load all its data lazily. There are also some
    important differences in the interface (eg. ``.data`` lets you reset its
    contents after initialization).

    Documentation is provided where our class differs.

    See also
    --------
    NDData
    NDArithmeticMixin
    NDSlicingMixin

    Examples
    --------

    The mixins allow operation that are not possible with ``NDData`` or
    ``NDDataBase``, i.e. simple arithmetics::

        >>> from astropy.nddata import StdDevUncertainty
        >>> import numpy as np
        >>> data = np.ones((3,3), dtype=np.float)
        >>> ndd1 = NDAstroData(data, uncertainty=StdDevUncertainty(data))
        >>> ndd2 = NDAstroData(data, uncertainty=StdDevUncertainty(data))
        >>> ndd3 = ndd1.add(ndd2)
        >>> ndd3.data
        array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]])
        >>> ndd3.uncertainty.array
        array([[1.41421356, 1.41421356, 1.41421356],
            [1.41421356, 1.41421356, 1.41421356],
            [1.41421356, 1.41421356, 1.41421356]])

    see ``NDArithmeticMixin`` for a complete list of all supported arithmetic
    operations.

    But also slicing (indexing) is possible::

        >>> ndd4 = ndd3[1,:]
        >>> ndd4.data
        array([2., 2., 2.])
        >>> ndd4.uncertainty.array
        array([1.41421356, 1.41421356, 1.41421356])

    See ``NDSlicingMixin`` for a description how slicing works (which
    attributes) are sliced.

    """
    def __init__(self, data, uncertainty=None, mask=None, wcs=None,
                 meta=None, unit=None, copy=False, window=None, variance=None):

        if variance is not None:
            if uncertainty is not None:
                raise ValueError()
            uncertainty = ADVarianceUncertainty(variance)

        super().__init__(FakeArray(data) if is_lazy(data) else data,
                         None if is_lazy(uncertainty) else uncertainty,
                         mask, wcs, meta, unit, copy)

        if is_lazy(data):
            self.data = data
        if is_lazy(uncertainty):
            self.uncertainty = uncertainty

    def __deepcopy__(self, memo):
        new = self.__class__(
            self._data if is_lazy(self._data) else deepcopy(self.data, memo),
            self._uncertainty if is_lazy(self._uncertainty) else None,
            self._mask if is_lazy(self._mask) else deepcopy(self.mask, memo),
            deepcopy(self.wcs, memo), None, self.unit
        )
        new.meta = deepcopy(self.meta, memo)
        # Needed to avoid recursion because of uncertainty's weakref to self
        if not is_lazy(self._uncertainty):
            new.variance = deepcopy(self.variance)
        return new

    @property
    def window(self):
        """
        Interface to access a section of the data, using lazy access whenever
        possible.

        Returns
        --------
        An instance of ``NDWindowing``, which provides ``__getitem__``,
        to allow the use of square brackets when specifying the window.
        Ultimately, an ``NDWindowingAstrodata`` instance is returned.

        Examples
        ---------

        >>> ad[0].nddata.window[100:200, 100:200]  # doctest: +SKIP
        <NDWindowingAstrodata .....>

        """
        return NDWindowing(self)

    def _get_uncertainty(self, section=None):
        """Return the ADVarianceUncertainty object, or a slice of it."""
        if self._uncertainty is not None:
            if is_lazy(self._uncertainty):
                if section is None:
                    self.uncertainty = ADVarianceUncertainty(self._uncertainty.data)
                    return self.uncertainty
                else:
                    return ADVarianceUncertainty(self._uncertainty[section])
            elif section is not None:
                return self._uncertainty[section]
            else:
                return self._uncertainty

    def _get_simple(self, target, section=None):
        """Only use 'section' for image-like objects that have the same shape
        as the NDAstroData object; otherwise, return the whole object"""
        source = getattr(self, target)
        if source is not None:
            if is_lazy(source):
                if section is None:
                    ret = np.empty(source.shape, dtype=source.dtype)
                    ret[:] = source.data
                    setattr(self, target, ret)
                else:
                    ret = source[section]
                return ret
            elif hasattr(source, 'shape'):
                if section is None or source.shape != self.shape:
                    return np.array(source, copy=False)
                else:
                    return np.array(source, copy=False)[section]
            else:
                return source

    @property
    def data(self):
        """
        An array representing the raw data stored in this instance.
        It implements a setter.
        """
        return self._get_simple('_data')

    @data.setter
    def data(self, value):
        if value is None:
            raise ValueError("Cannot have None as the data value for an NDData object")

        if is_lazy(value):
            self.meta['header'] = value.header
        self._data = value

    @property
    def uncertainty(self):
        return self._get_uncertainty()

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None and not is_lazy(value):
            if value._parent_nddata is not None:
                value = value.__class__(value, copy=False)
            value.parent_nddata = self
        self._uncertainty = value

    @property
    def mask(self):
        return self._get_simple('_mask')

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def variance(self):
        """
        A convenience property to access the contents of ``uncertainty``,
        squared (as the uncertainty data is stored as standard deviation).
        """
        arr = self._get_uncertainty()
        if arr is not None:
            return arr.array

    @variance.setter
    def variance(self, value):
        self.uncertainty = (ADVarianceUncertainty(value) if value is not None
                            else None)

    def set_section(self, section, input):
        """
        Sets only a section of the data. This method is meant to prevent
        fragmentation in the Python heap, by reusing the internal structures
        instead of replacing them with new ones.

        Args
        -----
        section : ``slice``
            The area that will be replaced
        input : ``NDData``-like instance
            This object needs to implement at least ``data``, ``uncertainty``,
            and ``mask``. Their entire contents will replace the data in the
            area defined by ``section``.

        Examples
        ---------

        >>> sec = NDData(np.zeros((100,100)))  # doctest: +SKIP
        >>> ad[0].nddata.set_section((slice(None,100),slice(None,100)), sec)  # doctest: +SKIP

        """
        self.data[section] = input.data
        if self.uncertainty is not None and getattr(input, 'uncertainty', None) is not None:
            self.uncertainty.array[section] = input.uncertainty.array
        if self.mask is not None and getattr(input, 'mask', None) is not None:
            self.mask[section] = input.mask

    def __repr__(self):
        if is_lazy(self._data):
            return self.__class__.__name__ + '(Memmapped)'
        else:
            return super().__repr__()

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        unc = self.uncertainty
        new_wcs = deepcopy(self.wcs)
        inframe = new_wcs.input_frame
        new_wcs.insert_transform(inframe, models.Mapping(tuple(reversed(range(inframe.naxes)))), after=True)
        return self.__class__(
            self.data.T,
            uncertainty=None if unc is None else unc.__class__(unc.array.T),
            mask=None if self.mask is None else self.mask.T, wcs=new_wcs,
            meta=self.meta, copy=False
        )

    def _slice(self, item):
        """Additionally slice things like OBJMASK"""
        kwargs = super()._slice(item)
        if 'other' in kwargs['meta']:
            kwargs['meta'] = deepcopy(self.meta)
            for k, v in kwargs['meta']['other'].items():
                if isinstance(v, np.ndarray) and v.shape == self.shape:
                    kwargs['meta']['other'][k] = v[item]
        return kwargs
