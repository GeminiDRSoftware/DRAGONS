"""
This module implements a derivative class based on NDData with some Mixins,
implementing windowing and on-the-fly data scaling.
"""

from __future__ import (absolute_import, division, print_function)

from copy import deepcopy

from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.nddata.mixins.ndarithmetic import NDArithmeticMixin
from astropy.io.fits import ImageHDU
import numpy as np

__all__ = ['NDAstroData']

class StdDevAsVariance(object):
    def as_variance(self):
        if self.array is not None:
            return self.array ** 2
        else:
            return None

def new_variance_uncertainty_instance(array):
    obj = StdDevUncertainty(np.sqrt(array))
    cls = obj.__class__
    obj.__class__ = cls.__class__(cls.__name__ + "WithAsVariance", (cls, StdDevAsVariance), {})
    return obj

class FakeArray(object):
    def __init__(self, very_faked):
        self.data = very_faked
        self.shape = (100, 100) # Won't matter. This is just to fool NDData
        self.dtype = np.float32 # Same here
        self.__array__ = very_faked
    def __getitem__(self, index):
        # FAKE NEWS!
        return None

class FakeData(object):
    def __init__(self, data, mask):
        self.data = FakeArray(data)
        self.mask = mask

class NDWindowing(object):
    def __init__(self, target):
        self._target = target

    def __getitem__(self, slice):
        return NDWindowingAstroData(self._target, window=slice)

class NDWindowingAstroData(NDArithmeticMixin, NDSlicingMixin, NDData):
    """
    """
    def __init__(self, target, window):
        self._target = target
        self._window = window

    @property
    def unit(self):
        return self._target.unit

    @property
    def wcs(self):
        return self._target.wcs

    @property
    def data(self):
        return self._target._get_data(section=self._window)

    @property
    def uncertainty(self):
        return self._target._get_uncertainty(section=self._window)

    @property
    def mask(self):
        return self._target._get_mask(section=self._window)

class NDAstroData(NDArithmeticMixin, NDSlicingMixin, NDData):
    """Implements `NDData` with all Mixins.

    This class implements a `NDData`-like container that supports reading and
    writing as implemented in the ``astropy.io.registry`` and also slicing
    (indexing) and simple arithmetics (add, subtract, divide and multiply).

    Notes
    -----
    A key distinction from `NDDataArray` is that this class does not attempt
    to provide anything that was not defined in any of the parent classes.

    See also
    --------
    NDData
    NDArithmeticMixin
    NDSlicingMixin

    Examples
    --------
    The mixins allow operation that are not possible with `NDData` or
    `NDDataBase`, i.e. simple arithmetics::

        >>> from astropy.nddata import NDAstroData, StdDevUncertainty
        >>> import numpy as np

        >>> data = np.ones((3,3), dtype=np.float)
        >>> ndd1 = NDAstroData(data, uncertainty=StdDevUncertainty(data))
        >>> ndd2 = NDAstroData(data, uncertainty=StdDevUncertainty(data))

        >>> ndd3 = ndd1.add(ndd2)
        >>> ndd3.data
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.],
               [ 2.,  2.,  2.]])
        >>> ndd3.uncertainty.array
        array([[ 1.41421356,  1.41421356,  1.41421356],
               [ 1.41421356,  1.41421356,  1.41421356],
               [ 1.41421356,  1.41421356,  1.41421356]])

    see `NDArithmeticMixin` for a complete list of all supported arithmetic
    operations.

    But also slicing (indexing) is possible::

        >>> ndd4 = ndd3[1,:]
        >>> ndd4.data
        array([ 2.,  2.,  2.])
        >>> ndd4.uncertainty.array
        array([ 1.41421356,  1.41421356,  1.41421356])

    See `NDSlicingMixin` for a description how slicing works (which attributes)
    are sliced.
    """
    def __init__(self, data, uncertainty=None, mask=None, wcs=None,
                 meta=None, unit=None, copy=False, window=None):
        self._scalers = {}
        self._window = window
        self._lazy = {
                'data': False,
                'uncertainty': False,
                'mask': False
                }

        if isinstance(data, ImageHDU):
            kdata = FakeData(data, mask)
            kmask = None
            kuncert = None
        else:
            kdata = data
            kmask = mask
            kuncert = uncertainty
        super(NDAstroData, self).__init__(kdata, kuncert, kmask,
                                          wcs, meta, unit, copy)

        if isinstance(data, ImageHDU):
            self._data = self._data.__array__
            self._lazy['data'] = True
            self._set_scaling('data', data.header)
            self.meta['header'] = data.header
            self.uncertainty = uncertainty

    def __deepcopy__(self, memo):
        return self.__class__(self.data, self.uncertainty, self.mask, self.wcs,
                              deepcopy(self.meta), self.unit)

    def _set_scaling(self, key, header):
        bzero, bscale = header.get('BZERO'), header.get('BSCALE')
        if bzero is None and bscale is None:
            self._scalers[key] = lambda x: x
        else:
            self._scalers[key] = lambda x: (bscale * x) + bzero

    @property
    def window(self):
        return NDWindowing(self)

    @property
    def shape(self):
        return self._data.shape

    def _extract(self, source, scaling, section=None):
        return scaling(source.data if section is None else source.section[section])

    def _get_data(self, section=None):
        if self._lazy['data']:
            temp = self._extract(self._data, self._scalers['data'], section=section)
            if section is None:
                self._lazy['data'] = False
                self._data = temp
            return temp
        elif section is not None:
            return self._data[section]
        else:
            return self._data

    def _get_uncertainty(self, section=None):
        if self._uncertainty is not None:
            if self._lazy['uncertainty']:
                temp = new_variance_uncertainty_instance(self._extract(self._uncertainty, self._scalers['uncertainty'], section=section))
                if section is None:
                    self.uncertainty = new_variance_uncertainty_instance(self._extract(self._uncertainty, self._scalers['uncertainty']))
                return temp
            elif section is not None:
                return self._uncertainty[section]
            else:
                return self._uncertainty

    def _get_mask(self, section=None):
        if self._mask is not None:
            if self._lazy['mask']:
                temp = self._extract(self._mask, self._scalers['mask'], section=section)
                if section is None:
                    self._lazy['data'] = False
                    self._mask = temp
                return temp
            elif section is not None:
                return self._mask[section]
            else:
                return self._mask

    @property
    def data(self):
        return self._get_data()

    @property
    def uncertainty(self):
        return self._get_uncertainty()

    @uncertainty.setter
    def uncertainty(self, value):
        if isinstance(value, ImageHDU):
            self._lazy['uncertainty'] = True
            self._set_scaling('uncertainty', value.header)
            self._uncertainty = value
        else:
            self._lazy['uncertainty'] = False
            if value is not None:
                if value._parent_nddata is not None:
                    value = value.__class__(value, copy=False)
                value.parent_nddata = self
            self._uncertainty = value

    @property
    def mask(self):
        if self._mask is not None:
            if not self._lazy['mask']:
                return super(NDAstroData, self).mask
            return self._extract(self._mask, self._scalers['mask'])

    @mask.setter
    def mask(self, value):
        if isinstance(value, ImageHDU):
            self._lazy['mask'] = True
            self._set_scaling('mask', value.header)
        else:
            self._lazy['mask'] = False
        self._mask = value

    def set_section(self, section, input):
        self.data[section] = input.data
        if self.uncertainty is not None:
            self.uncertainty[section] = input.uncertainty
        if self.mask is not None:
            self.mask[section] = input.mask

    def __repr__(self):
        if self._lazy['data']:
            return self.__class__.__name__ + '(Memmapped)'
        else:
            return super(NDAstroData, self).__repr__()

