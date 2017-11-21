"""
This module implements a derivative class based on NDData with some Mixins,
implementing windowing and on-the-fly data scaling.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.nddata import NDData
from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.nddata.mixins.ndarithmetic import NDArithmeticMixin
from astropy.io.fits import ImageHDU

__all__ = ['NDAstroData']


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
            self._lazy['data'] = True
            self._set_scaling('data', data.header)
            self._data = data
            self.uncertainty = uncertainty
            self.mask = mask
            self._wcs = wcs
            self._unit = unit
        else:
            super(NDAstroData, self).__init__(data, uncertainty, mask,
                                              wcs, meta, unit, copy)


    def _set_scaling(self, key, header):
        bzero, bscale = header.get('BZERO'), header.get('BSCALE')
        if bzero is None and bscale is None:
            self._scalers[key] = lambda x: x
        else:
            self._scalers[key] = lambda x: (bscale * x) + bzero

    def window(self, slice):
        return NDAstroData(data=self._data, uncertainty=self._uncertainty,
                mask=self._mask, wcs=self._wcs, meta=self.meta,
                unit=self._unit, window=slice)

    def _lazy_property(self, prop_name):
        if self._lazy[prop_name]:

    def _extract(self, source, scaling):
        return scaling(source.data if not self._window else source.section[self._window])

    @property
    def data(self):
        if self._lazy['data']:
            self._data = self._extract(self._data, self._scalers['data'])
        return self._data

    @property
    def uncertainty(self):
        if not self._lazy['uncertainty']:
            return super(NDAstroData, self).uncertainty
        return self._extract(self._uncertainty, self._scalers['uncertainty'])

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
        if not self._lazy['mask']:
            return super(NDAstroData, self).mask
        return self._extract(self._mask, self._scalers['mask'])

    @mask.setter
    def mask(self, value):
        if isinstance(value, ImageHDU):
            self._lazy['mask'] = True
            self._set_scaling('mask', value.header)
            self._mask = value
        else:
            self._lazy['mask'] = False
            self._mask = value
