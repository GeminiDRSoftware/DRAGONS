# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.utils.compat import ignored
from astropy.units import Quantity
from astropy.nddata import NDUncertainty
from astropy.nddata import MissingDataAssociationException
from astropy.nddata import IncompatibleUncertaintiesException

class VarUncertainty(NDUncertainty):
    """
    A class for standard deviation uncertainties
    """

    support_correlated = False

    def __init__(self, array=None, copy=True):
        self._unit = None
        self.uncertainty_type = 'var'
        if array is None:
            self.array = None
        elif isinstance(array, VarUncertainty):
            self.array = np.array(array.array, copy=copy, subok=True)
        elif isinstance(array, Quantity):
            self.array = np.array(array.value, copy=copy, subok=True)
            self._unit = array.unit
        else:
            self.array = np.array(array, copy=copy, subok=True)

    @property
    def parent_nddata(self):
        message = "Uncertainty is not associated with an NDData object"
        try:
            if self._parent_nddata is None:
                raise MissingDataAssociationException(message)
            else:
                return self._parent_nddata
        except AttributeError:
            raise MissingDataAssociationException(message)

    @parent_nddata.setter
    def parent_nddata(self, value):
        if self.array is None or value is None:
            self._parent_nddata = value
        else:
            if value.data.shape != self.array.shape:
                raise ValueError("parent shape does not match "
                                 "array data shape")
        self._parent_nddata = value

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        if value is not None:
            with ignored(MissingDataAssociationException):
                if value.shape != self.parent_nddata.data.shape:
                    raise ValueError("array shape does not match "
                                     "parent data shape")
        self._array = value

    def propagate_add(self, other_nddata, result_data):

        if not isinstance(other_nddata.uncertainty, VarUncertainty):
            raise IncompatibleUncertaintiesException

        if self.array is None:
            raise ValueError("standard deviation values are not set")

        if other_nddata.uncertainty.array is None:
            raise ValueError("standard deviation values are not set "
                             "in other_nddata")

        result_uncertainty = VarUncertainty()
        result_uncertainty.array = self.array + other_nddata.uncertainty.array

        return result_uncertainty

    def __getitem__(self, item):
        new_array = self.array[item]
        return self.__class__(new_array, copy=False)

    def propagate_subtract(self, other_nddata, result_data):

        if not isinstance(other_nddata.uncertainty, VarUncertainty):
            raise IncompatibleUncertaintiesException

        if self.array is None:
            raise ValueError("standard deviation values are not set")

        if other_nddata.uncertainty.array is None:
            raise ValueError("standard deviation values are not set "
                             "in other_nddata")

        result_uncertainty = VarUncertainty()
        result_uncertainty.array = self.array + other_nddata.uncertainty.array

        return result_uncertainty

    def propagate_multiply(self, other_nddata, result_data):

        if not isinstance(other_nddata.uncertainty, VarUncertainty):
            raise IncompatibleUncertaintiesException

        if self.array is None:
            raise ValueError("standard deviation values are not set")

        if other_nddata.uncertainty.array is None:
            raise ValueError("standard deviation values are not set in "
                             "other_nddata")

        result_uncertainty = VarUncertainty()
        AdBsq = self.parent_nddata.data*2*other_nddata.uncertainty.array
        BdAsq = self.array*other_nddata.data**2
        result_uncertainty.array = AdBsq + BdAsq

        return result_uncertainty

    def propagate_divide(self, other_nddata, result_data):

        if not isinstance(other_nddata.uncertainty, VarUncertainty):
            raise IncompatibleUncertaintiesException

        if self.array is None:
            raise ValueError("standard deviation values are not set")

        if other_nddata.uncertainty.array is None:
            raise ValueError("standard deviation values are not set "
                             "in other_nddata")

        result_uncertainty = VarUncertainty()
        BdAsq = self.array*other_nddata.data**2
        AdBoverBsqsq = self.parent_nddata.array**2*other.nddata.uncertainty.array \
            / other.nddata.data**4
        result_uncertainty.array = BdA + AdBoverBsq
        return result_uncertainty
