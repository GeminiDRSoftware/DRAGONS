#!/usr/bin/env python
import warnings

import numpy as np
import pytest

from astrodata import nddata


def test_variance_uncertainty_warn_if_there_are_any_negative_numbers():
    arr = np.zeros((5, 5))
    arr[2, 2] = -0.001

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = nddata.VarianceUncertainty(arr)

        assert not (arr >= 0).all()
        assert len(w) == 1
        assert isinstance(result, nddata.StdDevUncertainty)
        assert result.array[250, 250] == 0


if __name__ == '__main__':
    pytest.main()
