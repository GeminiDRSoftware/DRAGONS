#!/usr/bin/env python

import numpy as np
import pytest

from astrodata.nddata import ADVarianceUncertainty


def test_variance_uncertainty_warn_if_there_are_any_negative_numbers():
    arr = np.zeros((5, 5))
    arr[2, 2] = -0.001

    with pytest.warns(RuntimeWarning, match='Negative variance values found.'):
        result = ADVarianceUncertainty(arr)

    assert not np.all(arr >= 0)
    assert isinstance(result, ADVarianceUncertainty)
    assert result.array[2, 2] == 0

    # check that it always works with a VarianceUncertainty instance
    result.array[2, 2] = -0.001

    with pytest.warns(RuntimeWarning, match='Negative variance values found.'):
        result2 = ADVarianceUncertainty(result)

    assert not np.all(arr >= 0)
    assert not np.all(result.array >= 0)
    assert isinstance(result2, ADVarianceUncertainty)
    assert result2.array[2, 2] == 0


def test_new_variance_uncertainty_instance_no_warning_if_the_array_is_zeros():
    arr = np.zeros((5, 5))
    with pytest.warns(None) as w:
        ADVarianceUncertainty(arr)
    assert len(w) == 0


if __name__ == '__main__':
    pytest.main()
