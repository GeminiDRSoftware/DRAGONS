
import numpy as np
import warnings
import pytest

from astrodata import nddata


def test_new_variance_uncertainty_instance_returns_none():

    result = nddata.new_variance_uncertainty_instance(None)

    assert result is None


def test_new_variance_uncertainty_instance_warns_negative_values():

    array = np.zeros((500, 500), dtype=np.float32)
    array[200, 300] = -0.001

    with warnings.catch_warnings(record=True) as w:

        result = nddata.new_variance_uncertainty_instance(array)
        warnings.simplefilter("always")


        for _w in w:
            print(str(_w.message))

        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)


def test_new_variance_uncertainty_instance():

    array = np.ones((500, 500), dtype=np.float32)

    with warnings.catch_warnings(record=True) as w:

        result = nddata.new_variance_uncertainty_instance(array)
        warnings.simplefilter("always")

        assert len(w) == 0


if __name__ == '__main__':
    pytest.main()