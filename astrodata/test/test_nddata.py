
import numpy as np
import warnings
import pytest

from astrodata import nddata


def test_new_variance_uncertainty_instance_returns_none():

    result = nddata.new_variance_uncertainty_instance(None)

    assert result is None


def test_new_variance_uncertainty_instance_no_warning_if_the_array_is_zeros():

    array = np.zeros((500, 500), dtype=np.float32)

    with warnings.catch_warnings(record=True) as w:

        result = nddata.new_variance_uncertainty_instance(array)
        warnings.simplefilter("always")

        for _w in w:
            print(str(_w.message))

        assert (array >= 0).all()
        assert len(w) == 0


def test_new_variance_uncertainty_instance_warn_if_there_are_any_negative_numbers():

    array = np.zeros((500, 500), dtype=np.float32)
    array[250, 250] = -0.001

    with warnings.catch_warnings(record=True) as w:
        result = nddata.new_variance_uncertainty_instance(array)
        warnings.simplefilter("always")

        for _w in w:
            print(str(_w.message))

        assert not (array >= 0).all()
        assert len(w) == 1
        assert isinstance(result, nddata.StdDevUncertainty)
        assert result.array[250, 250] == 0


def test_new_variance_uncertainty_instance():

    array = np.ones((500, 500), dtype=np.float32)

    with warnings.catch_warnings(record=True) as w:

        result = nddata.new_variance_uncertainty_instance(array)
        warnings.simplefilter("always")

        assert len(w) == 0
        assert isinstance(result, nddata.StdDevUncertainty)


if __name__ == '__main__':
    pytest.main()