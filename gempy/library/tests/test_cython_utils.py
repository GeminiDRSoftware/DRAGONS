import pytest

import numpy as np

from gempy.library import cython_utils



def test_masked_median():
    rng = np.random.default_rng(42)

    data = rng.normal(size=(100, 100)).astype(np.float32)
    mask = (rng.random(size=data.shape) > 0.99).astype(np.uint16)

    y = np.ma.masked_array(data, mask=mask)
    assert np.ma.median(y) == pytest.approx(cython_utils.masked_median(data.ravel(), mask.ravel(), data.size), rel=1e-5)
