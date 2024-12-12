"""
Tests for the astromodels module.
"""
import pytest
import numpy as np

from astropy.modeling import models
from astropy import units as u
from scipy.interpolate import BSpline

from gempy.library import astromodels as am


@pytest.mark.parametrize("model", ("Chebyshev1D", "Legendre1D", "Polynomial1D"))
def test_astropy1d_table_recovery(model):
    """Convert a model to a Table and back"""
    m = getattr(models, model)(degree=3, c0=0, c1=1, c2=0.5, c3=0.25)
    if model != "Polynomial1D":
        m.domain = (-10, 10)
    m.meta = {"xunit": u.nm, "yunit": u.electron}

    t = am.model_to_table(m)
    m2 = am.table_to_model(t)

    assert m.__class__ == m2.__class__
    assert m.domain == m2.domain
    for p in m.param_names:
        assert getattr(m, p) == getattr(m2, p)

    # Need this mess because u.Unit("nm") == "nm"
    keys1 = list(m.meta.keys())
    keys2 = list(m2.meta.keys())
    assert keys1 == keys2
    assert all(m.meta[k1] is m2.meta[k2] for k1, k2 in zip(keys1, keys2))


@pytest.mark.parametrize("k", (1, 2, 3, 4, 5))
def test_spline_table_recovery(k):
    """Convert a spline to a Table and back"""
    order = 6  # number of spline pieces
    knots = np.concatenate([[0.] * k, np.linspace(0, 10, order+1), [10.] * k])
    coeffs = np.concatenate([np.ones((order+k,)), [0.] * (k+1)])
    m = BSpline(knots, coeffs, k)

    t = am.model_to_table(m, xunit=u.nm, yunit=u.electron)
    m2 = am.table_to_model(t)

    np.testing.assert_array_equal(m.t, m2.t)
    np.testing.assert_array_equal(m.c, m2.c)
    assert m.k == m2.k
    assert m2.meta["xunit"] is u.nm
    assert m2.meta["yunit"] is u.electron
