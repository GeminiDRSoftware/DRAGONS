"""
Tests for the astromodels module.
"""
import pytest

import itertools
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


def test_create_distortion_model():
    """Test of the ability to create a distortion model and its inverse"""
    # These are actual SLITEDGE models
    m1 = models.Chebyshev1D(degree=3, c0=176.45779746891014,
                            c1=193.23784274403934, c2=21.045307058459432,
                            c3=1.91811981332223, domain=(0,1023))
    m2 = models.Chebyshev1D(degree=3, c0=276.9670524738892,
                            c1=187.69722504387903, c2=18.239222297723266,
                            c3=1.1955380503269126, domain=(0,1023))
    y = np.arange(1024)
    x1, x2 = m1(y), m2(y)
    xorder, yorder = 1, m1.degree

    x1ref, x2ref = x1[511], x2[511]
    in_coords = np.asarray([list(x1) + list(x2),
                            list(y) + list(y)])
    ref_coords = np.asarray([[x1ref] * y.size + [x2ref] * y.size,
                             list(y) + list(y)])
    m_init = models.Chebyshev2D(x_degree=xorder, y_degree=yorder,
                                x_domain=[0, 1023], y_domain=[0, 1023])
    m2slits, m2final, m2inverse = am.create_distortion_model(
        m_init, 0, in_coords, ref_coords, fixed_linear=False)
    m1slit, m1final, m1inverse = am.create_distortion_model(
        m_init, 0, np.asarray([x1, y]),
        np.asarray([[x1ref] * y.size, y]), fixed_linear=True)

    assert m2slits.meta['fwd_rms'] < 0.05
    assert m2slits.meta['inv_rms'] < 0.05

    # Round-trip check. We would like the round-trip to be better, but it's
    # not actually possible, after experimentation.
    xt, yt = m2slits.inverse(*m2slits(*in_coords))
    np.testing.assert_allclose(yt, in_coords[1])
    np.testing.assert_allclose(xt, in_coords[0], atol=0.1)

    assert m1slit.meta['fwd_rms'] < 0.05
    assert m1slit.meta['inv_rms'] < 0.05

    # Round-trip check
    xt, yt = m1slit.inverse(*m1slit(x1, y))
    np.testing.assert_allclose(yt, y)
    np.testing.assert_allclose(xt, x1, atol=0.1)


def test_make_inverse_chebyshev1d(ntrials=100):
    """Rather simple test of predominantly linear models"""
    rng = np.random.default_rng(10)
    for trial in range(ntrials):
        # Ensure that higher-order terms are small compared to the linear term
        coeffs = {f"c{i + 2}": 10**(-i-1)*r for i, r in enumerate(rng.normal(size=(2,)))}
        inputs = np.arange(-1, 1.01, 0.01)
        m = models.Chebyshev1D(degree=3, c0=0, c1=10, **coeffs)
        outputs = m(inputs)
        minv = am.make_inverse_chebyshev1d(m, sampling=0.05, max_deviation=0.01)
        np.testing.assert_allclose(minv(outputs), inputs, atol=0.01)


@pytest.mark.parametrize("xdeg,ydeg,replace", itertools.product(range(1, 4), range(1, 4), "xy"))
def test_reduce_dimensionality_2d_to_1d(xdeg, ydeg, replace, ntrials=50):
    rng = np.random.default_rng(10)
    x, y = np.mgrid[:101, :101]
    x=x.astype(float).flatten()
    y=y.astype(float).flatten()
    for trial in range(ntrials):
        coeffs = {f"c{i % (xdeg+1)}_{i // (xdeg+1)}": r
                  for i, r in enumerate(rng.normal(size=(xdeg+1)*(ydeg+1)))}
        m = models.Chebyshev2D(x_degree=xdeg, y_degree=ydeg,
                               x_domain=(0,100), y_domain=(0,100), **coeffs)
        for val in 100 * rng.uniform(size=10):
            m2 = am.reduce_dimensionality(m, **{replace: val})
            if replace == "x":
                outputs2 = m(np.full_like(y, val), y)
                outputs3 = m2(y)
            else:
                outputs2 = m(x, np.full_like(x, val))
                outputs3 = m2(x)
            np.testing.assert_allclose(outputs2, outputs3)


@pytest.mark.parametrize("xdeg,ydeg,zdeg,replace",
                         itertools.product(range(1, 4), range(1, 4), range(1, 4), "xyz"))
def test_reduce_dimensionality_3d_to_2d(xdeg, ydeg, zdeg, replace, ntrials=3):
    from geminidr.igrins.cheb3d import Chebyshev3D
    rng = np.random.default_rng(10)
    x, y, z = np.mgrid[:101, :101, :101]
    x=x.astype(float).flatten()
    y=y.astype(float).flatten()
    z=z.astype(float).flatten()
    for trial in range(ntrials):
        coeffs = {f"c{i % (xdeg+1)}_{(i // (xdeg+1)) % (ydeg+1)}_{i // ((xdeg+1)*(ydeg+1))}": r
                  for i, r in enumerate(rng.normal(size=(xdeg+1)*(ydeg+1)*(zdeg+1)))}
        m = Chebyshev3D(x_degree=xdeg, y_degree=ydeg, z_degree=zdeg,
                        x_domain=(0,100), y_domain=(0,100), z_domain=(0,100),
                        **coeffs)
        for val in 100 * rng.uniform(size=3):
            m2 = am.reduce_dimensionality(m, **{replace: val})
            if replace == "x":
                outputs2 = m(np.full_like(y, val), y, z)
                outputs3 = m2(y, z)
            elif replace == "y":
                outputs2 = m(x, np.full_like(x, val), z)
                outputs3 = m2(x, z)
            else:
                outputs2 = m(x, y, np.full_like(x, val))
                outputs3 = m2(x, y)
            np.testing.assert_allclose(outputs2, outputs3)


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

