"""
Tests for the astrotools module.
"""

import numpy as np
import pytest
from gempy.library import astrotools as at
from astropy.coordinates import SkyCoord
from astropy import units as u

from astrodata.nddata import NDAstroData


@pytest.fixture(scope='module')
def flat_images():
    # Produce 6 NDAstroData objects with different mean values and some noise.
    rng = np.random.default_rng(42)
    images = []
    for i in range(6):
        img = NDAstroData(data=rng.normal(loc=50*(i+1), scale=20, size=(100, 100)).astype(np.float32),
                          mask=(np.random.rand(100, 100) > 0.99).astype(np.uint16))
        images.append(img)
    return images


def test_array_from_list():
    values = (1, 2, 3)
    unit = u.m
    x = [i * unit for i in values]
    result = at.array_from_list(x)
    assert result.unit is unit
    np.testing.assert_array_equal(result.value, values)


def test_boxcar_logical_or():
    x = np.zeros((100,), dtype=bool)
    x[45:55] = True
    y = at.boxcar(x, size=2, operation=np.logical_or)
    assert y.sum() == 14


def test_boxcar_median():
    x = np.zeros((100,))
    x[45:55] = 1
    y = at.boxcar(x, size=2)
    np.testing.assert_allclose(x, y)


def test_calculate_pixel_edges():
    x = np.arange(100)
    edges = at.calculate_pixel_edges(x)
    np.testing.assert_allclose(edges, np.arange(-0.5, 100.4))


def test_calculate_scaling_without_outlier_removal():
    x = np.arange(1, 11)
    y = 2 * x
    y[3:5] = [5, 7]
    sigma_x = np.ones_like(x)
    sigma_y = np.full_like(x, 2.)
    assert abs(at.calculate_scaling(x, y, sigma=None) - 1.92987013) < 1e-7
    assert abs(at.calculate_scaling(x, y, sigma_x=sigma_x, sigma=None)
               - 1.95154778) < 1e-7
    assert abs(at.calculate_scaling(x, y, sigma_y=sigma_y, sigma=None)
               - 1.92987013) < 1e-7
    assert abs(at.calculate_scaling(x, y, sigma_x=sigma_x, sigma_y=sigma_y, sigma=None)
               - 1.95100518) < 1e-7


def test_calculate_scaling_with_outlier_removal():
    x = np.arange(1, 11)
    y = 2 * x
    y[3:5] = [5, 7]
    sigma_x = np.ones_like(x)
    sigma_y = np.full_like(x, 2.)
    assert abs(at.calculate_scaling(x, y, sigma=2, niter=2) - 2) < 1e-7
    assert abs(at.calculate_scaling(x, y, sigma_x=sigma_x, sigma=2, niter=2)
               - 2) < 1e-7
    assert abs(at.calculate_scaling(x, y, sigma_y=sigma_y, sigma=2, niter=2)
               - 2) < 1e-7
    assert abs(at.calculate_scaling(x, y, sigma_x=sigma_x, sigma_y=sigma_y, sigma=2, niter=2)
               - 2) < 1e-7


def test_divide0():
    ones = np.array([1, 1, 1])
    zeros = np.array([0, 0, 0])
    twod = np.arange(12).reshape(4, 3)

    # scalar / scalar
    assert at.divide0(1, 0) == 0
    # array / scalar
    np.testing.assert_array_equal(at.divide0(ones, 0), zeros)
    # scalar / array
    np.testing.assert_array_equal(at.divide0(1, zeros), zeros)
    # array / array
    np.testing.assert_array_equal(at.divide0(ones, zeros), zeros)

    # 2d array / 1d array
    np.testing.assert_array_equal(at.divide0(twod, ones), twod)
    np.testing.assert_array_equal(at.divide0(twod, zeros), np.zeros_like(twod))


def test_fit_spline_to_data():
    rng = np.random.default_rng(0)
    x = np.arange(100)
    y = np.ones_like(x)
    spline = at.fit_spline_to_data(y)
    np.testing.assert_allclose(spline(x), y)

    # Test with a curve and noise
    y = x + x*x + rng.normal(size=x.size)
    spline = at.fit_spline_to_data(y)
    np.testing.assert_allclose(spline(x), y, atol=3)

    # Test with some masking
    mask = np.zeros_like(x, dtype=bool)
    safe = y[10]
    y[10] = 1000
    mask[10] = True
    spline = at.fit_spline_to_data(y, mask=mask)
    y[10] = safe
    np.testing.assert_allclose(spline(x), y, atol=3)


# test both old and new behavior, the values are slightly different
@pytest.mark.parametrize("subtract,limit", ([False, 0.02], [True, 0.2]))
@pytest.mark.parametrize("separation", [1,2,3,4,5])
def test_std_from_pixel_variations(separation, subtract, limit):
    # Test passes with ths seed and number of samples
    rng = np.random.default_rng(1)
    data = rng.normal(size=10000)
    assert abs(at.std_from_pixel_variations(
        data, separation=separation, subtract_linear_fits=subtract) - 1) < limit


@pytest.mark.parametrize("data,reference",
                         [([15.2, 15.6, 14.9, 25.4], np.array([False, False, False, True])),
                          ([0, 17, 0, 0], np.array([False, True, False, False])),
                          ([0, 0, 0, 0], np.array([False, False, False, False])),
                          ([1, 1.1, 1.1, 1], np.array([False, False, False, False]))])
def test_find_outliers(data, reference):
    # Test that it works correctly when some values are repeated.
    assert np.allclose(at.find_outliers(data), reference)


def test_get_corners_2d():
    corners = at.get_corners((300, 500))
    assert corners == [(0, 0), (299, 0), (0, 499), (299, 499)]


def test_get_corners_3d():
    corners = at.get_corners((300, 500, 400))
    expected_corners = [(0, 0, 0), (299, 0, 0), (0, 499, 0),
                        (299, 499, 0), (0, 0, 399), (299, 0, 399),
                        (0, 499, 399), (299, 499, 399)]
    assert corners == expected_corners


def test_clipped_mean():
    dist = np.array([5, 5, 4, 7, 7, 4, 3, 5, 2, 6, 5, 12, 0,
                     9, 10, 13, 2, 14, 6, 3, 50])
    results = at.clipped_mean(dist)
    expected_values = (6.1, 3.7)
    assert np.allclose(results, expected_values)


def test_parse_user_regions():
    parse = at.parse_user_regions
    assert parse("*") == [(None, None)]
    assert parse("") == [(None, None)]
    assert parse(None) == [(None, None)]
    assert parse("1:10,20:50") == [(1, 10), (20, 50)]
    assert parse("1:10.2,20:50.2", dtype=float) == [(1, 10.2), (20, 50.2)]
    assert parse("1:10,20:50:2", allow_step=True) == [(1, 10), (20, 50, 2)]
    with pytest.raises(ValueError):
        parse("1:10:2")
    assert parse("50:20") == [(20, 50), ]


def test_cartesian_regions_to_slices():
    cart = at.cartesian_regions_to_slices
    assert cart('')[0] == slice(None)
    assert cart('1:10')[0] == slice(0, 10)
    assert cart('[1:10]')[0] == slice(0, 10)
    assert cart('1:10:2')[0] == slice(0, 10, 2)
    assert cart('1-10:2')[0] == slice(0, 10, 2)
    assert cart('1:10,20:30') == (slice(19, 30), slice(0, 10))
    assert cart('[:,:10]') == (slice(None, 10), slice(None))

    assert cart('100,*') == (slice(None), slice(99, 100))
    assert cart('100, *') == (slice(None), slice(99, 100))
    assert cart(':100,*') == (slice(None), slice(100))

    assert cart('*,12') == (slice(11, 12), slice(None))
    assert cart('*, 12:') == (slice(11, None), slice(None))

    with pytest.raises(TypeError):
        cart(12)


@pytest.mark.parametrize("return_scaling", (True, False))
def test_optimal_normalization(flat_images, return_scaling):
    """
    Quick test to check image scaling/offsetting. This doesn't test the
    memory-mapping part of the function.
    """
    retval = at.optimal_normalization(flat_images, return_scaling=return_scaling)

    if return_scaling:
        np.testing.assert_allclose(retval, 1. / (np.arange(len(flat_images)) + 1), rtol=0.01)
    else:
        np.testing.assert_allclose(retval, -np.arange(len(flat_images)) * 50, atol=1.0)


@pytest.mark.parametrize("separate_ext", (True, False))
def test_optimal_normalization_multiple_extensions(flat_images, separate_ext):
    """
    Confirm that the function works with multiple extensions, either computing
    the offsets separately or together.
    """
    # Pass the list as 2 images with 3 extensions each
    retval = at.optimal_normalization(flat_images, return_scaling=True,
                                      num_ext=3, separate_ext=separate_ext)

    if separate_ext:
        assert retval.shape == (3, 2)  # 3 extensions, 2 images
        np.testing.assert_allclose(retval, [[1., 0.25], [1., 0.4], [1., 0.5]], rtol=0.01)
    else:
        # Because the data *don't* have a common scaling, the result here
        # depends on how one decides to calculate the average scaling.
        # The extensions in image 1 have signals (50, 100, 150), and in
        # image 2 (200, 250, 300), so the divided image will have 1/3 pixels
        # ~0.25, 1/3 being ~0.4, and 1/3 being ~0.5; hence median is 0.4
        np.testing.assert_allclose(retval, [1, 0.4], rtol=0.01)


def test_spherical_offsets_by_pa():
    c1 = SkyCoord(ra=120, dec=0, unit='deg')
    c2 = SkyCoord(ra=120.01, dec=0.05, unit='deg')
    assert np.allclose(at.spherical_offsets_by_pa(c1, c2, position_angle=0),
                       (180, 36), atol=1e-5)
    assert np.allclose(at.spherical_offsets_by_pa(c1, c2, position_angle=90),
                       (36, -180), atol=1e-5)
    assert np.allclose(at.spherical_offsets_by_pa(c1, c2, position_angle=-90),
                       (-36, 180), atol=1e-5)


def test_weighted_sigma_clip():
    from astropy.stats import sigma_clipped_stats
    x = np.arange(20) ** 2
    mean_astropy = sigma_clipped_stats(x, sigma=2, cenfunc='mean')[0]
    mean_astrotools = at.weighted_sigma_clip(x, sigma=2).mean()
    assert mean_astropy == pytest.approx(mean_astrotools)


def test_magnitude_wavelengths():
    """
    Confirm behaviour that a Quantity is returned without units, and
    a float if units are provided
    """
    m = at.Magnitude("J=10")
    assert isinstance(m.wavelength(), u.Quantity)
    assert m.wavelength().to(u.um).value == pytest.approx(1.25, abs=0.01)
    assert m.wavelength(units="um") == pytest.approx(1.25, abs=0.01)


@pytest.mark.parametrize("filter_name", [k for k in at.Magnitude.VEGA_INFO])
def test_magnitude_flux_densities_ab(filter_name):
    """
    Check that we get the same flux density for all things with ABmag=0
    """
    m = at.Magnitude(f"{filter_name}=0", abmag=True)
    assert isinstance(m.flux_density(), u.Quantity)
    assert m.flux_density().to("Jy").value == pytest.approx(3630, rel=0.001)
    assert m.flux_density(units="Jy") == pytest.approx(3630, rel=0.001)


@pytest.mark.parametrize("num_values", [1, 2, 3, 4, 5])
def test_weighted_median_equal_weights(num_values):
    x = np.arange(num_values)
    w = np.ones_like(x)
    assert at.weighted_median(x, w) == pytest.approx(at.weighted_median(x), abs=0.001)


def test_weighted_median_unequal_weights():
    x = np.arange(10)
    w = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    assert at.weighted_median(x, w) == pytest.approx(6, abs=0.001)
    x = np.arange(9)
    w = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2])
    assert at.weighted_median(x, w) == pytest.approx(5.5, abs=0.001)
