"""
Tests applied to primitives_spect.py

Notes
-----

    For extraction tests, your input wants to be a 2D image with an `APERTURE`
    table attached. You'll see what happens if you take a spectrophotometric
    standard and run it through the standard reduction recipe, but the
    `APERTURE` table has one row per aperture with the following columns:

    - number : sequential list of aperture number

    - ndim, degree, domain_start, domain_end, c0, [c1, c2, c3...] : standard
    Chebyshev1D definition of the aperture centre (in pixels) as a function of
    pixel in the dispersion direction

    - aper_lower : location of bottom of aperture relative to centre (always
    negative)

    - aper_upper : location of top of aperture relative to centre (always
    positive)

    The ndim column will always be 1 since it's always 1D Chebyshev, but the
    `model_to_dict()` and `dict_to_model()` functions that convert the Model
    instance to a dict create/require this.
"""

import os

import numpy as np
import pytest
from astropy import table
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from scipy import optimize

from specutils.utils.wcs_utils import air_to_vac

from gempy.library import astromodels as am
from geminidr.core import primitives_spect
from geminidr.niri.primitives_niri_image import NIRIImage

# -- Tests --------------------------------------------------------------------


def test_extract_1d_spectra():
    # Input Parameters ----------------
    width = 200
    height = 100

    # Boilerplate code ----------------
    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data[height // 2] = 1
    ad[0].APERTURE = get_aperture_table(height, width)

    # Running the test ----------------
    _p = primitives_spect.Spect([])

    # todo: if input is a single astrodata,
    #  should not the output have the same format?
    ad_out = _p.extractSpectra([ad])[0]

    np.testing.assert_equal(ad_out[0].shape[0], ad[0].shape[1])
    np.testing.assert_allclose(ad_out[0].data, ad[0].data[height // 2], atol=1e-3)


def test_extract_1d_spectra_with_sky_lines():
    # Input Parameters ----------------
    width = 600
    height = 300
    source_intensity = 1

    # Boilerplate code ----------------
    np.random.seed(0)
    sky = fake_emission_line_spectrum(width, n_lines=20, max_intensity=1, fwhm=2.)
    sky = np.repeat(sky[np.newaxis, :], height, axis=0)

    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data += sky
    ad[0].data[height // 2] += source_intensity
    ad[0].APERTURE = get_aperture_table(height, width)

    # Running the test ----------------
    _p = primitives_spect.Spect([])

    # todo: if input is a single astrodata,
    #  should not the output have the same format?
    ad_out = _p.extractSpectra([ad])[0]

    np.testing.assert_equal(ad_out[0].shape[0], ad[0].shape[1])
    np.testing.assert_allclose(ad_out[0].data, source_intensity, atol=1e-3)


@pytest.mark.xfail(reason="The fake data needs a DQ plane")
def test_find_apertures():
    _p = primitives_spect.Spect([])
    _p.findApertures()


@pytest.mark.parametrize('in_vacuo', (False, True, None))
def test_get_spectrophotometry(path_to_outputs, in_vacuo):

    wavelengths = np.arange(350., 750., 10)

    def create_fake_table():

        flux = np.ones(wavelengths.size)
        bandpass = np.ones(wavelengths.size) * 5.

        _table = table.Table(
            [wavelengths, flux, bandpass],
            names=['WAVELENGTH', 'FLUX', 'FWHM'])

        _table.name = os.path.join(path_to_outputs, 'specphot.dat')
        _table.write(_table.name, format='ascii')

        return _table.name

    _p = primitives_spect.Spect([])
    fake_table = _p._get_spectrophotometry(create_fake_table(),
                                           in_vacuo=in_vacuo)
    np.testing.assert_allclose(fake_table['FLUX'], 1)

    assert 'WAVELENGTH_AIR' in fake_table.columns
    assert 'WAVELENGTH_VACUUM' in fake_table.columns
    assert 'FLUX' in fake_table.columns
    assert 'WIDTH' in fake_table.columns

    assert hasattr(fake_table['WAVELENGTH_AIR'], 'quantity')
    assert hasattr(fake_table['WAVELENGTH_VACUUM'], 'quantity')
    assert hasattr(fake_table['FLUX'], 'quantity')
    assert hasattr(fake_table['WIDTH'], 'quantity')

    if in_vacuo:
        np.testing.assert_allclose(fake_table['WAVELENGTH_VACUUM'], wavelengths)
    else:  # False or None
        np.testing.assert_allclose(fake_table['WAVELENGTH_AIR'], wavelengths)


def test_QESpline_optimization():
    """
    Test the optimization of the QESpline. This defines 3 regions, each of a
    different constant value, with gaps between them. The spline optimization
    should determine the relative offsets.
    """
    from geminidr.core.primitives_spect import QESpline

    gap = 20
    data_length = 300
    real_coeffs = [0.5, 1.2]

    # noinspection PyTypeChecker
    data = np.array([1] * data_length +
                    [0] * gap +
                    [real_coeffs[0]] * data_length +
                    [0] * gap +
                    [real_coeffs[1]] * data_length)

    masked_data = np.ma.masked_where(data == 0, data)
    xpix = np.arange(len(data))
    weights = np.where(data > 0, 1., 0.)
    boundaries = (data_length, 2 * data_length + gap)

    coeffs = np.ones((2,))
    order = 8

    result = optimize.minimize(
        QESpline, coeffs,
        args=(xpix, masked_data, weights, boundaries, order),
        tol=1e-8,
        method='Nelder-Mead'
    )

    np.testing.assert_allclose(real_coeffs, 1. / result.x, atol=0.01)


def test_sky_correct_from_slit():
    # Input Parameters ----------------
    width = 200
    height = 100
    n_sky_lines = 50

    # Simulate Data -------------------
    np.random.seed(0)

    source_model_parameters = {'c0': height // 2, 'c1': 0.0}

    source = fake_point_source_spatial_profile(
        height, width, source_model_parameters, fwhm=0.05 * height)

    sky = SkyLines(n_sky_lines, width - 1)

    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data += source
    ad[0].data += sky(ad[0].data, axis=1)

    # Running the test ----------------
    p = primitives_spect.Spect([])
    ad_out = p.skyCorrectFromSlit([ad], function="spline3", order=5,
                                  grow=2, niter=3, lsigma=3, hsigma=3,
                                  aperture_growth=2)[0]

    np.testing.assert_allclose(ad_out[0].data, source, atol=1e-3)


def test_sky_correct_from_slit_with_aperture_table():
    # Input Parameters ----------------
    width = 200
    height = 100
    n_sky_lines = 50

    # Simulate Data -------------------
    np.random.seed(0)

    source_model_parameters = {'c0': height // 2, 'c1': 0.0}

    source = fake_point_source_spatial_profile(
        height, width, source_model_parameters, fwhm=0.08 * height)

    sky = SkyLines(n_sky_lines, width - 1)

    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data += source
    ad[0].data += sky(ad[0].data, axis=1)
    ad[0].APERTURE = get_aperture_table(height, width)

    # Running the test ----------------
    p = primitives_spect.Spect([])
    ad_out = p.skyCorrectFromSlit([ad], function="spline3", order=5,
                                  grow=2, niter=3, lsigma=3, hsigma=3,
                                  aperture_growth=2)[0]

    np.testing.assert_allclose(ad_out[0].data, source, atol=1e-3)


def test_sky_correct_from_slit_with_multiple_sources():
    width = 200
    height = 100
    n_sky_lines = 50
    np.random.seed(0)

    y0 = height // 2
    y1 = 7 * height // 16
    fwhm = 0.05 * height

    source = (
        fake_point_source_spatial_profile(height, width, {'c0': y0, 'c1': 0.0}, fwhm=fwhm) +
        fake_point_source_spatial_profile(height, width, {'c0': y1, 'c1': 0.0}, fwhm=fwhm)
    )

    sky = SkyLines(n_sky_lines, width - 1)

    ad = create_zero_filled_fake_astrodata(height, width)

    ad[0].data += source
    ad[0].data += sky(ad[0].data, axis=1)
    ad[0].APERTURE = get_aperture_table(height, width, center=height // 2)
    # Ensure a new row is added correctly, regardless of column order
    new_row = {'number': 2, 'c0': y1, 'aper_lower': -3, 'aper_upper': 3}
    ad[0].APERTURE.add_row([new_row[c] for c in ad[0].APERTURE.colnames])

    # Running the test ----------------
    p = primitives_spect.Spect([])
    ad_out = p.skyCorrectFromSlit([ad], function="spline3", order=5,
                                  grow=2, niter=3, lsigma=3, hsigma=3,
                                  aperture_growth=2)[0]

    np.testing.assert_allclose(ad_out[0].data, source, atol=1e-3)


def test_trace_apertures():
    # Input parameters ----------------
    width = 400
    height = 200
    trace_model_parameters = {'c0': height // 2, 'c1': 5.0, 'c2': -0.5, 'c3': 0.5}

    # Boilerplate code ----------------
    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data += fake_point_source_spatial_profile(height, width, trace_model_parameters)
    ad[0].APERTURE = get_aperture_table(height, width)

    # Running the test ----------------
    _p = primitives_spect.Spect([])
    ad_out = _p.traceApertures([ad], order=len(trace_model_parameters) + 1)[0]

    keys = trace_model_parameters.keys()

    desired = np.array([trace_model_parameters[k] for k in keys])
    actual = np.array([ad_out[0].APERTURE[0][k] for k in keys])

    np.testing.assert_allclose(desired, actual, atol=0.18)


@pytest.mark.parametrize('unit', ("", "electron", "W / (m2 nm)"))
@pytest.mark.parametrize('flux_calibrated', (False, True))
@pytest.mark.parametrize('user_conserve', (False, True, None))
def test_flux_conservation_consistency(astrofaker, caplog, unit,
                                       flux_calibrated, user_conserve):
    # UNIT, FLUX_CALIBRATED, USER_CONSERVE: CORRECT_OUTCOME, WARN
    RESULTS = {("electron", False, None): (True, False),
               ("electron", False, True): (True, False),
               ("electron", False, False): (False, True),
               ("electron", True, None): (True, True),
               ("electron", True, True): (True, True),
               ("electron", True, False): (False, True),
               ("", False, None): (True, False),
               ("", False, True): (True, False),
               ("", False, False): (False, True),
               ("", True, None): (False, False),
               ("", True, True): (True, True),
               ("", True, False): (False, False),
               ("W / (m2 nm)", False, None): (False, False),
               ("W / (m2 nm)", False, True): (True, True),
               ("W / (m2 nm)", False, False): (False, False),
               ("W / (m2 nm)", True, None): (False, False),
               ("W / (m2 nm)", True, True): (True, True),
               ("W / (m2 nm)", True, False): (False, False)
               }

    ad = astrofaker.create("NIRI")
    ad.init_default_extensions()
    p = NIRIImage([ad])  # doesn't matter but we need a log object
    ad.hdr["BUNIT"] = unit
    conserve = primitives_spect.conserve_or_interpolate(ad[0],
                    user_conserve=user_conserve,
                    flux_calibrated=flux_calibrated, log=p.log)
    correct, warn = RESULTS[unit, flux_calibrated, user_conserve]
    assert conserve == correct
    warning_given = any("WARNING" in record.message for record in caplog.records)
    assert warn == warning_given


def test_resample_spec_table():

    waves_air = np.arange(3500, 7500.001, 100) * u.AA
    waves_vac = air_to_vac(waves_air)
    bandpass = np.full(waves_air.size, 5.) * u.nm
    flux = np.ones(waves_air.size) * u.Unit("W/(m^2 Hz)")
    spec_table = table.Table(
        [waves_air, waves_vac, flux, bandpass],
        names=['WAVELENGTH_AIR', 'WAVELENGTH_VAC', 'FLUX', 'WIDTH'])

    t = primitives_spect.resample_spec_table(spec_table, 0.1)

    assert len(t) == 4001
    np.testing.assert_allclose(t['WAVELENGTH_AIR'].quantity,
                               np.arange(350, 750.001, 0.1) * u.nm)
    assert all([bw.to(u.nm).value == 0.1 if i % 10 else 5
               for i, bw in enumerate(t['WIDTH'].quantity)])
    np.testing.assert_allclose(t['FLUX'].data, 1.0)


# --- Fixtures and helper functions -------------------------------------------


def create_zero_filled_fake_astrodata(height, width):
    """
    Helper function to generate a fake astrodata object filled with zeros.

    Parameters
    ----------
    height : int
        Output 2D array's number of rows.
    width : int
        Output 2D array's number of columns.

    Returns
    -------
    AstroData
        Single-extension zero filled object.
    """
    astrofaker = pytest.importorskip("astrofaker")

    data = np.zeros((height, width))

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = data

    ad = astrofaker.create('GMOS-S')
    ad.add_extension(hdu, pixel_scale=1.0)

    return ad


def fake_point_source_spatial_profile(height, width, model_parameters, fwhm=5):
    """
    Generates a 2D array with a fake point source with constant intensity in the
    spectral dimension and a gaussian distribution in the spatial dimension. The
    center of the gaussian changes depends on the Chebyshev1D model defined
    by the input parameters.

    Parameters
    ----------
    height : int
        Output 2D array's number of rows.
    width : int
        Output 2D array's number of columns.
    model_parameters : dict
        Model parameters with keys defined as 'c0', 'c1', ..., 'c{n-1}', where
        'n' is the Chebyshev1D order.
    fwhm : float
        Full-width at half-maximum of the gaussian profile.

    Returns
    -------
    np.ndarray
        2D array with a fake point source
    """
    order = len(model_parameters) + 1

    trace_model = models.Chebyshev1D(
        order, domain=[0, width - 1], **model_parameters)

    x = np.arange(width)
    y = trace_model(x)
    n = y.size

    gaussian_model = models.Gaussian1D(
        mean=y,
        amplitude=[1] * n,
        stddev=[fwhm / (2. * np.sqrt(2 * np.log(2)))] * n,
        n_models=n
    )

    source = gaussian_model(np.arange(height), model_set_axis=False).T

    return source


def fake_emission_line_spectrum(size, n_lines, max_intensity=1, fwhm=2):
    """
    Generates a 1D array with the a fake emission-line spectrum using lines at
    random positions and with random intensities.

    Parameters
    ----------
    size : int
        Output array's size.
    n_lines : int
        Number of sky lines.
    max_intensity : float
        Maximum sky line intensity (default=1).
    fwhm : float
        Lines width in pixels (default=2).

    Returns
    -------
    np.ndarray
        Modeled emission-line spectrum
    """

    lines_positions = np.random.randint(low=0, high=size - 1, size=n_lines)
    lines_intensities = np.random.rand(n_lines) * max_intensity

    stddev = [fwhm / (2. * np.sqrt(2. * np.log(2.)))] * n_lines

    print(len(lines_positions), len(lines_intensities), len(stddev))

    model = models.Gaussian1D(
        amplitude=lines_intensities,
        mean=lines_positions,
        stddev=stddev,
        n_models=n_lines
    )

    source = model(np.arange(size), model_set_axis=False)
    source = source.sum(axis=0)

    return source


def get_aperture_table(height, width, center=None):
    """

    Parameters
    ----------
    height : int
        Output 2D array's number of rows.
    width : int
        Output 2D array's number of columns.
    center : None or int
        Center of the aperture. If None, defaults to the half of the height.

    Returns
    -------
    astropy.table.Table
        Aperture table containing the parameters defining the aperture

    """
    center = height // 2 if center is None else center
    apmodel = models.Chebyshev1D(degree=0, domain=[0, width-1], c0=center)
    aperture = am.model_to_table(apmodel)
    aperture['number'] = 1
    aperture['aper_lower'] = -3
    aperture['aper_upper'] = 3

    return aperture


class SkyLines:
    """
    Helper class to simulate random sky lines for tests. Use `np.random.seed()`
    to have the same lines between calls.

    Parameters
    ----------
    n_lines : int
        Number of lines to be included.
    max_position : int
        Maximum position value.
    max_value : float
        Maximum float value.

    """

    def __init__(self, n_lines, max_position, max_value=1.):
        self.positions = np.random.randint(low=0, high=max_position, size=n_lines)
        self.intensities = np.random.random(size=n_lines) * max_value

    def __call__(self, data, axis=0):
        """
        Generates a sky frame filled with zeros and with the random sky lines.

        Parameters
        ----------
        data : ndarray
            2D ndarray representing the detector.
        axis : {0, 1}
            Dispersion axis: 0 for rows or 1 for columns.

        Returns
        -------
        numpy.ndarray
            2D array matching input shape filled with zeros and the random sky
            lines.
        """
        sky_data = np.zeros_like(data)
        if axis == 0:
            sky_data[self.positions] = self.intensities
        elif axis == 1:
            sky_data[:, self.positions] = self.intensities
        else:
            raise ValueError(
                "Wrong value for dispersion axis. "
                "Expected 0 or 1, found {:d}".format(axis))

        return sky_data


if __name__ == '__main__':
    pytest.main()
