#!/usr/bin/env python
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
from copy import copy

import numpy as np
import os
import pytest

import astrodata

from astropy import table
from astropy.io import fits
from astropy.modeling import models
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import optimize

from geminidr.core import primitives_spect

astrofaker = pytest.importorskip("astrofaker")


# -- Tests ---------------------------------------------------------------------
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
    ad_out = _p.extract1DSpectra([ad])[0]

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
    ad_out = _p.extract1DSpectra([ad])[0]

    np.testing.assert_equal(ad_out[0].shape[0], ad[0].shape[1])
    np.testing.assert_allclose(ad_out[0].data, source_intensity, atol=1e-3)


@pytest.mark.xfail(reason="The fake data needs a DQ plane")
def test_find_apertures():
    _p = primitives_spect.Spect([])
    _p.findSourceApertures()


def test_get_spectrophotometry(path_to_outputs):

    def create_fake_table():

        wavelengths = np.arange(350., 750., 10)
        flux = np.ones(wavelengths.size)
        bandpass = np.ones(wavelengths.size) * 5.

        _table = table.Table(
            [wavelengths, flux, bandpass],
            names=['WAVELENGTH', 'FLUX', 'FWHM'])

        _table.name = os.path.join(path_to_outputs, 'specphot.dat')
        _table.write(_table.name, format='ascii')

        return _table.name

    _p = primitives_spect.Spect([])
    fake_table = _p._get_spectrophotometry(create_fake_table())
    np.testing.assert_allclose(fake_table['FLUX'], 1)

    assert 'WAVELENGTH' in fake_table.columns
    assert 'FLUX' in fake_table.columns
    assert 'WIDTH' in fake_table.columns

    assert hasattr(fake_table['WAVELENGTH'], 'quantity')
    assert hasattr(fake_table['FLUX'], 'quantity')
    assert hasattr(fake_table['WIDTH'], 'quantity')


def test_split_mosaic_into_extensions(request):
    """
    Tests helper function that split a mosaicked data into several extensions
    based on another multi-extension file that contains gWCS.
    """
    # For now...
    from geminidr.gmos.lookups import geometry_conf as geotable
    from gempy.library import transform
    from gempy.gemini import gemini_tools as gt

    ad = astrofaker.create('GMOS-S')
    ad.init_default_extensions(binning=2)

    ad = transform.add_mosaic_wcs(ad, geotable)
    ad = gt.trim_to_data_section(
        ad, keyword_comments={'NAXIS1': "", 'NAXIS2': "", 'DATASEC': "",
                              'TRIMSEC': "", 'CRPIX1': "", 'CRPIX2': ""})

    for i, ext in enumerate(ad):
        x1 = ext.detector_section().x1
        x2 = ext.detector_section().x2
        xb = ext.detector_x_bin()

        data = np.arange(x1 // xb, x2 // xb)[np.newaxis, :]
        data = np.repeat(data, ext.data.shape[0], axis=0)
        data = data + 0.1 * (0.5 - np.random.random(data.shape))

        ext.data = data

    mosaic_ad = transform.resample_from_wcs(
        ad, "mosaic", attributes=None, order=1, process_objcat=False)

    mosaic_ad[0].data = np.pad(mosaic_ad[0].data, 10, mode='edge')

    mosaic_ad[0].hdr[mosaic_ad._keyword_for('data_section')] = \
        '[1:{},1:{}]'.format(*mosaic_ad[0].shape[::-1])

    ad2 = primitives_spect._split_mosaic_into_extensions(
        ad, mosaic_ad, border_size=10)

    if request.config.getoption("--do-plots"):

        palette = copy(plt.cm.viridis)
        palette.set_bad('r', 1)

        fig = plt.figure(num="Test: Split Mosaic Into Extensions", figsize=(8, 6.5), dpi=120)
        fig.suptitle("Test Split Mosaic Into Extensions\n Difference between"
                     " input and mosaicked/demosaicked data")

        gs = fig.add_gridspec(nrows=4, ncols=len(ad) // 3, wspace=0.1, height_ratios=[1, 1, 1, 0.1])

        for i, (ext, ext2) in enumerate(zip(ad, ad2)):

            data1 = ext.data
            data2 = ext2.data
            diff = np.ma.masked_array(data1 - data2, mask=np.abs(data1 - data2) > 1)
            height, width = data1.shape

            row = i // 4
            col = i % 4

            ax = fig.add_subplot(gs[row, col])
            ax.set_title("Ext {}".format(i + 1))
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            _ = [ax.spines[s].set_visible(False) for s in ax.spines]

            if col == 0:
                ax.set_ylabel("Det {}".format(row + 1))

            sub_gs = gridspec.GridSpecFromSubplotSpec(2, 2, ax, wspace=0.05, hspace=0.05)

            for j in range(4):
                sx = fig.add_subplot(sub_gs[j])
                im = sx.imshow(diff, origin='lower', cmap=palette, vmin=-0.1, vmax=0.1)

                sx.set_xticks([])
                sx.set_yticks([])
                sx.set_xticklabels([])
                sx.set_yticklabels([])
                _ = [sx.spines[s].set_visible(False) for s in sx.spines]

                if j == 0:
                    sx.set_xlim(0, 25)
                    sx.set_ylim(height - 25, height)

                if j == 1:
                    sx.set_xlim(width - 25, width)
                    sx.set_ylim(height - 25, height)

                if j == 2:
                    sx.set_xlim(0, 25)
                    sx.set_ylim(0, 25)

                if j == 3:
                    sx.set_xlim(width - 25, width)
                    sx.set_ylim(0, 25)

        cax = fig.add_subplot(gs[3, :])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Difference levels")

        output_dir = "./plots/geminidr/core/"
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(
            os.path.join(output_dir, "test_split_mosaic_into_extensions.png"))

    # Actual test ----
    for i, (ext, ext2) in enumerate(zip(ad, ad2)):
        data1 = np.ma.masked_array(ext.data[1:-1, 1:-1], mask=ext.mask)
        data2 = np.ma.masked_array(ext2.data[1:-1, 1:-1], mask=ext2.mask)

        np.testing.assert_almost_equal(data1, data2, decimal=1)


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
    order = 10

    result = optimize.minimize(
        QESpline, coeffs,
        args=(xpix, masked_data, weights, boundaries, order),
        tol=1e-7,
        method='Nelder-Mead'
    )

    np.testing.assert_allclose(real_coeffs, 1. / result.x, atol=0.01)


def test_sky_correct_from_slit():
    # Input Parameters ----------------
    width = 200
    height = 100

    n_sky_lines = 500

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
    _p = primitives_spect.Spect([])

    # ToDo @csimpson: Is it modifying the input ad?
    ad_out = _p.skyCorrectFromSlit([ad])[0]

    np.testing.assert_allclose(ad_out[0].data, source, atol=0.00625)


def test_sky_correct_from_slit_with_aperture_table():
    # Input Parameters ----------------
    width = 200
    height = 100

    # Simulate Data -------------------
    np.random.seed(0)

    source_model_parameters = {'c0': height // 2, 'c1': 0.0}

    source = fake_point_source_spatial_profile(
        height, width, source_model_parameters, fwhm=0.08 * height)

    sky = SkyLines(n_lines=width // 2, max_position=width - 1)

    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data += source
    ad[0].data += sky(ad[0].data, axis=1)
    ad[0].APERTURE = get_aperture_table(height, width)

    # Running the test ----------------
    _p = primitives_spect.Spect([])

    # ToDo @csimpson: Is it modifying the input ad?
    ad_out = _p.skyCorrectFromSlit([ad])[0]

    np.testing.assert_allclose(ad_out[0].data, source, atol=0.00625)


# noinspection PyPep8Naming


def test_sky_correct_from_slit_with_multiple_sources():
    width = 200
    height = 100
    np.random.seed(0)

    y0 = height // 2
    y1 = 7 * height // 16
    fwhm = 0.05 * height

    source = (
            fake_point_source_spatial_profile(height, width, {'c0': y0, 'c1': 0.0}, fwhm=fwhm) +
            fake_point_source_spatial_profile(height, width, {'c0': y1, 'c1': 0.0}, fwhm=fwhm)
    )

    sky = SkyLines(n_lines=width // 2, max_position=width - 1)

    ad = create_zero_filled_fake_astrodata(height, width)

    ad[0].data += source
    ad[0].data += sky(ad[0].data, axis=1)
    ad[0].APERTURE = get_aperture_table(height, width, center=height // 2)
    ad[0].APERTURE.add_row([1, 1, 0, 0, width - 1, y1, -3, 3])

    # Running the test ----------------
    _p = primitives_spect.Spect([])

    # ToDo @csimpson: Is it modifying the input ad?
    ad_out = _p.skyCorrectFromSlit([ad])[0]

    np.testing.assert_allclose(ad_out[0].data, source, atol=0.00625)


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
    ad_out = _p.traceApertures([ad], trace_order=len(trace_model_parameters) + 1)[0]

    keys = trace_model_parameters.keys()

    desired = np.array([trace_model_parameters[k] for k in keys])
    actual = np.array([ad_out[0].APERTURE[0][k] for k in keys])
    np.testing.assert_allclose(desired, actual, atol=0.05)


# --- Fixtures and helper functions --------------------------------------------
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
        Aperture table containing the parameters needed to build a Chebyshev1D
        model (number, ndim, degree, domain_start, domain_end, aper_lower,
        aper_uper, c0, c1, c...)

    """
    center = height // 2 if center is None else center

    aperture = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [width - 1],  # domain_end
         [center],  # c0
         [-3],  # aper_lower
         [3],  # aper_upper
         ],
        names=[
            'number',
            'ndim',
            'degree',
            'domain_start',
            'domain_end',
            'c0',
            'aper_lower',
            'aper_upper'],
    )

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
