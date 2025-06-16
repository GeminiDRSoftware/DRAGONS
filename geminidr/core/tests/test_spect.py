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
import logging

import numpy as np
import pytest
from astropy import table
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from scipy import optimize

from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS

from specutils.utils.wcs_utils import air_to_vac

import astrodata, gemini_instruments
from astrodata.testing import ad_compare, assert_most_close
from gempy.library import astromodels as am
from gempy.library.config.config import FieldValidationError
from geminidr.core import primitives_spect
from geminidr.f2.primitives_f2_longslit import F2Longslit
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.niri.primitives_niri_longslit import NIRILongslit
from geminidr.gnirs import primitives_gnirs_longslit
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from recipe_system.mappers.primitiveMapper import PrimitiveMapper

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

    ad_out = _p.extractSpectra([ad])[0]

    # AD only has an imaging WCS so cannot get RA, DEC of extraction region
    assert 'XTRACTRA' not in ad_out[0].hdr
    assert ad_out[0].hdr['XTRACTED'] == height // 2
    assert ad_out[0].hdr['XTRACTHI'] - ad_out[0].hdr['XTRACTLO'] == 6

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


@pytest.mark.preprocessed_data
@pytest.mark.parametrize('filename',
                         [('N20220706S0306_stack.fits'), # GNIRS
                          ('N20170601S0291_stack.fits'), # GNIRS
                          ('N20180201S0052_stack.fits'), # GNIRS
                          ('N20070204S0098_stack.fits'), # NIRI
                          ('N20061114S0193_stack.fits'), # NIRI
                          ('S20210430S0138_stack.fits'), # F2
                          ('S20210709S0035_stack.fits'), # F2
                          ('N20240102S0010_stack.fits'), # GMOS
                          ('N20190501S0054_stack.fits')]) # GMOS
def test_find_apertures(filename, path_to_inputs, change_working_dir):
    # Set the defaults at the time of test creation
    params = {'max_apertures': None, 'percentile': 80, 'section': "",
              'min_sky_region': 50, 'min_snr': 5., 'use_snr': True,
              'threshold': 0.10, 'max_separation': None}

    apertures = {'N20220706S0306_stack.fits': [(544.8, 17.6)],
                 'N20170601S0291_stack.fits': [(280.6, 7.2), (716.9, 6.4)],
                 'N20070204S0098_stack.fits': [(423.7, 19)],
                 'N20061114S0193_stack.fits': [(420.0, 12)],
                 'S20210430S0138_stack.fits': [(1064.6, 30), (105.4, 6.4),
                                               (306.5, 5.8)],
                 'S20210709S0035_stack.fits': [(959.8, 5.8), (378.4, 34.7)],
                 'N20240102S0010_stack.fits': [(495.2, 19.5), (1072.1, 36.0)],
                 'N20190501S0054_stack.fits': [(1066.6, 7.3)],
                 'N20180201S0052_stack.fits': [(501.8, 15.0)]}

    extra_params = {'S20210430S0138_stack.fits': {'threshold': 0.15, 'min_snr': 10},
                    'N20070204S0098_stack.fits': {'min_snr': 6},
                    'N20061114S0193_stack.fits': {'min_snr': 6},
                    'S20210709S0035_stack.fits': {'min_snr': 10}}

    with change_working_dir(path_to_inputs):
        ad = astrodata.open(filename)

    try: # Check for custom parameter values for individual tests
        params.update(extra_params[filename])
    except KeyError:
        pass

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])

    ad_out = p.findApertures(**params)[0]
    # Check we haven't created lots of spurious apertures
    assert len(ad_out[0].APERTURE) <= len(apertures[filename]) + 2

    # Check that the required apertures exist in the correct locations and
    # with the correct sizes (large tolerance)
    for loc, width in apertures[filename]:
        apnum = np.argmin(abs(ad_out[0].APERTURE['c0'] - loc))
        aperture = ad_out[0].APERTURE[apnum]
        assert loc == pytest.approx(aperture['c0'], abs=0.5)
        assert width == pytest.approx(aperture['aper_upper'] -
                                      aperture['aper_lower'],
                                      abs=1, rel=0.5)


@pytest.mark.preprocessed_data
def test_create_new_aperture(path_to_inputs):
    ad = astrodata.open(os.path.join(path_to_inputs, 'S20060826S0305_2D.fits'))
    p = GNIRSLongslit([ad])

    # Test creating a new aperture
    p.createNewAperture(aperture=1, shift=100)
    assert ad[0].APERTURE[1]['c0'] == pytest.approx(471.745)
    assert ad[0].APERTURE[1]['aper_lower'] == pytest.approx(-21.13415)
    assert ad[0].APERTURE[1]['aper_upper'] == pytest.approx(23.07667)

    # Create another aperature and test aper_lower & aper_upper parameters
    p.createNewAperture(aperture=1, shift=-100, aper_lower=-10, aper_upper=10)
    assert ad[0].APERTURE[2]['c0'] == pytest.approx(271.745)
    assert ad[0].APERTURE[2]['aper_lower'] == pytest.approx(-10)
    assert ad[0].APERTURE[2]['aper_upper'] == pytest.approx(10)

    # Delete aperture in the midde, test that aperture number increments
    del ad[0].APERTURE[1]
    p.createNewAperture(aperture=1, shift=100)
    assert ad[0].APERTURE[2]['c0'] == pytest.approx(471.745)


@pytest.mark.preprocessed_data
def test_create_new_aperture_warnings_and_errors(path_to_inputs, caplog):
    ad = astrodata.open(os.path.join(path_to_inputs, 'S20060826S0305_2D.fits'))
    p = GNIRSLongslit([ad])

    # Check that only passing one 'aper' parameter raises a ValueError
    with pytest.raises(ValueError):
        p.createNewAperture(aperture=1, shift=100, aper_lower=10, aper_upper=None)
        p.createNewAperture(aperture=1, shift=100, aper_lower=None, aper_upper=10)

    # Check that aper_upper & aper_lower limits are respected
    with pytest.raises(FieldValidationError):
        p.createNewAperture(aperture=1, shift=10, aper_lower=-2, aper_upper=-1)
    with pytest.raises(FieldValidationError):
        p.createNewAperture(aperture=1, shift=10, aper_lower=1, aper_upper=2)
    with pytest.raises(FieldValidationError):
        p.createNewAperture(aperture=1, shift=100, aper_lower=5, aper_upper=10)
    with pytest.raises(FieldValidationError):
        p.createNewAperture(aperture=1, shift=100, aper_lower=-10, aper_upper=-5)

    # Check that appropriate warnings are generated when creating apertures
    # with either the 'center' or an edge off the end of the array. Do them in
    # this order since the third and fourth also generate the warnings of the
    # first two.
    p.createNewAperture(aperture=1, shift=600, aper_lower=-5, aper_upper=400)
    assert any('New aperture is partially off right of image.' in record.message
                for record in caplog.records)
    p.createNewAperture(aperture=1, shift=-300, aper_lower=-500, aper_upper=5)
    assert any('New aperture is partially off left of image.' in record.message
                for record in caplog.records)
    p.createNewAperture(aperture=1, shift=1000)
    assert any('New aperture is entirely off right of image.' in record.message
                for record in caplog.records)
    p.createNewAperture(aperture=1, shift=-1000)
    assert any('New aperture is entirely off left of image.' in record.message
                for record in caplog.records)


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
        _table.write(_table.name, format='ascii', overwrite=True)

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


def test_sky_correct_from_slit_negative_beams(caplog):
    # Input Parameters ----------------
    width = 200
    height = 100
    np.random.seed(0)

    ad = create_zero_filled_fake_astrodata(height, width)
    # We need noise or get_extrema() doesn't work
    ad[0].data += np.random.randn(*ad[0].data.shape)

    for i in range(1, 4):
        source_model_parameters = {'c0': 0.25 * i * height, 'c1': 0.0}
        source = 100 * fake_point_source_spatial_profile(
            height, width, source_model_parameters, fwhm=0.05 * height)
        if i == 2:
            ad[0].data += source
        else:
            ad[0].data -= 0.5 * source

    ad[0].APERTURE = get_aperture_table(height, width)

    # Running the test ----------------
    caplog.set_level(logging.DEBUG)
    p = primitives_spect.Spect([])
    ad.phu[p.timestamp_keys['subtractSky']] = '2023-10-01T00:00:00.000'
    ad_out = p.skyCorrectFromSlit([ad], function="chebyshev", order=2,
                                  grow=2, niter=3, lsigma=3, hsigma=3,
                                  aperture_growth=2)[0]

    passing = False
    for record in caplog.records:
        if 'beam offsets' in record.message:
            fields = record.message.strip().split()
            assert len(fields) == 5, "Not 2 beam offsets found"
            passing = np.allclose(sorted([float(x) for x in fields[-2:]]),
                                  (-0.25 * height, 0.25 * height), atol=0.1)
    assert passing

@pytest.mark.preprocessed_data
@pytest.mark.parametrize('in_shift', [0, -1.2, 2.75])
def test_adjust_wavelength_zero_point_shift(in_shift, change_working_dir,
                                            path_to_inputs):
    """Apply a shift and confirm that the WCS has changed correctly"""
    with change_working_dir(path_to_inputs):
        ad = astrodata.open('N20220706S0337_wavelengthSolutionAttached.fits')

    dispaxis = 2 - ad.dispersion_axis()[0]  # python sense
    center = ad[0].shape[1 - dispaxis] // 2
    pixels = np.arange(ad[0].shape[dispaxis])
    waves = ad[0].wcs([center] * pixels.size, pixels)[0]

    p = GNIRSLongslit([ad])
    ad_out = p.adjustWavelengthZeroPoint(shift=in_shift).pop()
    new_waves = ad_out[0].wcs([center] * pixels.size, pixels - in_shift)[0]
    np.testing.assert_allclose(waves, new_waves)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize('in_shift', [-16, 7.7])
def test_adjust_wavelength_zero_point_overlarge_shift(in_shift,
                                                      change_working_dir,
                                                      path_to_inputs):
    with change_working_dir(path_to_inputs):
        ad = astrodata.open('N20220706S0337_wavelengthSolutionAttached.fits')

    p = GNIRSLongslit([ad])
    with pytest.raises(ValueError):
        p.adjustWavelengthZeroPoint(shift=in_shift).pop()


@pytest.mark.skip("Results derived from incorrect code")
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize('filename,result',
                         [('N20220706S0337', 0.0005),
                          ('N20110331S0400', 0.1696875),
                          ('N20150511S0123', -0.0273750),
                          ('N20220718S0140', 2.083875),
                          ('N20130827S0128', -3.2903125),
                          #('S20140610S0077', -1.755625),
                          #('S20210430S0138', 0.2556250),
                          #('S20150629S0230', 0.3927500),
                          #('S20210709S0035', 0.3030625),
                          #('S20170215S0111', 0.0551250),
                          #('S20180125S0028', -0.046375),
                          ('N20050918S0135', 0.6130625),
                          ('N20050627S0040', -0.059625),
                          ('N20070615S0118', -0.029875),
                          ('N20061114S0193', 0.1915000),
                          ])
def test_adjust_wavelength_zero_point_auto_shift(filename, result,
                                                 change_working_dir,
                                                 path_to_inputs):
    classes_dict = {'GNIRS': GNIRSLongslit,
                    'F2': F2Longslit,
                    'NIRI': NIRILongslit}

    # In some files the aperture lies (at least partly) in the cental column/row
    centers = {'N20130827S0128': 800,
               'N20220718S0140': 300,
               'S20140610S0077': 600}
    center = centers.get(filename)

    with change_working_dir(path_to_inputs):
        ad = astrodata.open(filename + '_wavelengthSolutionAttached.fits')

    instrument = ad.instrument()
    p = classes_dict[instrument]([ad])
    ad_out = p.adjustWavelengthZeroPoint(center=center, shift=None).pop()
    transform = ad_out[0].wcs.get_transform('pixels',
                                             'wavelength_scale_adjusted')
    param = 'offset_0' if instrument == 'NIRI' else 'offset_1'
    shift = getattr(transform, param)

    assert shift == pytest.approx(result, abs=0.001)


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize('filename,center,shift',
                         [('N20220706S0337', 300, 2),
                         ])
def test_adjust_wavelength_zero_point_controlled(filename, center, shift,
                                                 path_to_inputs):
    """
    Artificially shift a spectrogram along the dispersion axis and check
    that the shift is recovered by adjustWavelengthZeroPoint(). This assumes
    that the wavelength solution is correct and there is no significant
    flexure to within the tolerance (0.1 pixels).
    """
    classes_dict = {'GNIRS': GNIRSLongslit,
                    'F2': F2Longslit,
                    'NIRI': NIRILongslit}

    ad = astrodata.open(os.path.join(path_to_inputs,
                                     filename + '_wavelengthSolutionAttached.fits'))
    p = classes_dict[ad.instrument()]([ad])

    dispaxis = 2 - ad.dispersion_axis()[0]  # python sense
    pixels = np.arange(ad[0].shape[dispaxis])
    waves = ad[0].wcs([center] * pixels.size, pixels)[0]
    dw = abs(np.diff(waves)).mean()

    dispaxis = 2 - ad.dispersion_axis()[0]  # python sense
    ad[0].data = np.roll(ad[0].data, -shift, axis=dispaxis)
    ad_out = p.adjustWavelengthZeroPoint(center=center, shift=None).pop()

    new_waves = ad_out[0].wcs([center] * pixels.size, pixels - shift)[0]
    np.testing.assert_allclose(waves, new_waves, atol=0.1*dw)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize('in_file,instrument',
                         [# GNIRS 111/mm LongBlue, off right edge of detector
                          ('N20121118S0375', 'GNIRS'),
                          # GNIRS 10/mm LongRed, one-off slit length shorter
                          ('N20110718S0129', 'GNIRS'),
                          # F2 1pix-slit, HK, off left edge of detector
                          ('S20140728S0282', 'F2'),
                          # F2 2pix-slit, R3K. Efficiency drops to zero in middle
                          ('S20140111S0155', 'F2'),
                          # NIRI f/6 4pix "blue" slit
                          ('N20081223S0263', 'NIRI'),
                          # NIRI f/32 10pix slit, which is also the f/6 2pix slit
                          ('N20090925S0312', 'NIRI'),
                          ])
def test_mask_beyond_slit(in_file, instrument, change_working_dir,
                          path_to_inputs, path_to_refs):

    classes_dict = {'GNIRS': GNIRSLongslit,
                    'F2': F2Longslit,
                    'NIRI': NIRILongslit}

    ad = astrodata.open(os.path.join(path_to_inputs,
                                     in_file + '_slitEdgesDetermined.fits'))
    p = classes_dict[instrument]([ad])
    ad_out = p.maskBeyondSlit().pop()
    ref = astrodata.open(os.path.join(path_to_refs,
                                      in_file + '_maskedBeyondSlit.fits'))
    # Find the size of the smallest extension in the file; we don't need the
    # mask to match *exactly*, so as long as the mismatch isn't more than 0.1 of
    # the smallest extension it should be fine.
    size = 0
    for ext in ref:
        if ext.data.size < size or size == 0:
            size = ext.data.size

    assert ad_compare(ad_out, ref, compare=['attributes'], max_miss=size*0.001)


@pytest.mark.skip("Needs redoing/moving, think about how best to test slit rectification")
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('filename,instrument',
                         [# GNIRS, 111/mm LongBlue
                          ('N20121118S0375_distortionCorrected.fits', 'GNIRS'),
                          # GNIRS 32/mm ShortRed
                          ('N20100915S0162_distortionCorrected.fits', 'GNIRS'),
                          # F2 6 pix slit, JH
                          ('S20131019S0050_distortionCorrected.fits', 'F2'),
                          # F2 2 pix slit, HK
                          ('S20131127S0229_distortionCorrected.fits', 'F2'),
                          # NIRI 6 pix slit, f/6
                          ('N20100614S0569_distortionCorrected.fits', 'NIRI'),
                          # NIRI 2 pix slit, f/6 (stay light streaks)
                          ('N20100619S0602_distortionCorrected.fits', 'NIRI'),
                          ])
def test_slit_rectification(filename, instrument, change_working_dir,
                              path_to_inputs):

    classes_dict = {'GNIRS': GNIRSLongslit,
                    'F2': F2Longslit,
                    'NIRI': NIRILongslit}

    with change_working_dir(path_to_inputs):
        ad = astrodata.open(filename)

    p = classes_dict[instrument]([ad])

    ad_out = p.determineSlitEdges().pop()

    for coeff in ('c1', 'c2', 'c3'):
        np.testing.assert_allclose(ad_out[0].SLITEDGE[coeff], 0, atol=0.25)


@pytest.mark.parametrize("gnirs1d", [np.ones((1000,), dtype=np.float32),
                                     np.arange(1000, dtype=np.float32)],
                         indirect=True)
@pytest.mark.parametrize("wavescale", ["linear", "loglinear"])
def test_resample1d_conserve(gnirs1d, wavescale):
    """
    Simple test to resample a synthetic 1D spectrum with a linear wavelength
    solution to linear and loglinear and check that the flux is conserved.
    """
    gnirs1d[0].hdr['BUNIT'] = 'electron'
    p = GNIRSLongslit([gnirs1d])
    sum_before = gnirs1d[0].data.sum()
    adout = p.resampleToCommonFrame(output_wave_scale=wavescale).pop()
    sum_after = adout[0].data.sum()
    # Tolerance allows for edge effects
    assert sum_after == pytest.approx(sum_before, rel=0.002)


@pytest.mark.parametrize("gnirs1d", [np.arange(1000, 2000, dtype=np.float32)],
                         indirect=True)
@pytest.mark.parametrize("wavescale", ["linear", "loglinear"])
def test_resample1d_interpolate(gnirs1d, wavescale):
    """
    Simple test to resample a synthetic 1D spectrum with a linear wavelength
    solution to linear and loglinear and check that the data are interpolated.
    The input spectrum has signal=wavelength so it's easy to check the result.
    """
    gnirs1d[0].hdr['BUNIT'] = 'W / (m2 AA)'  # will interpolate
    p = GNIRSLongslit([gnirs1d])
    adout = p.resampleToCommonFrame(output_wave_scale=wavescale).pop()
    waves = adout[0].wcs(np.arange(adout[0].data.size))
    assert_most_close(waves, adout[0].data, max_miss=2, rtol=1e-5)


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
    warning_given = any(record.levelname == 'WARNING' for record in caplog.records)
    assert warn == warning_given


@pytest.mark.preprocessed_data
@pytest.mark.regression
def test_get_sky_spectrum(path_to_inputs, path_to_refs):
    # Spectrum of F2 OH-emission sky lines for plotting
    # We use the _wavelengthSolutionDetermined file because thw WAVE model
    # is a Chebyshev1D, as required. (In normal reduction, a Cheb1D will be
    # provided bto _get_sky_spectrum() y determineWavelengthSolution,
    # regardless of the state of the input file.)
    ad_f2 = astrodata.open(os.path.join(
        path_to_inputs, 'S20180114S0104_wavelengthSolutionDetermined.fits'))
    wave_model = am.get_named_submodel(ad_f2[0].wcs.forward_transform, 'WAVE')

    p = F2Longslit([])
    refplot_data_f2 = p._get_sky_spectrum(wave_model=wave_model, ext=ad_f2[0])
    ref_refplot_spec_f2 = np.loadtxt(
        os.path.join(path_to_refs, "S20180114S0104_refplot_spec.dat"))

    np.testing.assert_allclose(ref_refplot_spec_f2, refplot_data_f2["refplot_spec"], atol=1e-3)


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


def compare_frames(frame1, frame2):
    """Compare the important stuff of two CoordinateFrame instances"""
    for attr in ("naxes", "axes_type", "axes_order", "unit", "axes_names"):
        assert getattr(frame1, attr) == getattr(frame2, attr)


#@pytest.mark.skip
@pytest.mark.preprocessed_data
@pytest.mark.regression
def test_transfer_distortion_model(change_working_dir, path_to_inputs, path_to_refs):
    """
    The input files are created using makeWavecalFromSkyAbsorption recipe,
    run up until p.transferDistortionModel():
        p.prepare()
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=True)
        p.flatCorrect()
        p.attachWavelengthSolution()
        p.copyInputs(instream="main", outstream="with_distortion_model")
        p.separateSky()
        p.associateSky()
        p.skyCorrect()
        p.cleanReadout()
        p.distortionCorrect()
        p.adjustWCSToReference()
        p.resampleToCommonFrame(force_linear=False)
        p.stackFrames()
        p.findApertures()
        p.determineWavelengthSolution(absorption=True)
    """
    ad_no_dist_model = astrodata.open(os.path.join(path_to_inputs, 'N20121221S0199_wavelengthSolutionDetermined.fits'))
    ad_with_dist_model = astrodata.open(os.path.join(path_to_inputs, 'N20121221S0199_wavelengthSolutionAttached.fits'))
    p = primitives_gnirs_longslit.GNIRSLongslit([ad_no_dist_model])
    p.streams["with_distortion_model"] = ad_with_dist_model
    ad_with_dist_model_transferred = p.transferDistortionModel(source="with_distortion_model")
    p.writeOutputs()
    with change_working_dir(path_to_refs):
        ref_with_dist_model_transferred = \
        astrodata.open(os.path.join(path_to_refs, "N20121221S0199_distortionModelTransferred.fits"))

    # Compare output WCS as well as pixel values (by evaluating it at the
    # ends of the ranges, since there are multiple ways of constructing an
    # equivalent WCS, eg. depending on the order of various Shift models):
    ad_wcs = ad_with_dist_model_transferred[0][0].wcs
    ref_wcs = ref_with_dist_model_transferred[0].wcs
    for f in ref_wcs.available_frames:
        compare_frames(getattr(ref_wcs, f), getattr(ad_wcs, f))
    corner1, corner2 = (0, 0), tuple(v-1 for v in ref_with_dist_model_transferred[0].shape[::-1])
    world1, world2 = ref_wcs(*corner1), ref_wcs(*corner2)
    np.testing.assert_allclose(ad_wcs(*corner1), world1, rtol=1e-6)
    np.testing.assert_allclose(ad_wcs(*corner2), world2, rtol=1e-6)
    # The inverse is not highly accurate and transforming back & forth can
    # just exceed even a tolerance of 0.01 pix, but it's best to compare
    # with the true corner values rather than the same results from the
    # reference, otherwise it would be too easy to overlook a problem with
    # the inverse when someone checks the reference file.
    np.testing.assert_allclose(ad_wcs.invert(*world1), corner1, atol=0.02)
    np.testing.assert_allclose(ad_wcs.invert(*world2), corner2, atol=0.02)


# --- Fixtures and helper functions -------------------------------------------

@pytest.fixture
def gnirs1d(request, astrofaker):
    ad = astrofaker.create('GNIRS', mode="SPECT")
    ad.init_default_extensions()
    data = request.param
    mask = np.zeros_like(data, dtype=np.uint16)
    variance = np.ones_like(data, dtype=np.float32)
    ad[0].reset(data=data, mask=mask, variance=variance)
    wave_model = models.Shift(1000) | models.Scale(1)
    input_frame = astrodata.wcs.pixel_frame(1)
    output_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                    axes_names=("WAVE",))
    ad[0].wcs = gWCS([(input_frame, wave_model),
                      (output_frame, None)])
    return ad


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
