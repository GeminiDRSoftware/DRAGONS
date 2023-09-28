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

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from gempy.library import astromodels as am
from gempy.library.config.config import FieldValidationError
from geminidr.core import primitives_spect
from geminidr.f2.primitives_f2_longslit import F2Longslit
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.niri.primitives_niri_longslit import NIRILongslit

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


@pytest.mark.xfail(reason="The fake data needs a DQ plane")
def test_find_apertures():
    _p = primitives_spect.Spect([])
    _p.findApertures()


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

@pytest.mark.preprocessed_data
@pytest.mark.parametrize('in_shift', [0, -1.2, 2.75])
def test_adjust_wavelength_zero_point_shift(in_shift, change_working_dir,
                                            path_to_inputs):
    with change_working_dir(path_to_inputs):
        ad = astrodata.open('N20220706S0337_wavelengthSolutionAttached.fits')

    p = GNIRSLongslit([ad])
    ad_out = p.adjustWavelengthZeroPoint(shift=in_shift).pop()
    transform = ad_out[0].wcs.get_transform('pixels',
                                            'wavelength_scale_adjusted')
    shift = getattr(transform, 'offset_1')
    assert shift == pytest.approx(in_shift)

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

@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize('filename,instrument',
                         [('N20220706S0337', 'GNIRS'),
                          ('N20110331S0400', 'GNIRS'),
                          ('N20150511S0123', 'GNIRS'),
                          ('N20220718S0140', 'GNIRS'),
                          ('N20130827S0128', 'GNIRS'),
                          ('S20140610S0077', 'F2'),
                          ('S20210430S0138', 'F2'),
                          ('S20150629S0230', 'F2'),
                          ('S20210709S0035', 'F2'),
                          ('S20170215S0111', 'F2'),
                          ('S20180125S0028', 'F2'),
                          ('N20050918S0135', 'NIRI'),
                          ('N20050627S0040', 'NIRI'),
                          ('N20070615S0118', 'NIRI'),
                          ('N20061114S0193', 'NIRI'),
                          ])
def test_adjust_wavelength_zero_point_auto_shift(filename, instrument,
                                                 change_working_dir,
                                                 path_to_inputs):
    # Dictionary of shift values (in pixels) for each file.
    results = {'N20220706S0337': -0.0119375, # GNIRS 111/mm 0.10" LongBlue
               'N20110331S0400': 0.1669375,  # GNIRS 111/mm 0.30" ShortBlue
               'N20150511S0123': -0.0273750, # GNIRS 32/mm  0.45" ShortBlue
               'N20220718S0140': 2.083875,   # GNIRS 32/mm  0.10" LongBlue
               'N20130827S0128': -3.2903125, # GNIRS 10/mm  0.10" LongBlue
               'S20140610S0077': -1.755625,  # F2    R3K 1pix-slit f/16
               'S20210430S0138': 0.2556250,  # F2    R3K 2pix-slit f/16
               'S20150629S0230': 0.3927500,  # F2    JH  3pix-slit f/16
               'S20210709S0035': 0.3030625,  # F2    JH  4pix-slit f/16
               'S20170215S0111': 0.0551250,  # F2    HK  6pix-slit f/16
               'S20180125S0028': -0.046375,  # F2    JH  8pix-slit f/16
               'N20050918S0135': 0.6130625,  # NIRI  Hgrism f6-6pix
               'N20050627S0040': -0.059625,  # NIRI  Hgrism f6-6pix
               'N20070615S0118': -0.029875,  # NIRI  Jgrism f6-6pix
               'N20061114S0193': 0.1915000,  # NIRI  Kgrism f6-2pix
               }

    classes_dict = {'GNIRS': GNIRSLongslit,
                    'F2': F2Longslit,
                    'NIRI': NIRILongslit}

    # In some files the aperture lies (at least partly) in the cental column/row
    centers = {'N20130827S0128': 800,
               'N20220718S0140': 300,
               'S20140610S0077': 600}
    try:
        center = centers[filename]
    except KeyError:
        center = None

    with change_working_dir(path_to_inputs):
        ad = astrodata.open(filename + '_wavelengthSolutionAttached.fits')

    p = classes_dict[instrument]([ad])
    ad_out = p.adjustWavelengthZeroPoint(center=center, shift=None).pop()
    transform = ad_out[0].wcs.get_transform('pixels',
                                             'wavelength_scale_adjusted')
    param = 'offset_0' if instrument == 'NIRI' else 'offset_1'
    shift = getattr(transform, param)

    assert shift == pytest.approx(results[filename])


@pytest.mark.preprocessed_data
@pytest.mark.parametrize('filename,instrument',
                         [('N20121118S0375_stack.fits', 'GNIRS'),
                          ('N20180605S0138_stack.fits', 'GNIRS'),
                          ('S20040413S0268_stack.fits', 'GNIRS'),
                          ('N20110718S0129_stack.fits', 'GNIRS'),
                          ('S20140728S0282_stack.fits', 'F2'),
                          ('S20131015S0043_stack.fits', 'F2'),
                          ('S20140111S0155_stack.fits', 'F2'),
                          ('N20090925S0312_stack.fits', 'NIRI'),
                          ('N20081223S0263_stack.fits', 'NIRI'),
                          ])
def test_determine_slit_edges(filename, instrument, change_working_dir,
                              path_to_inputs):

    # Dictionary of SLITEDGE table results of the fits to the edges of the
    # illuminated region of the given flats.
    results_dict = {
        'N20121118S0375_stack.fits': {
            # GNIRS 111/mm LongBlue, off right edge of detector.
            'c0': (33.532680341140335, 1025.8427086736597),
            'c1': (-6.962491642935299, -6.962491642935299),
            'c2': (-0.008639441981396224, -0.008639441981396224),
            'c3': (0.006629301489564899, 0.006629301489564899)
            },
        'N20180605S0138_stack.fits': {
            # GNIRS 111/mm LongBlue, off left edge of detector
            'c0': (-16.666408005668814, 975.6436215963819),
            'c1': (-7.318016142494172, -7.318016142494172),
            'c2': (-0.029505932328147154, -0.029505932328147154),
            'c3': (0.03802744314590496, 0.03802744314590496)
            },
        'S20040413S0268_stack.fits': {
            # GNIRS 32/mm ShortRed, centered
            'c0': (175.1038780784617, 833.0919423028884),
            'c1': (-1.539657063828468, -1.1023324795483416),
            'c2': (-0.18673502127430647, 0.2131179499350503),
            'c3': (-0.014948550646791112, 0.006686383003339903)
            },
        'N20110718S0129_stack.fits': {
            # GNIRS 10/mm LongRed, one-off slit length shorter than normal.
            'c0': (3.566833182251458, 897.1397974632922),
            'c1': (-6.0544148638266035, -9.960620341672538),
            'c2': (0.8947922316257532, 0.025634028590147614),
            'c3': (0.5814317533641548, 0.012033436540264349)
            },
        'S20140728S0282_stack.fits': {
            # F2 1pix-slit, HK, off left edge of detector.
            'c0': (14.2347097150639, 1523.234709715064),
            'c1': (62.14113511752838, 62.14113511752838),
            'c2': (-1.9125412940944726, -1.9125412940944726),
            'c3': (-0.050606934418499595, -0.050606934418499595)
            },
        'S20131015S0043_stack.fits': {
            # F2 2pix-slit, JH.
            'c0': (34.536800027052735, 1504.7962402622616),
            'c1': (-2.152961200179425, 2.3511131672766807),
            'c2': (-1.601418225294633, -2.1174852354416442),
            'c3': (-1.5700522625671025, 0.09513029975888616),
            'c4': (-4.24411360006377, 0.2518257633168614)
            },
        'S20140111S0155_stack.fits': {
            # F2 2pix-slit, R3K. Efficiency drops to zero in middle.
            'c0': (43.70026068842856, 1507.7421230551906),
            'c1': (-6.235410307377804, -6.578431648509839),
            'c2': (6.929845453000161, -2.996437641711837),
            'c3': (0.23347486719205496, -0.04967233624948784)
            },
        'N20081223S0263_stack.fits': {
            # NIRI f/6 4pix "blue" slit
            'c0': (275.5204150559385, 715.3069316746302),
            'c1': (2.872550744587704, 2.712112393776839),
            'c2': (-0.10925863620137954, 0.7216836016094347),
            'c3': (0.00016773603353222372, -0.0023568404108366085)
            },
        'N20090925S0312_stack.fits': {
            # NIRI f/32 10pix slit, which is also the f/6 2pix slit
            'c0': (-1.7415119488127857, 1013.2584880511872),
            'c1': (1.3095852317414478, 1.3095852317414478),
            'c2': (-0.18732505002211716, -0.18732505002211716),
            'c3': (-0.19919994899021326, -0.19919994899021326)
        }
    }

    classes_dict = {'GNIRS': GNIRSLongslit,
                    'F2': F2Longslit,
                    'NIRI': NIRILongslit}

    with change_working_dir(path_to_inputs):

        ad = astrodata.open(filename)

    p = classes_dict[instrument]([ad])

    if filename == 'N20110718S0129_stack.fits':
        # Give edges explicitly since the slit is shorter than nominal.
        e1, e2 = [10], [906]
    else:
        e1, e2 = None, None
    if filename == 'S20131015S0043_stack.fits':
        # This file benefits from a 4th-order fit.
        order = 4
    else:
        order = 3

    ad_out = p.determineSlitEdges(edges1=e1, edges2=e2,
                                  spectral_order=order).pop()

    for i, row in enumerate(ad_out[0].SLITEDGE):
        m = am.table_to_model(row)
        m_ref = m.copy()
        for param in m.param_names:
            setattr(m_ref, param, results_dict[filename][param][i])
        x = np.arange(*m.domain)
        np.testing.assert_allclose(m(x), m_ref(x), atol=1.)


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
    assert ad_compare(ad_out, ref)


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


@pytest.mark.preprocessed_data
@pytest.mark.regression
def test_trace_pinhole_apertures(change_working_dir, path_to_inputs):

    c0_0_1_values = (89.69, 73.72, 74.15, 83.76, 99.95, 120.51)

    with change_working_dir(path_to_inputs):
        ad = astrodata.open("S20060507S0125_flatCorrected.fits")

    p = GNIRSCrossDispersed([ad])

    ad_out = p.tracePinholeApertures()[0]

    for ext, c_val in zip(ad_out, c0_0_1_values):
        model = ext.wcs.get_transform('pixels', 'rectified')
        assert model.name == 'PNHLRECT'
        assert len(model._parameters) == 8
        assert model.c0_0_1 == pytest.approx(c_val, abs=0.15)
        assert model.inputs == ('x0', 'x1')
        assert model.outputs == ('z', 'x0')


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
