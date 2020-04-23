#!/usr/bin/env python
"""
Tests for the `calculateSensitivity` primitive using GMOS-S and GMOS-N data.
"""
import numpy as np
import os
import pytest

import astrodata
import gemini_instruments

from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from scipy.interpolate import BSpline

from astrodata import testing
from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.utils import logutils


datasets = [
    "N20180109S0287.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
]


# --- Tests -------------------------------------------------------------------
@pytest.mark.gmosls
def test_calculate_sensitivity_from_science_equals_one_and_table_equals_one(path_to_outputs):

    def _create_fake_data():
        astrofaker = pytest.importorskip('astrofaker')

        hdu = fits.ImageHDU()
        hdu.header['CCDSUM'] = "1 1"
        hdu.data = np.zeros((1000, 1))

        _ad = astrofaker.create('GMOS-S')
        _ad.add_extension(hdu, pixel_scale=1.0)

        _ad[0].data = _ad[0].data.ravel() + 1.
        _ad[0].mask = np.zeros(_ad[0].data.size, dtype=np.uint16)  # ToDo Requires mask
        _ad[0].variance = np.ones_like(_ad[0].data)  # ToDo Requires Variance

        _ad[0].phu.set('OBJECT', "DUMMY")
        _ad[0].phu.set('EXPTIME', 1.)
        _ad[0].hdr.set('BUNIT', "electron")
        _ad[0].hdr.set('CTYPE1', "Wavelength")
        _ad[0].hdr.set('CUNIT1', "nm")
        _ad[0].hdr.set('CRPIX1', 1)
        _ad[0].hdr.set('CRVAL1', 350.)
        _ad[0].hdr.set('CDELT1', 0.1)
        _ad[0].hdr.set('CD1_1', 0.1)

        assert _ad.object() == 'DUMMY'
        assert _ad.exposure_time() == 1

        return _ad

    def _create_fake_table():

        wavelengths = np.arange(200., 900., 5) * u.Unit("nm")
        flux = np.ones(wavelengths.size) * u.Unit("erg cm-2 s-1 AA-1")
        bandpass = np.ones(wavelengths.size) * u.Unit("nm")

        _table = QTable([wavelengths, flux, bandpass],
                        names=['WAVELENGTH', 'FLUX', 'FWHM'])

        _table.name = os.path.join(path_to_outputs, 'std_data.dat')
        _table.write(_table.name, format='ascii.ecsv')

        return _table.name

    def _get_wavelength_calibration(hdr):

        from astropy.modeling.models import Linear1D, Const1D

        _wcal_model = (
            Const1D(hdr.get('CRVAL1')) +
            Linear1D(slope=hdr.get('CD1_1'), intercept=hdr.get('CRPIX1')-1))

        assert _wcal_model(0) == hdr.get('CRVAL1')

        return _wcal_model

    table_name = _create_fake_table()
    ad = _create_fake_data()

    p = primitives_gmos_spect.GMOSSpect([ad])
    s_ad = p.calculateSensitivity(filename=table_name).pop()

    assert hasattr(s_ad[0], 'SENSFUNC')

    for s_ext in s_ad:

        sens_table = s_ext.SENSFUNC
        sens_model = BSpline(
            sens_table['knots'].data, sens_table['coefficients'].data, 3)

        wcal_model = _get_wavelength_calibration(s_ext.hdr)

        wlengths = (wcal_model(np.arange(s_ext.data.size + 1)) * u.nm).to(
            sens_table['knots'].unit)

        sens_factor = sens_model(wlengths) * sens_table['coefficients'].unit

        np.testing.assert_allclose(sens_factor.physical.value, 1, atol=1e-4)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_regression_on_calculate_sensitivity(output_path, preprocessed_ad, reference_ad):

    with output_path():
        p = primitives_gmos_spect.GMOSSpect([preprocessed_ad])
        p.calculateSensitivity()
        calc_sens_ad = p.writeOutputs().pop()

    assert hasattr(calc_sens_ad[0], 'SENSFUNC')

    ref_ad = reference_ad(calc_sens_ad.filename)

    for calc_sens_ext, ref_ext in zip(calc_sens_ad, ref_ad):
        np.testing.assert_allclose(
            calc_sens_ext.data, ref_ext.data, atol=1e-4)


# --- Helper functions and fixtures -------------------------------------------
@pytest.fixture(scope='function', params=datasets)
def preprocessed_ad(request, cache_path, new_path_to_inputs, reduce_arc,
                    reduce_bias, reduce_data,  reduce_flat):
    """
    Reads the input data or cache/process it in a temporary folder.

    Parameters
    ----------
    request : pytest.fixture
        PyTest's built-in fixture with information about the test itself.
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_arc : pytest.fixture
        Recipe to reduce the arc file.
    reduce_bias : pytest.fixture
        Recipe to reduce the bias files.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before `calculateSensitivity`.
    reduce_flat : pytest.fixture
        Recipe to reduce the flat file.

    Returns
    -------
    AstroData
        Preprocessed data to be used as input for `calculateSensitivity`.
    """
    basename = request.param
    should_preprocess = request.config.getoption("--force-preprocess-data")

    input_fname = basename.replace('.fits', '_extracted.fits')
    input_path = os.path.join(new_path_to_inputs, input_fname)

    if os.path.exists(input_path):
        input_data = astrodata.open(input_path)

    elif should_preprocess:
        filename = cache_path(basename)
        ad = astrodata.open(filename)
        cals = testing.get_associated_calibrations(basename)
        cals = [cache_path(c) for c in cals.filename.values]

        master_bias = reduce_bias(
            ad.data_label(), dataselect.select_data(cals, tags=['BIAS']))

        master_flat = reduce_flat(
            ad.data_label(), dataselect.select_data(cals, tags=['FLAT']), master_bias)

        master_arc = reduce_arc(
            ad.data_label(), dataselect.select_data(cals, tags=['ARC']))

        input_data = reduce_data(ad, master_arc, master_bias, master_flat)

    else:
        raise IOError(
            'Could not find input file:\n' +
            '  {:s}\n'.format(input_path) +
            '  Run pytest with "--force-preprocess-data" to get it')

    return input_data


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Factory for function for data reduction prior to `calculateSensitivity`.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the standard star file, process them
    using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad, master_arc, master_bias, master_flat):
        with output_path():
            # Use config to prevent outputs when running Reduce via API
            logutils.config(file_name='log_{}.txt'.format(ad.data_label()))

            p = primitives_gmos_longslit.GMOSLongslit([ad])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.biasCorrect(bias=master_bias)
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.flatCorrect(flat=master_flat)
            p.applyQECorrection(arc=master_arc)
            p.distortionCorrect(arc=master_arc)
            p.findSourceApertures(max_apertures=1)
            p.skyCorrectFromSlit()
            p.traceApertures()
            p.extract1DSpectra()

            processed_ad = p.writeOutputs().pop()

        return processed_ad
    return _reduce_data
