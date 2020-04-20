#!/usr/bin/env python
"""
Tests for the `calculateSensitivity` primitive using GMOS-S and GMOS-N data.
"""
import numpy as np
import os
import pytest

import astrodata
import gemini_instruments

from astrodata import testing
from astropy.io import fits
from scipy.interpolate import BSpline

from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.utils import logutils


datasets = [
    "N20180109S0287.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
]


# --- Tests -------------------------------------------------------------------
@pytest.mark.gmosls
def test_flux_calibration_with_fake_data():

    def _get_spectrophotometric_file_path(specphot_name):

        from geminidr.gemini.lookups import spectrophotometric_standards

        path = list(spectrophotometric_standards.__path__).pop()
        file_path = os.path.join(path, specphot_name.lower().replace(' ', '') + ".dat")

        return file_path

    def _get_spectrophotometric_data(object_name):

        file_path = _get_spectrophotometric_file_path(object_name)
        table = primitives_gmos_spect.Spect([])._get_spectrophotometry(file_path)

        std_wavelength = table['WAVELENGTH'].data
        std_flux = table['FLUX'].data

        if std_wavelength[0] // 1000 > 1:  # Converts from \AA to nm
            std_wavelength = std_wavelength / 10

        std_flux = std_flux[(350 <= std_wavelength) * (std_wavelength <= 850)]
        std_wavelength = std_wavelength[(350 <= std_wavelength) & (std_wavelength <= 850)]

        spline = BSpline(std_wavelength, std_flux, 3)
        wavelength = np.linspace(std_wavelength.min(), std_wavelength.max(), 1000)
        flux = spline(wavelength)
        return wavelength, flux

    def _create_fake_data(object_name):
        from astropy.table import Table
        astrofaker = pytest.importorskip('astrofaker')

        wavelength, flux = _get_spectrophotometric_data(object_name)

        wavecal = {
            'ndim': 1.,
            'degree': 1.,
            'domain_start': 0.,
            'domain_end': wavelength.size - 1,
            'c0': wavelength.mean(),
            'c1': wavelength.mean() / 2,
        }

        hdu = fits.ImageHDU()
        hdu.header['CCDSUM'] = "1 1"
        hdu.data = flux[np.newaxis, :]  # astrofaker needs 2D data

        _ad = astrofaker.create('GMOS-S')
        _ad.add_extension(hdu, pixel_scale=1.0)

        _ad[0].data = _ad[0].data.ravel()
        _ad[0].mask = np.zeros(_ad[0].data.size, dtype=np.uint16)  # ToDo Requires mask
        _ad[0].variance = np.ones_like(_ad[0].data)  # ToDo Requires Variance
        _ad[0].WAVECAL = Table(
            [list(wavecal.keys()), list(wavecal.values())],
            names=("name", "coefficients"),
            dtype=(str, float))

        _ad[0].hdr.set('NAXIS', 1)
        _ad[0].phu.set('OBJECT', object_name)
        _ad[0].phu.set('EXPTIME', 1.)
        _ad[0].hdr.set('BUNIT', "electron")
        _ad[0].hdr.set('CTYPE1', "Wavelength")
        _ad[0].hdr.set('CUNIT1', "nm")
        _ad[0].hdr.set('CRPIX1', 0)
        _ad[0].hdr.set('CRVAL1', wavelength.min())
        _ad[0].hdr.set('CDELT1', wavelength.ptp() / wavelength.size)
        _ad[0].hdr.set('CD1_1', wavelength.ptp() / wavelength.size)

        assert _ad.object() == object_name
        assert _ad.exposure_time() == 1

        return _ad

    ad = _create_fake_data("Feige 34")

    p = primitives_gmos_spect.GMOSSpect([ad])
    std_ad = p.calculateSensitivity()[0]
    flux_corrected_ad = p.fluxCalibrate(standard=std_ad)[0]

    for ext, flux_corrected_ext in zip(ad, flux_corrected_ad):
        np.testing.assert_allclose(flux_corrected_ext.data, ext.data, atol=1e-4)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_regression_on_flux_calibration(flux_cal_ad, reference_ad):

    ref_ad = reference_ad(flux_cal_ad.filename)

    for calc_sens_ext, ref_ext in zip(flux_cal_ad, ref_ad):
        np.testing.assert_allclose(
            calc_sens_ext.data, ref_ext.data, atol=1e-4)


# --- Helper functions and fixtures -------------------------------------------
@pytest.fixture(scope='module', params=datasets)
def flux_cal_ad(request, get_input_ad, output_path):
    """
    Parameters
    ----------
    request
    get_input_ad
    output_path

    Returns
    -------
    """
    filename = request.param
    pre_process = request.config.getoption("--force-preprocess-data")

    input_ad = get_input_ad(filename, pre_process)

    with output_path():
        p = primitives_gmos_spect.GMOSSpect([input_ad])
        p.fluxCalibrate(standard=input_ad)
        flux_calibrated_ad = p.writeOutputs().pop()

    return flux_calibrated_ad


@pytest.fixture(scope='module')
def get_input_ad(cache_path, new_path_to_inputs, reduce_arc, reduce_bias,
                 reduce_data,  reduce_flat):
    """
    Reads the input data or cache/process it in a temporary folder.

    Parameters
    ----------
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_arc : pytest.fixture
        Recipe to reduce the arc file.
    reduce_bias : pytest.fixture
        Recipe to reduce the bias files.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before `applyQECorrect`.
    reduce_flat : pytest.fixture
        Recipe to reduce the flat file.

    Returns
    -------
    flat_corrected_ad : AstroData
        Bias and flat corrected data.
    master_arc : AstroData
        Master arc data.
    """
    def _get_input_ad(basename, should_preprocess):

        input_fname = basename.replace('.fits', '_sensitivityCalculated.fits')
        input_path = os.path.join(new_path_to_inputs, input_fname)
        cals = testing.get_associated_calibrations(basename)

        if should_preprocess:

            filename = cache_path(basename)
            ad = astrodata.open(filename)

            cals = [cache_path(c) for c in cals.filename.values]

            master_bias = reduce_bias(
                ad.data_label(),
                dataselect.select_data(cals, tags=['BIAS']))

            master_flat = reduce_flat(
                ad.data_label(),
                dataselect.select_data(cals, tags=['FLAT']), master_bias)

            master_arc = reduce_arc(
                ad.data_label(),
                dataselect.select_data(cals, tags=['ARC']))

            input_data = reduce_data(
                ad, master_arc, master_bias, master_flat)

        elif os.path.exists(input_path):
            input_data = astrodata.open(input_path)

        else:
            raise IOError(
                'Could not find input file:\n' +
                '  {:s}\n'.format(input_path) +
                '  Run pytest with "--force-preprocess-data" to get it')

        return input_data

    return _get_input_ad


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Factory for function for FLAT data reduction.

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
            p.calculateSensitivity()

            processed_ad = p.writeOutputs().pop()

        return processed_ad
    return _reduce_data
