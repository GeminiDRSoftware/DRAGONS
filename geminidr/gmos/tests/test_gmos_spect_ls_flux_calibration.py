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

from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.library import astromodels
from gempy.utils import logutils

from .test_gmos_spect_ls_apply_qe_correction import (
    cache_path, get_associated_calibrations, get_master_arc, output_path,
    reduce_arc, reduce_bias, reduce_flat, reference_ad)


datasets = [
    "N20180109S0287.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
]


# --- Tests -------------------------------------------------------------------
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
        p.fluxCalibrate(standard=input_ad.object())
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
        cals = get_associated_calibrations(basename)

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
