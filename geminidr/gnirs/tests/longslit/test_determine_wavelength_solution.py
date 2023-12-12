#!/usr/bin/env python
"""
Tests related to GNIRS Long-slit Spectroscopy Arc primitives.

"""
# import multiprocessing as mp
# mp.set_start_method('fork')

import glob
import tarfile
import logging

import numpy as np
import pytest
from matplotlib import pyplot as plt
from importlib import import_module

import astrodata
import geminidr

from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from gempy.library import astromodels as am
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals, set_btypes

import os
from astrodata.testing import download_from_archive
from geminidr.gnirs.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS

# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'global',
    'fwidth': None,
    'order': None,
    'min_snr': None,
    'debug_min_lines': None,
    'in_vacuo': True,
    'num_atran_lines': None,
    "combine_method": "optimal",
    "wv_band": "header",
    "resolution": None
}

input_pars = [
    # Process Arcs: GNIRS ---
    # (Input File, params)
    # If a grating+camera+site combination is not present here -> no data exists
    # 10 l/mm, LongCam, GN
    ("N20210810S0833_varAdded.fits", dict()), # X, 1.100um, 0.1" slit
    ("N20210810S0835_varAdded.fits", dict()), # J, 1.250um, 0.1" slit
    ("N20140713S0193_varAdded.fits", dict()), # H, 1.650um, 0.1" slit
    ("N20210803S0133_varAdded.fits", dict()), # K, 2.220um, 0.1" slit
    # 32 l/mm, ShortCam, GN
    ("N20140413S0124_varAdded.fits", dict()), # X, 1.100um, 1" slit
    ("N20210810S0819_varAdded.fits", dict()), # J, 1.250um, 0.3" slit
    ("N20181102S0037_varAdded.fits", dict()), # H, 1.650um, 0.68" slit
    ("N20130101S0207_varAdded.fits", dict()), # K, 2.220um, 0.3" slit
    ("N20170511S0305_varAdded.fits", dict()), # K, 2.110um, 0.68" slit
    # 32 l/mm, ShortCam, GS
    ("S20060719S0224_varAdded.fits", dict()), # H, 1.650um, 1" slit
    ("S20060314S0319_varAdded.fits", dict()), # K, 2.220um, 0.675" slit
    # 32 l/mm, LongCam, GN
    ("N20210810S0843_varAdded.fits", dict()), # J, 1.250um, 0.1" slit
    ("N20210810S0845_varAdded.fits", dict()), # H, 1.650um, 0.1" slit
    ("N20191221S0157_varAdded.fits", dict()), # K, 2.350um, 0.1" slit
    ("N20210810S0847_varAdded.fits", dict()), # K, 2.200um, 0.1" slit
    # 111 l/mm, ShortCam, GN
    ("N20210722S0332_varAdded.fits", dict()), # X, 1.083um, 0.3" slit
    ("N20220222S0151_varAdded.fits", dict()), # X, 1.120um, 1" slit
    ("N20220222S0199_varAdded.fits", dict()), # X, 1.160um, 1" slit
    ("N20101203S0888_varAdded.fits", dict()), # J, 1.250um, 0.45" slit
    ("N20110923S0478_varAdded.fits", dict()), # J, 1.340um, 1" slit
    ("N20130808S0145_varAdded.fits", dict()), # H, 1.591um, 0.3" slit
    ("N20210613S0358_varAdded.fits", dict()), # H, 1.740um, 1" slit, 6 lines
    ("N20170429S0265_varAdded.fits", dict()), # K, 2.020um, 0.45" slit
    ("N20220303S0271_varAdded.fits", dict()), # K, 2.100um, 0.3" slit
    ("N20160618S0255_varAdded.fits", dict()), # K, 2.200um, 0.45" slit, 6 lines
    ("N20210615S0171_varAdded.fits", dict()), # K, 2.330um, 0.3" slit, 5 lines
    # 111 l/mm, ShortCam, GS
    ("S20060214S0406_varAdded.fits", dict()), # X, 1.083um, 0.3" slit
    ("S20051109S0095_varAdded.fits", dict()), # J, 1.207um, 0.675" slit
    ("S20040614S0367_varAdded.fits", dict()), # H, 1.475um, 0.3" slit
    ("S20050927S0054_varAdded.fits", dict(min_snr=50)), # K, 2.360um, 0.3" slit, 4 lines, pattern noise
    ("S20040904S0321_varAdded.fits", dict()), # K, 2.400um, 0.675" slit, very bright bkg
    # 111 l/mm, LongCam, GN
    ("N20111018S1547_varAdded.fits", dict()), # X, 1.050um, 0.2" slit
    ("N20121221S0204_varAdded.fits", dict()), # X, 1.083um, 0.1" slit
    ("N20210810S0865_varAdded.fits", dict()), # X, 1.100um, 0.1" slit
    ("N20111018S1549_varAdded.fits", dict()), # X, 1.110um, 0.2" slit
    ("N20150602S0501_varAdded.fits", dict()), # J, 1.180um, 0.1" slit
    ("N20100815S0237_varAdded.fits", dict()), # J, 1.270um, 0.1" slit, SV data with large cenwave shift.
    ("N20200914S0035_varAdded.fits", dict()), # J, 1.300um, 0.15" slit
    ("N20111020S0664_varAdded.fits", dict()), # J, 1.340um, 0.45" slit
    ("N20110727S0332_varAdded.fits", dict()), # H, 1.520um, 0.1" slit
    ("N20110601S0753_varAdded.fits", dict()), # H, 1.565um, 0.1" slit
    ("N20180605S0147_varAdded.fits", dict()), # H, 1.600um, 0.1" slit
    ("N20140801S0183_varAdded.fits", dict()), # H, 1.690um, 0.1" slit
    ("N20141218S0289_varAdded.fits", dict()), # H, 1.710um, 0.1" slit, 2 lines
    ("N20111019S1056_varAdded.fits", dict()), # H, 1.750um, 0.45" slit
    ("N20130624S0196_varAdded.fits", dict()), # H, 1.778um, 0.1" slit
    ("N20110531S0510_varAdded.fits", dict()), # K, 2.030um, 0.1" slit
    ("N20110715S0523_varAdded.fits", dict()), # K, 2.090um, 0.1" slit
    ("N20160109S0118_varAdded.fits", dict()), # K, 2.115um, 0.15" slit
    ("N20150613S0087_varAdded.fits", dict()), # K, 2.166um, 0.1" slit
    ("N20110618S0031_varAdded.fits", dict()), # K, 2.208um, 0.3" slit
    ("N20110923S0569_varAdded.fits", dict()), # K, 2.260um, 0.1" slit
    ("N20201022S0051_varAdded.fits", dict()), # K, 2.310um, 0.1" slit
    ("N20160804S0155_varAdded.fits", dict()), # K, 2.350um, 0.1" slit
    ("N20121218S0198_varAdded.fits", dict()), # K, 2.420um, 0.1" slit
    ("N20161217S0001_varAdded.fits", dict()), # K, 2.500um, 0.1" slit, one line, no solution
    # 111 l/mm, LongCam, GS
    ("S20040907S0141_varAdded.fits", dict()), # J, 1.200um, 0.1" slit
    ("S20040907S0129_varAdded.fits", dict()), # H, 1.600um, 0.1" slit
    #("S20040907S0122_varAdded.fits", dict()), # K, 1.975um, 0.1" slit. Works with smaller snr.
    ("S20050119S0117_varAdded.fits", dict()), # K, 2.170um, 0.1" slit, one line, no solution
    ("S20050119S0122_varAdded.fits", dict()), # K, 2.300um, 0.1" slit, one line, no solution

    # Thermal infrared, wavecal from sky lines
    # L-band, 111 l/mm, LongCam
    ("N20130204S0068_varAdded.fits", dict()), #	3005nm, 2nm, R=19000, 0.10arcsec
    #("N20101209S0201_varAdded.fits", dict()), # 3329nm, 19nm, R=12666, 0.15arcsec. Large dw, SV data
    #("N20101209S0184_varAdded.fits", dict()), # 3330nm, 18nm, R=12666, 0.15arcsec. Large dw, SV data
    #("N20101209S0174_varAdded.fits", dict()), # 3345nm, 18nm, R=12666, 0.15arcsec. Large dw, SV data
    #("N20111017S0101_varAdded.fits", dict()), # 3348nm, 9nm, R=12666, 0.15arcsec. Large dw
    ("N20200827S0024_varAdded.fits", dict()), #	3360nm, 4.9nm, R=19000, 0.10arcsec
    ("N20130113S0134_varAdded.fits", dict()), #	3360nm, 0nm, R=19000, 0.10arcsec
    #("N20170519S0188_varAdded.fits", dict()), # 3425nm, 4.9nm, R=19000, 0.10arcsec. Unstable (might pick diff lines for solution). The stack is stable
    ("N20130629S0193_varAdded.fits", dict()), #	3530nm, 2nm, R=19000, 0.10arcsec
    ("N20130709S0165_varAdded.fits", dict()), #	3690nm, 0.1nm, R=19000, 0.10arcsec
    ("N20130815S0354_varAdded.fits", dict()), #	3695nm, 2.8nm,	R=19000, 0.10arcsec
    #("N20100820S0188_varAdded.fits", dict()), # 3700nm, 11nm, R=1900, 1.0arcsec. Large dw
    #("N20160126S0091_varAdded.fits", dict()), # 3900nm, 5.8nm, R=19000, 0.10arcsec, wrong/unstable solution, >3800nm region
    #("N20170519S0315_varAdded.fits", dict()), # 3953nm, 3nm, R=19000, 0.10arcsec, wrong/unstable solution, >3800nm region
    #("N20150530S0202_varAdded.fits", dict()), # 3970nm, ?nm, R=19000, 0.10arcsec, wrong/unstable solution, >3800nm region
    #("N20190419S0144_varAdded.fits", dict()), # 4110nm, ?nm, R=19000, 0.10arcsec, wrong/unstable solution, >3800nm region
    # L-band, 111 l/mm, ShortCam
    ("N20100820S0214_varAdded.fits", dict()), #	3200nm,	11nm, R=1920, 1.0arcsec.
    ("N20100820S0218_varAdded.fits", dict()), #	3400nm,	12nm, R=1920, 1.0arcsec.
    ("S20060214S0266_varAdded.fits", dict()), #	3400nm,	3nm, R=6400, 0.3arcsec
    ("N20100821S0281_varAdded.fits", dict()), #	3500nm,	11nm, R=4266, 0.45arcsec
    ("N20100821S0242_varAdded.fits", dict()), #	3600nm,	9nm, R=1920, 1.0arcsec.
    ("N20110724S0207_varAdded.fits", dict()), #	3640nm,	2nm, R=4266, 0.45arcsec
    ("S20060811S0056_varAdded.fits", dict()), #	3680nm,	6nm, R=4266, 0.45arcsec
    ("N20110106S0296_varAdded.fits", dict()), #	3700nm,	3nm, R=6400, 0.30arcsec
    ("N20100821S0247_varAdded.fits", dict()), #	3800nm,	11nm, R=1920, 1.0arcsec
    # the automatic solution is wrong/unstable beyond 3850nm due to regular line pattern and/or low signal
    #("N20101209S0316_varAdded.fits", dict()), #	3940nm,	?nm, R=6400, 0.30arcsec
    # L-band, 32 l/mm, LongCam
    ("N20130822S0258_varAdded.fits", dict()), #	3400nm,	13nm, R=5400, 0.10arcsec
    ("N20130122S0184_varAdded.fits", dict()), #	3500nm,	3nm, R=5400, 0.10arcsec
    #("N20121211S0367_varAdded.fits", dict()), #	3700nm,	15nm, R=5400, 0.10arcsec. Unstable solution
    ("N20121217S0176_varAdded.fits", dict()), #	3700nm,	16nm, R=5400, 0.10arcsec
    #("N20100820S0180_varAdded.fits", dict()), # 3700nm, 38nm, R=540, 1.0arcsec. Large dw, SV data
    #("N20131206S0129_varAdded.fits", dict()), # 3900nm, 11nm, R=5400, 0.10arcsec. Unstable ("comb" region)
    # L-band, 32 l/mm, ShortCam
    #("N20100821S0263_varAdded.fits", dict()), # 3300nm, 59nm, R=540, 1.0arcsec. Unstable (but all solutions are right). Stack is stable.
    ("N20101207S0295_varAdded.fits", dict()), #	3300nm,	62nm, R=794, 0.68arcsec
    ("N20111007S0439_varAdded.fits", dict()), #	3350nm,	31nm, R=1200, 0.45arcsec
    ("S20051113S0085_varAdded.fits", dict()), #	3400nm,	18nm, R=1200, 0.45arcsec
    ("S20070325S0115_varAdded.fits", dict()), #	3400nm,	8nm, R=1200, 0.45arcsec
    ("N20110601S0327_varAdded.fits", dict()), #	3500nm,	5nm, R=1800, 0.30arcsec
   # ("N20100722S0225_varAdded.fits", dict()), # 3600nm, 42nm, R=1800, 0.30arcsec. Unstable. Probably ok with a stack.
    ("N20101205S0202_varAdded.fits", dict()), #	3400nm,	61nm, R=540, 1.0arcsec, SV
    ("N20110326S0346_varAdded.fits", dict()), #	3400nm,	3nm, R=794, 0.68arcsec
    #("N20101203S0315_varAdded.fits", dict()), # 3700nm, 68nm, R=1800, 0.30arcsec. Unstable, gradient.
    #("N20100915S0155_varAdded.fits", dict(min_snr=5)), # 3770nm, 63nm, R=1800, 0.30arcsec, crazy gradient, unstable
    # L-band, 10 l/mm, LongCam
    ("N20160321S0222_varAdded.fits", dict()), #	3100nm,	35nm, R=264, 0.68arcsec
    #("N20110718S0086_varAdded.fits", dict()), # 3300nm, 97nm, R=400, 0.45arcsec. Very large dw
    ("N20130516S0183_varAdded.fits", dict()), #	3400nm,	25nm, R=400, 0.45arcsec
    ("N20161115S0402_varAdded.fits", dict()), #	3400nm,	46nm, R=1200, 0.15arcsec
    ("N20170620S0138_varAdded.fits", dict()), #	3400nm,	35nm, R=264, 0.68arcsec, no solution (too low res)
   # ("N20121201S0126_varAdded.fits", dict()), # 3500nm, 8nm, R=600, 0.30arcsec. Unstable. Stacking doesn't help, the solution might be wrong.
    ("N20180817S0152_varAdded.fits", dict()), #	3500nm,	21nm, R=1200, 0.15arcsec
    ("N20160810S0300_varAdded.fits", dict()), #	3500nm,	29nm, R=400, 0.45arcsec
    ("N20200704S0207_varAdded.fits", dict()), #	3530nm,	41nm, R=265, 0.68arcsec
    ("N20160909S0082_varAdded.fits", dict()), #	3550nm,	35nm, R=900, 0.20arcsec
    ("N20200908S0127_varAdded.fits", dict(min_snr=5)), # 3560nm, 42nm, R=600, 0.30arcsec
    ("N20121212S0216_varAdded.fits", dict()), #	3700nm,	7nm, R=1800, 0.10arcsec
    # M-band, 111 l/mm, LongCam
    ("N20110331S0183_varAdded.fits", dict()), #	4634nm,	8nm, R=12800, 0.10arcsec
    ("N20220724S0152_varAdded.fits", dict()), #	4670nm,	1nm, R=6400, 0.20arcsec
    ("N20160321S0242_varAdded.fits", dict()), #	4670nm,	11nm, R=1882, 0.68arcsec.
    ("N20180813S0029_varAdded.fits", dict()), #	4672nm,	11nm, R=2844, 0.45arcsec
    ("N20151229S0288_varAdded.fits", dict()), #	4690nm,	49nm, R=12800, 0.10arcsec
    ("N20151229S0221_varAdded.fits", dict()), #	4690nm,	9nm, R=12800, 0.10arcsec
    #("N20100820S0195_varAdded.fits", dict()), # 4850nm, 17nm, R=1280, 1.0arcsec. Wrong solution, SV data. Very low res
    # M-band, 111 l/mm, ShortCam
    ("N20100821S0255_varAdded.fits", dict()), #	4700nm,	23nm, R=1290, 1.0arcsec
    ("S20040412S0071_varAdded.fits", dict()), #	4750nm,	6nm, R=4300, 0.30arcsec
    ("N20101205S0275_varAdded.fits", dict()), #	4800nm,	41nm, R=2866, 0.45arcsec
    ("S20040413S0097_varAdded.fits", dict()), #	4850nm,	9nm, R=4300, 0.30arcsec
    ("S20050223S0092_varAdded.fits", dict()), #	4850nm,	5nm, R=4300, 0.30arcsec
    ("N20100821S0260_varAdded.fits", dict()), #	5100nm,	25nm, R=1290, 1.0arcsec
    # M-band, 32 l/mm, LongtCam
    ("S20040308S0034_varAdded.fits", dict()), # 4749nm, 0nm, R=1850, 0.20arcsec
    ("N20170708S0248_varAdded.fits", dict(min_snr=5)), # 4849nm, 42nm, R=544, 0.68arcsec
    #("N20100820S0182_varAdded.fits", dict()), # 4849nm, 73nm, R=370, 1.0arcsec. Large dw, SV data
    # M-band, 32 l/mm, ShortCam
    ("N20100821S0270_varAdded.fits", dict()), # 4850nm, 86nm, R=372, 1.0arcsec
    # M-band, 10 l/mm, LongtCam
    ("N20100820S0160_varAdded.fits", dict()), # 4850nm, ?nm, R=120, 1.0arcsec # no solution

    # wavecal from OH emission sky lines. Currently flatCorrect() is included in
    # GNIRS arc recipes, since it can improve distortion model. It doesn't however
    # affect wavelength calibration. Still, do test on one flat-corrected arc just in case:
    ("N20181102S0023_flatCorrected.fits", dict()), # 32 l/mm, 1.650um.
    # wavecal from sky absorption
    ("N20121216S0120_aperturesFound.fits", dict(absorption=True)), # 111l/mm, 2.362um.

]
associated_calibrations_oh_emis = {
    "N20181102S0023.fits": {
        'flat': ["N20181102S0031.fits"],
    }
}
associated_calibrations_absorp = {
    "N20121216S0120.fits": {
        'flat': ["N20121216S0106.fits"],
        'arc': ["N20121216S0105.fits"],
    }
}


# Tests Definitions ------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad, params", input_pars, indirect=['ad'])
def test_regression_determine_wavelength_solution(
        ad, params, caplog, change_working_dir, path_to_refs, request):
    """
    Make sure that the wavelength solution gives same results on different
    runs.
    """
    caplog.set_level(logging.INFO, logger="geminidr")

    with change_working_dir():
        logutils.config(file_name='log_regress_{:s}.txt'.format(ad.data_label()))
        p = GNIRSLongslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)

        p.determineWavelengthSolution(**{**determine_wavelength_solution_parameters,
                                         **params})

        wcalibrated_ad = p.streams["main"][0]

        for record in caplog.records:
            if record.levelname == "WARNING":
                assert "No acceptable wavelength solution found" not in record.message

    ref_ad = astrodata.open(os.path.join(path_to_refs, wcalibrated_ad.filename))
    model = am.get_named_submodel(wcalibrated_ad[0].wcs.forward_transform, "WAVE")
    ref_model = am.get_named_submodel(ref_ad[0].wcs.forward_transform, "WAVE")

    x = np.arange(wcalibrated_ad[0].shape[1])
    wavelength = model(x)
    ref_wavelength = ref_model(x)

    pixel_scale = wcalibrated_ad[0].pixel_scale()  # arcsec / px
    slit_size_in_arcsec = float(wcalibrated_ad[0].focal_plane_mask(pretty=True).replace('arcsec', ''))
    slit_size_in_px = slit_size_in_arcsec / pixel_scale
    dispersion = abs(wcalibrated_ad[0].dispersion(asNanometers=True))  # nm / px

    # We don't care about what the wavelength solution is doing at
    # wavelengths outside where we've matched lines
    lines = ref_ad[0].WAVECAL["wavelengths"].data
    indices = np.where(np.logical_and(ref_wavelength > lines.min(),
                                      ref_wavelength < lines.max()))
    tolerance = 0.5 * (slit_size_in_px * dispersion)

    write_report = request.config.getoption('--do-report', False)
    failed = False
    try:
        np.testing.assert_allclose(wavelength[indices], ref_wavelength[indices],
                               atol=tolerance)
    except AssertionError:
        failed = True
        raise
    finally:
        if write_report:
            do_report(wcalibrated_ad, ref_ad, failed=failed)

    if request.config.getoption("--do-plots"):
        do_plots(wcalibrated_ad)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the
        `determineWavelengthSolution` primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad

def get_linelist(ad):
    is_lowres = ad.disperser(pretty=True).startswith('10') or \
                    (ad.disperser(pretty=True).startswith('32') and
                        ad.camera(pretty=True).startswith('Short'))
    if 'ARC' in ad.tags:
        if 'Xe' in ad.object():
            linelist ='Ar_Xe.dat'
        elif "Ar" in ad.object():
            if is_lowres:
                linelist = 'lowresargon.dat'
            else:
                linelist = 'argon.dat'
        else:
            raise ValueError(f"No default line list found for {ad.object()}-type arc. Please provide a line list.")
    else:
        if ad.filter_name(pretty=True).startswith('L'):
            linelist = 'skyLband.dat'
        elif ad.filter_name(pretty=True).startswith('M'):
            linelist = 'skyMband.dat'
        elif is_lowres:
            linelist = 'sky.dat'
        else:
            linelist = 'nearIRsky.dat'

    return linelist

# for this to work use --do-report as test argument, and
# add it to ~/pytest_dragons/pytest_dragons/plugin.py
def do_report(ad, ref_ad, failed):
    """
    Generate text file with test details.

    """
    output_dir = ("../DRAGONS_tests/geminidr/gnirs/longslit/"
                  "test_determine_wavelength_solution")
    os.makedirs(output_dir, exist_ok=True)
    report_filename = 'report.txt'
    report_path = os.path.join(output_dir, report_filename)

    ref_wavecal_model = am.get_named_submodel(ref_ad[0].wcs.forward_transform, "WAVE")
    wavecal_model = am.get_named_submodel(ad[0].wcs.forward_transform, "WAVE")
    domain = wavecal_model.domain
    dw = np.diff(wavecal_model(domain))[0] / np.diff(domain)[0]
    ref_dw = np.diff(ref_wavecal_model(domain))[0] / np.diff(domain)[0]
    nmatches = np.count_nonzero(ad[0].WAVECAL['peaks'])
    ref_nmatches = np.count_nonzero(ref_ad[0].WAVECAL['peaks'])
    rms = ad[0].WAVECAL['coefficients'][6]
    ref_rms = ref_ad[0].WAVECAL['coefficients'][6]
    fwidth = ad[0].WAVECAL['coefficients'][7]
    ref_fwidth = ref_ad[0].WAVECAL['coefficients'][7]

    with open(report_path, 'a') as report_output:
        if os.lseek(report_output.fileno(), 0, os.SEEK_CUR) == 0:
            print("Filename matched_lines final_order cenwave_delta disp_delta fwidth rms",
                  file=report_output)
        if failed:
            print("Reference parameters:",
                  file=report_output)
            print(f"{ref_ad.filename} {ref_nmatches} {ref_wavecal_model.degree} "
                  f"{((ref_wavecal_model(511)-ref_ad[0].central_wavelength(asNanometers=True))):.1f} {(ref_dw-ref_ad[0].dispersion(asNanometers=True)):.3f} "
                  f"{ref_fwidth} {ref_rms}", file=report_output)
            print("Failed test file parameters:",
                  file=report_output)
        print(f"{ad.filename} {nmatches} {wavecal_model.degree} "
                  f"{((wavecal_model(511)-ad[0].central_wavelength(asNanometers=True))):.1f} {(dw-ad[0].dispersion(asNanometers=True)):.3f}",
                  f"{fwidth} {rms}", file=report_output)


def do_plots(ad):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : astrodata
    """
    output_dir = ("./plots/geminidr/gnirs/"
                  "test_gnirs_spect_ls_determine_wavelength_solution")
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grism = ad.disperser(pretty=True)
    filter = ad.filter_name(pretty=True)
    camera = ad.camera(pretty=True)

    central_wavelength = ad.central_wavelength(asNanometers=True)

    p = GNIRSLongslit([ad])
    lookup_dir = os.path.dirname(import_module('.__init__',
                                                   p.inst_lookups).__file__)
    arc_table = os.path.join(lookup_dir, get_linelist(ad))
    arc_lines = np.loadtxt(arc_table, usecols=[0]) / 10.0

    def save_filename(*args):
        "Construct a filename out of several components"
        args = [('{:.0f}'.format(arg) if isinstance(arg, (float, int)) else arg)
                for arg in args]
        return '_'.join(args).replace('/', '_')

    for ext_num, ext in enumerate(ad):

        if not hasattr(ext, "WAVECAL"):
            continue

        peaks = ext.WAVECAL["peaks"] - 1  # ToDo: Refactor peaks to be 0-indexed
        wavelengths = ext.WAVECAL["wavelengths"]
        wavecal_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")

        middle = ext.data.shape[0] // 2
        sum_size = 10
        r1 = middle - sum_size // 2
        r2 = middle + sum_size // 2

        mask = np.round(np.average(ext.mask[r1:r2], axis=0)).astype(int)
        data = np.ma.masked_where(mask > 0, np.sum(ext.data[r1:r2], axis=0))
        data = (data - data.min()) / data.ptp()

        # -- Plot lines --
        fig, ax = plt.subplots(
            dpi=150, num=save_filename(name, grism, filter, camera, central_wavelength))

        w = wavecal_model(np.arange(data.size))

        arcs = [ax.vlines(line, 0, 1, color="k", alpha=0.25) for line in arc_lines]
        wavs = [ax.vlines(peak, 0, 1, color="r", ls="--", alpha=0.25)
                for peak in wavecal_model(peaks)]

        plot, = ax.plot(w, data, "k-", lw=0.75)

        ax.legend((plot, arcs[0], wavs[0]),
                  ("Normalized Data", "Reference Lines", "Matched Lines"))

        x0, x1 = wavecal_model([0, data.size])
        ax.grid(alpha=0.1)
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Normalized intensity")
        ax.set_title("Wavelength Calibrated Spectrum for\n"
                     "{:s}\n obtained with {:s},{:s},{:s} at {:.0f} nm".format(
                        name, grism, filter, camera, central_wavelength))

        if x0 > x1:
            ax.invert_xaxis()

        fig_name = os.path.join(output_dir,
                        save_filename(name, grism, filter, camera, central_wavelength))

        fig.savefig(fig_name)
        del fig, ax

        # -- Plot non-linear components ---
        fig, ax = plt.subplots(
            dpi=150, num="{:s}_{:s}_{:s}_{:s}_{:.0f}_non_linear_comps".format(
                name, grism, filter, camera, central_wavelength))

        non_linear_model = wavecal_model.copy()
        _ = [setattr(non_linear_model, "c{}".format(k), 0) for k in [0, 1]]
        residuals = wavelengths - wavecal_model(peaks)

        p = np.linspace(min(peaks), max(peaks), 1000)
        ax.plot(wavecal_model(p), non_linear_model(p),
                "C0-", label="Generic Representation")
        ax.plot(wavecal_model(peaks), non_linear_model(peaks) + residuals,
                "ko", label="Non linear components and residuals")

        ax.legend()
        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_title("Non-linear components for\n"
                     "{:s} obtained with {:s},{:s},{:s} at {:.0f}".format(
                        name, grism, filter, camera, central_wavelength))

        fig_name = os.path.join(output_dir,
                        save_filename(name, grism, filter, camera, central_wavelength, "_non_linear_comps.png"))

        fig.savefig(fig_name)
        del fig, ax

        # -- Plot Wavelength Solution Residuals ---
        fig, ax = plt.subplots(
            dpi=150, num="{:s}_{:s}_{:s}_{:s}_{:.0f}_residuals".format(
                name, grism, filter, camera, central_wavelength))

        ax.plot(wavelengths, wavelengths - wavecal_model(peaks), "ko")

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Residuum [nm]")
        ax.set_title("Wavelength Calibrated Residuum for\n"
                     "{:s} obtained with {:s},{:s},{:s} at {:.0f}".format(
                        name, grism, filter, camera, central_wavelength))

        fig_name = os.path.join(output_dir,
                save_filename(name, grism, filter, camera, central_wavelength, "_residuals.png"))

        fig.savefig(fig_name)

    # -- Create artifacts ---
    if "BUILD_ID" in os.environ:
        branch_name = os.environ["BRANCH_NAME"].replace("/", "_")
        build_number = int(os.environ["BUILD_NUMBER"])

        tar_name = os.path.join(output_dir, "plots_{:s}_b{:03d}.tar.gz".format(
            branch_name, build_number))

        with tarfile.open(tar_name, "w:gz") as tar:
            for _file in glob.glob(os.path.join(output_dir, "*.png")):
                tar.add(name=_file, arcname=os.path.basename(_file))

        target_dir = "./plots/"
        target_file = os.path.join(target_dir, os.path.basename(tar_name))

        os.makedirs(target_dir, exist_ok=True)
        os.rename(tar_name, target_file)

# -- Recipe to create pre-processed data ---------------------------------------
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, _ in input_pars:
        print('Downloading files...')
        basename = filename.split("_")[0] + ".fits"
        sci_path = download_from_archive(basename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = GNIRSLongslit([sci_ad])
        p.prepare(bad_wcs="fix")
        p.addDQ()
        p.addVAR(read_noise=True)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        #  p.flatCorrect()
        p.makeIRAFCompatible()

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_ad.filename))

    for filename, cals in associated_calibrations_oh_emis.items():
        print(filename)

        arc_path = download_from_archive(filename)
        flat_path = [download_from_archive(f) for f in cals['flat']]

        arc_ad = astrodata.open(arc_path)
        data_label = arc_ad.data_label()

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_path)
        flat_reduce.uparms = [('normalizeFlat:threshold','0.01')]
        flat_reduce.runr()
        processed_flat = flat_reduce.output_filenames.pop()
        del flat_reduce

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        p = GNIRSLongslit([arc_ad])
        p.prepare(bad_wcs="fix")
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.flatCorrect(flat=processed_flat, suffix="_flatCorrected")
        p.makeIRAFCompatible()

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_ad.filename))
        
    for filename, cals in associated_calibrations_absorp.items():
        print(filename)

        arc_path = download_from_archive(filename)
        flat_path = [download_from_archive(f) for f in cals['flat']]
        arc_arc_path = [download_from_archive(f) for f in cals['arc']]

        arc_ad = astrodata.open(arc_path)
        data_label = arc_ad.data_label()

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_path)
        flat_reduce.uparms = [('normalizeFlat:threshold','0.01')]
        flat_reduce.runr()
        processed_flat = flat_reduce.output_filenames.pop()
        calibration_files = ['processed_flat:{}'.format(processed_flat)]
        del flat_reduce

        logutils.config(file_name='log_arc_arc_{}.txt'.format(data_label))
        arc_arc_reduce = Reduce()
        arc_arc_reduce.files.extend(arc_arc_path)
        arc_arc_reduce.ucals = normalize_ucals(calibration_files)
        arc_arc_reduce.runr()
        processed_arc_arc = arc_arc_reduce.output_filenames.pop()
        del arc_arc_reduce

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        p = GNIRSLongslit([arc_ad])
        p.prepare(bad_wcs="fix")
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=True)
        p.flatCorrect(flat=processed_flat, suffix="_flatCorrected")
        p.attachWavelengthSolution(arc=processed_arc_arc, suffix="_wavelengthSolutionAttached")
        p.copyInputs(instream="main", outstream="with_distortion_model")
        p.separateSky()
        p.associateSky()
        p.skyCorrect()
        p.cleanReadout()
        p.distortionCorrect()
        p.adjustWCSToReference()
        p.resampleToCommonFrame(force_linear=False)
        #p.scaleCountsToReference()
        p.stackFrames()
        p.findApertures()

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_ad.filename))

def create_refs_recipe():
    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("refs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, params in input_pars:
        ad = astrodata.open(os.path.join('inputs', filename))
        p = GNIRSLongslit([ad])
        p.determineWavelengthSolution(**{**determine_wavelength_solution_parameters,
                                         **params})
        os.chdir('refs/')
        p.writeOutputs()
        os.chdir('..')

if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    if "--create-refs" in sys.argv[1:]:
        create_refs_recipe()
    else:
        pytest.main()
