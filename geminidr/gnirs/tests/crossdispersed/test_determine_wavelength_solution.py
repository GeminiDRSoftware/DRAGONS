import logging
import os
import pytest
from gempy.utils import logutils
from gempy.library import astromodels as am

import astrodata, gemini_instruments, geminidr
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from geminidr.gnirs.tests.crossdispersed import CREATED_INPUTS_PATH_FOR_TESTS

import glob
import tarfile

import numpy as np
from matplotlib import pyplot as plt
from importlib import import_module

from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals, set_btypes
from astrodata.testing import download_from_archive

@pytest.mark.gnirsxd
@pytest.mark.wavecal
@pytest.mark.preprocessed_data
def test_determine_wavelength_solution_wcs_has_2_world_axes(path_to_inputs, caplog):
    """
    Check that the WCS has two world axes after determining the wavelength
    solution. This dataset fails to find a solution on at least one extension,
    so is a good check that the WCS is correct regardless of whether a solution
    is found or not.
    """
    caplog.set_level(logging.INFO, logger="geminidr")
    ad = astrodata.open(os.path.join(path_to_inputs,
                                     "N20190928S0085_aperturesFound.fits"))
    p = GNIRSCrossDispersed([ad])
    adout = p.determineWavelengthSolution(absorption=True).pop()

    for record in caplog.records:
        if "Matched 0/0" in record.message:
            break
    else:
        pytest.fail("No log message indicating that no solution was found.")

    for ext in adout:
        assert ext.wcs.output_frame.naxes == 2, f"Problem with extension {ext.id}"



# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'global',
    'fwidth': None,
    'order': None,
    'min_snr': None,
    'debug_min_lines': '50,20',
    'in_vacuo': True,
    'num_lines': 50,
    "combine_method": "mean",
    "wv_band": "header",
    "resolution": None
}

input_pars = [
    # (Input File, params)
    # Arcs

    # OS: I commented out the test cases that contain line misidentifications in any order (they may be useful in the future),
    # and only left those that have correct solutions in all orders, or no solution at all (due to low signal).

    # 10 l/mm, LongCam, LXD
    ("N20190613S0173_pinholeRectificationAttached.fits", dict()), # 1.650um, 0.45" slit. All orders have correct solution.
    # 32 l/mm ShortCam, SXD
    ("N20201229S0067_pinholeRectificationAttached.fits", dict()), # 2.150um, 0.3" slit. All orders have correct solution.
    ("N20210228S0149_pinholeRectificationAttached.fits", dict()), # 1.65um, 0.68" slit. All orders have correct solution.
    # 32 l/mm LongCam, LXD
    ("N20201224S0198_pinholeRectificationAttached.fits", dict(min_snr=5)), # 1.98um, 0.45" slit. All orders have correct solution with min_snr=5
    ("N20201222S0222_pinholeRectificationAttached.fits", dict()), # 1.58um, 0.45" slit. All orders have correct solution (last order needs a lower fit order)
    # 111 l/mm ShortCam, SXD
    ("N20160907S0321_pinholeRectificationAttached.fits", dict()), # 1.53um, 0.3" slit. Misid. in ord. 6 due to missing line in the linelist.
    ("N20240821S0271_pinholeRectificationAttached.fits", dict(order=1)), # 1.65um, 0.3" slit. Ok with fit ord=1, otherwise misident in ord. 7 and 8
    #("N20140803S0102_pinholeRectificationAttached.fits", dict()), # 1.77um, 0.3" slit. Ord. 7 - misident.
    # 111 l/mm LongCam, LXD
    ("N20130630S0467_pinholeRectificationAttached.fits", dict()), # 1.942um, 0.1" slit. Order 3 has not arc lines, the rest ok
    ("N20130630S0468_pinholeRectificationAttached.fits", dict()), # 2.002um, 0.1" slit. Ok
    ("N20130630S0469_pinholeRectificationAttached.fits", dict()), # 2.062um, 0.1" slit. Ok, last order - low signal
    ("N20200818S0348_pinholeRectificationAttached.fits", dict()), # 2.122um, 0.1" slit. All orders have correct solution, one line misident.
    ("N20200818S0349_pinholeRectificationAttached.fits", dict()), # 2.182um, 0.1" slit. Ok
    #("N20200818S0350_pinholeRectificationAttached.fits", dict()), # 2.242um, 0.1" slit.  Only ord 4 and 5 have good solution. Other orders - low signal or to few lines.
    ("N20220816S0702_pinholeRectificationAttached.fits", dict()), # 2.302um, 0.1" slit. Only ord. 5 and 6 have solution. Other orders - low signal or to few lines.
    ("N20220816S0703_pinholeRectificationAttached.fits", dict()), # 2.362um, 0.1" slit. Ord 4,5,6 have solution, the rest -low signal or to few lines.
    ("N20220816S0704_pinholeRectificationAttached.fits", dict()), # 2.422um, 0.1" slit. Only ord 4 and 5 have enough lines for a solution.
    ("N20220816S0705_pinholeRectificationAttached.fits", dict()), # 2.482um, 0.1" slit. Ok solution for 3-6.

    # Airglow
    # 10 l/mm, LongCam, LXD
    #("N20190613S0147_pinholeRectificationAttached.fits", dict()), # 1.650um, 0.45" slit. Orders 3-5 correct with min_snr=5, but ord 7 - misident, low snr
    # 32 l/mm ShortCam, SXD
    ("N20201229S0029_pinholeRectificationAttached.fits", dict(min_snr=10)), # 2.150um, 0.3" slit. All orders ok with min_snr=10
    ("N20210228S0111_pinholeRectificationAttached.fits", dict(min_snr=5)), # 1.65um, 0.68" slit. All orders correct
    # 32 l/mm LongCam, LXD
    ("N20201223S0043_pinholeRectificationAttached.fits", dict()), # 1.98um, 0.45" slit. Ord 3,4 - ok; 5,6-low signal, no solution
    #("N20201222S0188_pinholeRectificationAttached.fits", dict()), # 1.58um, 0.45" slit. Ord 3,4,5-ok with snr=10; 6-misident, 7-no solution(low snr)
    # 111 l/mm ShortCam, SXD
    ("N20160907S0320_pinholeRectificationAttached.fits", dict()), # 1.53um, 0.3" slit. Last ord - low signal, no solution.
    # ("N20240821S0257_pinholeRectificationAttached.fits", dict()), # 1.65um, 0.3" slit. Ord 3-6 - ok, 7,8 - misident, or low signal
    ("N20140803S0125_pinholeRectificationAttached.fits", dict(order=1)), # 1.77um, 0.3" slit. Last order low signal. Other orders ok with ord=1
    # 111 l/mm LongCam, LX
    #("N20130630S0158_pinholeRectificationAttached.fits", dict()), # 1.942um, 0.1" slit. No bright lines.
    #("N20130630S0173_pinholeRectificationAttached.fits", dict()), # 2.002um, 0.1" slit. -//-
    #("N20130630S0187_pinholeRectificationAttached.fits", dict()), # 2.062um, 0.1" slit. -//-
    # #("N20200818S0104_pinholeRectificationAttached.fits", dict()), # 2.122um, 0.1" slit.-//-
    # ("N20200818S0119_pinholeRectificationAttached.fits", dict()), # 2.182um, 0.1" slit. -//-
    #("N20200818S0133_pinholeRectificationAttached.fits", dict()), # 2.242um, 0.1" slit. -//-
    # ("N20220816S0388_pinholeRectificationAttached.fits", dict()), # 2.302um, 0.1" slit -//-
    # ("N20220816S0402_pinholeRectificationAttached.fits", dict()), # 2.362um, 0.1" slit -//-
    # ("N20220816S0416_pinholeRectificationAttached.fits", dict()), # 2.422um, 0.1" slit -//-
    # ("N20220816S0430_pinholeRectificationAttached.fits", dict()), # 2.482um, 0.1" slit -//-

    # Absorption
    # 10 l/mm, LongCam, LXD
    #("N20190613S0140_aperturesFound.fits", dict(absorption=True,resolution=1000)), # 1.650um, 0.45" slit. Ord 6,7. Actual resolution is much higher because it's AO (R~1000, not 350).
    # 32 l/mm ShortCam, SXD
    ("N20201229S0018_aperturesFound.fits", dict(absorption=True, num_lines=100)), # 2.150um, 0.3" slit. Ord 3-7: ok; ord 8 - no solution, low signal.
    #("N20210228S0106_aperturesFound.fits", dict(absorption=True)), # 1.65um, 0.68" slit. Ord 3-6 ok, ord 7,8 - misident
    # 32 l/mm LongCam, LXD
    #("N20201223S0034_aperturesFound.fits", dict(absorption=True)), # 1.98um, 0.45" slit. Ord. 3,4 - ok, 5, 6 - misident, low snr
    #("N20201222S0173_aperturesFound.fits", dict(absorption=True)), # 1.58um, 0.45" slit. Mostly misidentifications
    # 111 l/mm ShortCam, SXD
    #("N20160907S0297_aperturesFound.fits", dict(absorption=True)), # 1.53um, 0.3" slit. Ord. 6 - low snr, misident
    ("N20240821S0305_aperturesFound.fits", dict(absorption=True, min_snr=5)), # 1.65um, 0.3" slit. Ord. 3-7 ok, 8 - low snr, no solution
    #("N20140803S0060_aperturesFound.fits", dict()), # 1.77um, 0.3" slit. Ord. 3-7 ok with min_snr=10, num_lines=100. Ord 8 - misident.
    # 111 l/mm LongCam, LXD
    ("N20130630S0305_aperturesFound.fits", dict(absorption=True)), # 1.942um, 0.1" slit. All ord. ok.
    #("N20130630S0309_aperturesFound.fits", dict(absorption=True)), # 2.002um, 0.1" slit. Order 6 - misident, low snr.
    #("N20130630S0313_aperturesFound.fits", dict(absorption=True)), # 2.062um, 0.1" slit. Only order 3, 4 are fine, the rest is low snr, misident
    #("N20200818S0030_aperturesFound.fits", dict(absorption=True)), # 2.122um, 0.1" slit. Ord. 6-misident, low snr; rest ok.
    #("N20200818S0034_aperturesFound.fits", dict(absorption=True, min_snr=10)), # 2.182um, 0.1" slit. Ord 7 - misident.
    ("N20200818S0038_aperturesFound.fits", dict(absorption=True)), # 2.242um, 0.1" slit; All good
    #("N20220816S0472_aperturesFound.fits", dict(absorption=True)), # 2.302um, 0.1" slit. Ord. 5 - difficult, blanket absorption, misident.
    #("N20220816S0476_aperturesFound.fits", dict(absorption=True)), # 2.362um, 0.1" slit -//-, ord. 7 - no lines, misident
    #("N20220816S0480_aperturesFound.fits", dict(absorption=True)), # 2.422um, 0.1" slit; one order with no arc lines -- WRONG SOLUTION!
    #("N20220816S0484_aperturesFound.fits", dict(absorption=True)), # 2.482um, 0.1" slit, ord 4 - difficult, blank absoprion, misident, or. 7 - misident, no lines.
]
associated_calibrations = {
    "N20190613S0147.fits": {
        'pinholes': ["N20190605S0119.fits"],
        'flats': ["N20190613S0157.fits",
                  "N20190613S0165.fits"],
        'arcs': ["N20190613S0173.fits"],
        'std':  ["N20190613S0140.fits"],
    },
    "N20201229S0029.fits": {
        'pinholes': ["N20201225S0290.fits"],
        'flats': ["N20201229S0047.fits",
                  "N20201229S0057.fits"],
        'arcs': ["N20201229S0067.fits"],
        'std':  ["N20201229S0018.fits"],
    },
    "N20210228S0111.fits": {
        'pinholes': ["N20210228S0167.fits"],
        'flats': ["N20210228S0133.fits",
                  "N20210228S0139.fits"],
        'arcs': ["N20210228S0149.fits"],
        'std':  ["N20210228S0106.fits"],
    },
    "N20201223S0043.fits": {
        'pinholes': ["N20201223S0105.fits"],
        'flats': ["N20201224S0190.fits",
                  "N20201224S0193.fits"],
        'arcs': ["N20201224S0198.fits"],
        'std':  ["N20201223S0034.fits"],
    },
    "N20201222S0188.fits": {
        'pinholes': ["N20201222S0320.fits"],
        'flats': ["N20201222S0214.fits",
                  "N20201222S0217.fits"],
        'arcs': ["N20201222S0222.fits"],
        'std':  ["N20201222S0173.fits"],
    },

    "N20160907S0320.fits": {
        'pinholes': ["N20160907S0468.fits"],
        'flats': ["N20160907S0323.fits",
                  "N20160907S0327.fits"],
        'arcs': ["N20160907S0321.fits"],
        'std': ["N20160907S0297.fits"]
    },

    "N20240821S0257.fits": {
        'pinholes': ["N20240821S0346.fits"],
        'flats': ["N20240821S0261.fits",
                  "N20240821S0269.fits"],
        'arcs': ["N20240821S0271.fits"],
        'std': ["N20240821S0305.fits"]
    },

    "N20140803S0125.fits": {
        'pinholes': ["N20140803S0288.fits"],
        'flats': ["N20140803S0092.fits",
                  "N20140803S0100.fits"],
        'arcs': ["N20140803S0102.fits"],
        'std': ["N20140803S0060.fits"]
    },

    "N20130630S0158.fits": {
        'pinholes': ["N20130630S0457.fits"],
        'flats': ["N20130630S0163.fits"],
        'arcs': ["N20130630S0467.fits"],
        'std':  ["N20130630S0305.fits"],
    },
    "N20130630S0173.fits": {
        'pinholes': ["N20130630S0458.fits"],
        'flats': ["N20130630S0177.fits"],
        'arcs': ["N20130630S0468.fits"],
        'std':  ["N20130630S0309.fits"],
    },
    "N20130630S0187.fits": {
        'pinholes': ["N20130630S0459.fits"],
        'flats': ["N20130630S0191.fits"],
        'arcs': ["N20130630S0469.fits"],
        'std':  ["N20130630S0313.fits"],
    },
    "N20200818S0104.fits": {
        'pinholes': ["N20200818S0358.fits"],
        'flats': ["N20200818S0108.fits"],
        'arcs': ["N20200818S0348.fits"],
        'std':  ["N20200818S0030.fits"],
    },
    "N20200818S0119.fits": {
        'pinholes': ["N20200818S0359.fits"],
        'flats': ["N20200818S0123.fits"],
        'arcs': ["N20200818S0349.fits"],
        'std':  ["N20200818S0034.fits"],
    },
    "N20200818S0133.fits": {
        'pinholes': ["N20200818S0360.fits"],
        'flats': ["N20200818S0137.fits"],
        'arcs': ["N20200818S0350.fits"],
        'std':  ["N20200818S0038.fits"],
    },
    "N20220816S0388.fits": {
        'pinholes': ["N20220816S0680.fits"],
        'flats': ["N20220816S0392.fits"],
        'arcs': ["N20220816S0702.fits"],
        'std':  ["N20220816S0472.fits"],
    },
    "N20220816S0402.fits": {
        'pinholes': ["N20220816S0681.fits"],
        'flats': ["N20220816S0406.fits"],
        'arcs': ["N20220816S0703.fits"],
        'std':  ["N20220816S0476.fits"],
    },
    "N20220816S0416.fits": {
        'pinholes': ["N20220816S0682.fits"],
        'flats': ["N20220816S0420.fits"],
        'arcs': ["N20220816S0704.fits"],
        'std':  ["N20220816S0480.fits"],
    },
    "N20220816S0430.fits": {
        'pinholes': ["N20220816S0683.fits"],
        'flats': ["N20220816S0434.fits"],
        'arcs': ["N20220816S0705.fits"],
        'std':  ["N20220816S0484.fits"],
    }
}

# Tests Definitions ------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.wavecal
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
        p = GNIRSCrossDispersed([ad])
        pixel_scale = ad[0].pixel_scale()  # arcsec / px
        p.viewer = geminidr.dormantViewer(p, None)

        p.determineWavelengthSolution(**{**determine_wavelength_solution_parameters,
                                         **params})

        wcalibrated_ad = p.streams["main"][0]

        for record in caplog.records:
            if record.levelname == "WARNING":
                assert "No acceptable wavelength solution found" not in record.message

    ref_ad = astrodata.open(os.path.join(path_to_refs, wcalibrated_ad.filename))
    for wcalibrated_ext, ref_ext in zip(wcalibrated_ad, ref_ad):

        model = am.get_named_submodel(wcalibrated_ext.wcs.forward_transform, "WAVE")
        ref_model = am.get_named_submodel(ref_ext.wcs.forward_transform, "WAVE")

        x = np.arange(wcalibrated_ext.shape[1])
        wavelength = model(x)
        ref_wavelength = ref_model(x)

        slit_size_in_arcsec = float(wcalibrated_ext.focal_plane_mask(pretty=True).replace('arcsec', '').replace("XD",""))
        slit_size_in_px = slit_size_in_arcsec / pixel_scale
        dispersion = abs(wcalibrated_ext.dispersion(asNanometers=True))  # nm / px

        # We don't care about what the wavelength solution is doing at
        # wavelengths outside where we've matched lines
        lines = ref_ext.WAVECAL["wavelengths"].data
        lines = lines[lines > 0]  # column is padded with zeros

        if lines.size == 0:
            continue

        indices = np.where(np.logical_and(ref_wavelength > lines.min(),
                                          ref_wavelength < lines.max()))
        tolerance = 0.5 * (slit_size_in_px * dispersion)

        write_report = request.config.getoption('--do-report', False)
        failed = False
        try:
            np.testing.assert_allclose(wavelength[indices], ref_wavelength[indices],
                                   atol=tolerance)
            print(f"Test passed for {ad.filename}, extension {wcalibrated_ext.id}")
        except AssertionError:
            failed = True
            raise
        finally:
            if write_report:
                do_report(wcalibrated_ext, ref_ext, failed=failed)

        if request.config.getoption("--do-plots"):
            do_plots(wcalibrated_ext)


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
                  "test_gnirs_spect_xd_determine_wavelength_solution")
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grism = ad.disperser(pretty=True)
    filter = ad.filter_name(pretty=True)
    camera = ad.camera(pretty=True)

    central_wavelength = ad.central_wavelength(asNanometers=True)

    p = GNIRSCrossDispersed([ad])
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
    import os
    from astrodata.testing import download_from_archive
    from geminidr.gnirs.tests.crossdispersed import CREATED_INPUTS_PATH_FOR_TESTS
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals, set_btypes


    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    # for filename, _ in input_pars:
    #     print('Downloading files...')
    #     basename = filename.split("_")[0] + ".fits"
    #     sci_path = download_from_archive(basename)
    #     sci_ad = astrodata.open(sci_path)
    #     data_label = sci_ad.data_label()
    #
    #     print('Reducing pre-processed data:')
    #     logutils.config(file_name='log_{}.txt'.format(data_label))
    #     p = GNIRSCrossDispersed([sci_ad])
    #     p.prepare(bad_wcs="fix")
    #     p.addDQ()
    #     p.addVAR(read_noise=True)
    #     p.ADUToElectrons()
    #     p.addVAR(poisson_noise=True)
    #     #  p.flatCorrect()
    #     p.makeIRAFCompatible()
    #
    #     os.chdir("inputs/")
    #     processed_ad = p.writeOutputs().pop()
    #     os.chdir("../")
    #     print('Wrote pre-processed file to:\n'
    #           '    {:s}'.format(processed_ad.filename))

    for filename, cals in associated_calibrations.items():
        print(filename)

        arc_path =[ download_from_archive(f) for f in cals['arcs']]
        flat_paths = [download_from_archive(f) for f in cals['flats']]
        pinholes_path = [download_from_archive(f) for f in cals['pinholes']]
        std_path = [download_from_archive(f) for f in cals['std']]
        sci_path = download_from_archive(filename)
        arc_ad = astrodata.open(arc_path[0])
        std_ad = astrodata.open(std_path[0])
        sci_ad = astrodata.open(sci_path)
        data_label = arc_ad.data_label()

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_paths)
        flat_reduce.uparms = [('normalizeFlat:threshold','0.0001')]
        flat_reduce.runr()
        processed_flat = flat_reduce.output_filenames.pop()
        flat_cal = ['processed_flat:{}'.format(processed_flat)]
        del flat_reduce

        logutils.config(file_name='log_pinholes_{}.txt'.format(data_label))
        pinholes_reduce = Reduce()
        pinholes_reduce.files.extend(pinholes_path)
        pinholes_reduce.ucals = normalize_ucals(flat_cal)
        if  "N20130630S0457" or "N20220816S0682" in pinholes_path[0]:
            pinholes_reduce.uparms = [('determinePinholeRectification:min_snr', '2')]
        pinholes_reduce.runr()
        processed_pinholes = pinholes_reduce.output_filenames.pop()
        pinhole_cal = ['processed_pinhole:{}'.format(processed_pinholes)]

        del pinholes_reduce

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend(arc_path)
        arc_reduce.ucals = normalize_ucals(flat_cal + pinhole_cal)
        if "N20201224S0198" in arc_path[0]:
            arc_reduce.uparms = [('determineWavelengthSolution:min_snr', '5')]
        if "N20240821S0271" in arc_path[0]:
            arc_reduce.uparms = [('determineWavelengthSolution:order', '1')]
        arc_reduce.runr()
        processed_arc =arc_reduce.output_filenames.pop()
        #arc_cal = ['processed_arc:{}'.format(processed_arc)]

        del arc_reduce

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        p = GNIRSCrossDispersed([arc_ad])
        p.prepare(bad_wcs="new")
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.applySlitModel(flat=processed_flat, suffix="_slitModelApplied")
        p.attachPinholeRectification(pinhole=processed_pinholes, suffix="_pinholeRectificationAttached")

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_ad.filename))

        p = GNIRSCrossDispersed([sci_ad])
        p.prepare(bad_wcs="new")
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.nonlinearityCorrect()
        p.flatCorrect(flat=processed_flat, suffix="_flatCorrected")
        p.attachPinholeRectification(pinhole=processed_pinholes, suffix="_pinholeRectificationAttached")

        os.chdir("inputs/")
        processed_sci = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_sci.filename))

        p = GNIRSCrossDispersed([std_ad])
        p.prepare(bad_wcs="new")
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.nonlinearityCorrect()
        p.flatCorrect(flat=processed_flat, suffix="_flatCorrected")
        p.attachWavelengthSolution(arc=processed_arc, suffix="_wavelengthSolutionAttached")
        p.copyInputs(instream="main", outstream="with_distortion_model")
        p.separateSky()
        p.associateSky()
        p.skyCorrect()
        p.attachPinholeRectification(pinhole=processed_pinholes, suffix="_pinholeRectificationAttached")
        p.distortionCorrect()
        p.adjustWCSToReference()
        p.resampleToCommonFrame(output_wave_scale='reference', trim_spectral=True)
        p.findApertures()

        os.chdir("inputs/")
        processed_std = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_std.filename))


def create_refs_recipe():
    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("refs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, params in input_pars:
        ad = astrodata.open(os.path.join('inputs', filename))
        print(f"Reducing {ad.filename}")
        p = GNIRSCrossDispersed([ad])
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



