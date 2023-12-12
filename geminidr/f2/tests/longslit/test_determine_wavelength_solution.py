#!/usr/bin/env python
"""
Tests related to F2 Long-slit Spectroscopy Arc primitives.

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

from geminidr.f2.primitives_f2_longslit import F2Longslit
from gempy.library import astromodels as am
from gempy.utils import logutils

import os
from geminidr.f2.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS

# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'global',
    'fwidth': None,
    'order': 3,
    'min_snr': 10,
    'debug_min_lines': 100000,
    'in_vacuo': True
}

input_pars = [
    # Process Arcs: F2 ---
    # (Input File, params)
    # If a grism+filter combination is not present here => no science data exists
    # grism: JH, filter: JH (1.400um)
    ("S20160928S0077_flatCorrected.fits", dict()), # 8-pix slit
    ("S20211216S0177_flatCorrected.fits", dict()), # 2-pix slit
    ("S20220525S0036_flatCorrected.fits", dict()), # 6-pix slit, new JH filter
    # grism: HK, filter: HK (1.870um)
    ("S20220319S0060_flatCorrected.fits", dict()), # 3-pix slit, new HK filter
    ("S20220527S0035_flatCorrected.fits", dict()), # 4-pix slit, new HK filter, ND4-5
    ("S20130930S0119_flatCorrected.fits", dict()), # 2-pix slit # no associated flat (the program one is not matching, , probably because of diff read modes)
    ("S20160215S0138_flatCorrected.fits", dict()), # 6-pix slit, ND4-5, all flats saturated
    # grism: HK, filter: JH (1.100um)
    ("S20220101S0044_flatCorrected.fits", dict()), # 3-pix slit
    # grism: R3K, filter: J-low (1.100um)
    ("S20131018S0230_flatCorrected.fits", dict()), # 6-pix slit
    # grism: R3K, filter: J (1.250um)
    ("S20170715S0121_flatCorrected.fits", dict()), # 1-pix slit, Clear
    ("S20200219S0105_flatCorrected.fits", dict()), # 4-pix slit, ND1.0
    # grism: R3K, filter: H (1.650um)
    ("S20210903S0053_flatCorrected.fits", dict()), # 6-pix slit
    ("S20131227S0114_flatCorrected.fits", dict()), # 2-pix slit # no associated flat (the program one is not matching, probably because of diff read modes)
    # grism: R3K, filter: Ks (2.200um)
    ("S20140220S0425_flatCorrected.fits", dict()), # 2-pix slit
    ("S20220515S0026_flatCorrected.fits", dict()), # 3-pix slit
    # grism: R3K, filter: K-long (2.200um)
    ("S20150624S0023_flatCorrected.fits", dict()), # 1-pix slit
    ("S20211018S0011_flatCorrected.fits", dict()), # 6-pix slit
    # wavecal from OH emission sky lines
    ("S20180114S0104_flatCorrected.fits", dict()), # HK, science
    ("S20140216S0079_flatCorrected.fits", dict()), # R3K, science
]

associated_calibrations = {
    "S20160928S0077.fits": {
        'arc_darks': ["S20161001S0378.fits",
                 "S20161001S0379.fits",
                 "S20161001S0380.fits",
                 "S20161001S0381.fits",
                 "S20161001S0382.fits"],
        'flat': ["S20160928S0078.fits"], # ND2.0
        'flat_darks': ["S20161008S0141.fits",
                 "S20161008S0142.fits",
                 "S20161008S0143.fits",
                 "S20161008S0144.fits",
                 "S20161008S0145.fits"],
    },
    "S20211216S0177.fits": {
        'arc_darks': ["S20211218S0462.fits",
                 "S20211218S0463.fits",
                 "S20211218S0464.fits",
                 "S20211218S0465.fits",
                 "S20211218S0466.fits"],
        'flat': ["S20211216S0176.fits"], # ND2.0
        'flat_darks': ["S20211218S0434.fits",
                 "S20211218S0435.fits",
                 "S20211218S0436.fits",
                 "S20211218S0437.fits",
                 "S20211218S0438.fits"],
    },
    "S20220525S0036.fits": {
        'arc_darks': ["S20220528S0154.fits",
                 "S20220528S0156.fits",
                 "S20220528S0157.fits",
                 "S20220528S0158.fits",
                 "S20220528S0160.fits"],
        'flat': ["S20220525S0037.fits"], # ND2.0
        'flat_darks': ["S20220417S0097.fits",
                 "S20220417S0099.fits",
                 "S20220417S0100.fits",
                 "S20220417S0101.fits",
                 "S20220417S0103.fits"],
    },
    "S20130930S0119.fits": {
        'arc_darks': ["S20130930S0156.fits",
                 "S20130930S0157.fits",
                 "S20130930S0158.fits",
                 "S20130930S0159.fits",
                 "S20130930S0160.fits"],
        'flat': ["S20130930S0120.fits"], # ND2.0
        'flat_darks': ["S20130930S0170.fits",
                 "S20130930S0171.fits",
                 "S20130930S0172.fits",
                 "S20130930S0173.fits",
                 "S20130930S0174.fits"],
    },
    "S20220527S0035.fits": {
        'arc_darks': ["S20220528S0188.fits",
                 "S20220528S0192.fits",
                 "S20220528S0195.fits",
                 "S20220528S0197.fits",
                 "S20220528S0200.fits"],
        'flat': ["S20220527S0072.fits"], #ND4-5 GCAL filter
        'flat_darks': ["S20220402S0329.fits",
                 "S20220402S0334.fits",
                 "S20220402S0340.fits",
                 "S20220402S0346.fits",
                 "S20220402S0347.fits"],
    },
    "S20220319S0060.fits": {
        'arc_darks': ["S20220322S0099.fits",
                 "S20220322S0100.fits",
                 "S20220322S0101.fits",
                 "S20220322S0102.fits",
                 "S20220322S0103.fits"],
        'flat': ["S20220319S0059.fits"], # ND2.0
        'flat_darks': ["S20220203S0025.fits",
                 "S20220203S0037.fits",
                 "S20220203S0055.fits",
                 "S20220203S0058.fits"],
    },
    "S20220101S0044.fits": {
        'arc_darks': ["S20220101S0108.fits",
                 "S20220101S0109.fits",
                 "S20220101S0111.fits",
                 "S20220101S0112.fits",
                 "S20220101S0114.fits"],
        'flat': ["S20220101S0043.fits"], # ND2.0
        'flat_darks': ["S20220101S0086.fits",
                 "S20220101S0087.fits",
                 "S20220101S0088.fits",
                 "S20220101S0090.fits",
                 "S20220101S0091.fits"],
    },
    "S20160215S0138.fits": {
        'arc_darks': ["S20160213S0413.fits",
                 "S20160213S0412.fits",
                 "S20160213S0411.fits",
                 "S20160213S0410.fits",
                 "S20160213S0409.fits"],
        'flat': ["S20160215S0139.fits"], # ND4-5, all available flats are saturated
        'flat_darks': ["S20160302S0214.fits",
                 "S20160302S0213.fits",
                 "S20160302S0212.fits",
                 "S20160302S0211.fits",
                 "S20160302S0210.fits"],
    },
    "S20131018S0230.fits": {
        'arc_darks': ["S20131023S0025.fits",
                 "S20131023S0026.fits",
                 "S20131023S0027.fits",
                 "S20131023S0028.fits",
                 "S20131023S0029.fits"],
        'flat': ["S20131018S0229.fits"], # Clear
        'flat_darks': ["S20131019S0270.fits",
                 "S20131019S0271.fits",
                 "S20131019S0272.fits",
                 "S20131019S0273.fits",
                 "S20131019S0274.fits"],
    },
    "S20170715S0121.fits": {
        'arc_darks': ["S20170715S0325.fits",
                 "S20170715S0326.fits",
                 "S20170715S0327.fits",
                 "S20170715S0328.fits",
                 "S20170715S0329.fits"],
        'flat': ["S20170715S0133.fits"], # Clear
        'flat_darks': ["S20170722S0219.fits",
                 "S20170722S0220.fits",
                 "S20170722S0221.fits",
                 "S20170722S0222.fits",
                 "S20170722S0223.fits"],
    },
    "S20200219S0105.fits": {
        'arc_darks': ["S20200223S0087.fits",
                 "S20200223S0088.fits",
                 "S20200223S0089.fits",
                 "S20200223S0090.fits",
                 "S20200223S0091.fits"],
        'flat': ["S20200219S0131.fits"], # ND1.0
        'flat_darks': ["S20200301S0223.fits",
                 "S20200301S0224.fits",
                 "S20200301S0225.fits",
                 "S20200301S0226.fits",
                 "S20200301S0227.fits"],
    },
    "S20210903S0053.fits": {
        'arc_darks': ["S20210905S0413.fits",
                 "S20210905S0414.fits",
                 "S20210905S0415.fits",
                 "S20210905S0416.fits",
                 "S20210905S0417.fits"],
        'flat': ["S20210903S0052.fits"], # ND2.0
        'flat_darks': ["S20210905S0392.fits",
                 "S20210905S0393.fits",
                 "S20210905S0394.fits",
                 "S20210905S0395.fits",
                 "S20210905S0396.fits"],
    },
    "S20140220S0425.fits": {
        'arc_darks': ["S20140220S0621.fits",
                 "S20140220S0622.fits",
                 "S20140220S0623.fits",
                 "S20140220S0624.fits",
                 "S20140220S0625.fits"],
        'flat': ["S20140220S0426.fits"], # ND2.0
        'flat_darks': ["S20140220S0615.fits",
                 "S20140220S0616.fits",
                 "S20140220S0617.fits",
                 "S20140220S0618.fits",
                 "S20140220S0619.fits"],
    },
    "S20131227S0114.fits": {
        'arc_darks': ["S20131227S0345.fits",
                 "S20131227S0346.fits",
                 "S20131227S0347.fits",
                 "S20131227S0348.fits",
                 "S20131227S0349.fits"],
        'flat': ["S20131227S0115.fits"], # ND2.0
        'flat_darks': ["S20131227S0352.fits",
                 "S20131227S0353.fits",
                 "S20131227S0354.fits",
                 "S20131227S0355.fits",
                 "S20131227S0356.fits"],
    },
    "S20220515S0026.fits": {
        'arc_darks': ["S20220521S0129.fits",
                 "S20220521S0130.fits",
                 "S20220521S0131.fits",
                 "S20220521S0132.fits",
                 "S20220521S0133.fits"],
        'flat': ["S20220515S0025.fits"], # ND2.0
        'flat_darks': ["S20220514S0281.fits",
                 "S20220514S0279.fits",
                 "S20220514S0277.fits",
                 "S20220514S0275.fits",
                 "S20220514S0274.fits"],
    },
    "S20150624S0023.fits": {
        'arc_darks': ["S20150523S0292.fits",
                 "S20150523S0291.fits",
                 "S20150523S0290.fits",
                 "S20150523S0289.fits",
                 "S20150523S0288.fits"],
        'flat': ["S20150624S0028.fits"], # ND2.0
        'flat_darks': ["S20150627S0337.fits",
                 "S20150627S0338.fits",
                 "S20150627S0339.fits",
                 "S20150627S0340.fits",
                 "S20150627S0341.fits"],
    },
    "S20211018S0011.fits": {
        'arc_darks': ["S20211023S0348.fits",
                 "S20211023S0349.fits",
                 "S20211023S0350.fits",
                 "S20211023S0351.fits",
                 "S20211023S0352.fits"],
        'flat': ["S20211018S0012.fits"], # ND2.0
        'flat_darks': ["S20210905S0350.fits",
                 "S20210905S0351.fits",
                 "S20210905S0352.fits",
                 "S20210905S0353.fits",
                 "S20210905S0354.fits"],
    },
    "S20180114S0104.fits": {
        'arc_darks': ["S20180117S0033.fits",
                      "S20180117S0034.fits",
                      "S20180117S0035.fits",
                      "S20180117S0036.fits",
                      "S20180117S0037.fits"],
        'flat': ["S20180114S0118.fits"], # ND2.0
        'flat_darks': ["S20180120S0222.fits",
                 "S20180120S0223.fits",
                 "S20180120S0224.fits",
                 "S20180120S0225.fits",
                 "S20180120S0226.fits"],
    },
    "S20140216S0079.fits": {
        'arc_darks': ["S20140216S0326.fits",
                      "S20140216S0327.fits",
                      "S20140216S0328.fits",
                      "S20140216S0329.fits",
                      "S20140216S0330.fits"],
        'flat': ["S20140216S0089.fits"], # ND2.0
        'flat_darks': ["S20140218S0043.fits",
                 "S20140218S0044.fits",
                 "S20140218S0045.fits",
                 "S20140218S0046.fits",
                 "S20140218S0047.fits"],
    },
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
        p = F2Longslit([ad])
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
    slit_size_in_px = wcalibrated_ad[0].slit_width() / pixel_scale
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
    isHK_JH = ad.disperser(pretty=True) == "HK" and \
                ad.filter_name(pretty=True) == "JH"
    if 'ARC' in ad.tags:
        linelist = 'argon.dat'
        if isHK_JH:
            linelist = 'lowresargon_with_2nd_ord.dat'
    else:
        linelist = 'nearIRsky.dat'
        if isHK_JH:
            linelist = 'nearIRsky_with_2nd_order.dat'

    return linelist

# add a fixture for this to work
def do_report(ad, ref_ad, failed):
    """
    Generate text file with test details.

    """
    output_dir = ("../DRAGONS_tests/geminidr/f2/longslit/"
                  "test_determine_wavelength_solution")
    os.makedirs(output_dir, exist_ok=True)
    report_filename = 'report.txt'
    report_path = os.path.join(output_dir, report_filename)

    ref_wavecal_model = am.get_named_submodel(ref_ad[0].wcs.forward_transform, "WAVE")
    wavecal_model = am.get_named_submodel(ad[0].wcs.forward_transform, "WAVE")
    domain = wavecal_model.domain
    dw = np.diff(wavecal_model(domain))[0] / np.diff(domain)[0]
    ref_dw = np.diff(ref_wavecal_model(domain))[0] / np.diff(domain)[0]
    nmatches = np.count_nonzero(ref_ad[0].WAVECAL['peaks'])
    ref_nmatches = np.count_nonzero(ref_ad[0].WAVECAL['peaks'])

    with open(report_path, 'a') as report_output:
        if os.lseek(report_output.fileno(), 0, os.SEEK_CUR) == 0:
            print("Filename matched_lines final_order cenwave_delta disp_delta",
                  file=report_output)
        if failed:
            print("Reference parameters:",
                  file=report_output)
            print(f"{ref_ad.filename} {ref_nmatches} {ref_wavecal_model.degree} "
                  f"{((ref_wavecal_model(511)-ref_ad[0].central_wavelength(asNanometers=True))):.1f} {(ref_dw-ref_ad[0].dispersion(asNanometers=True)):.3f}",
                  file=report_output)
            print("Failed test file parameters:",
                  file=report_output)
        print(f"{ad.filename} {nmatches} {wavecal_model.degree} "
                  f"{((wavecal_model(511)-ad[0].central_wavelength(asNanometers=True))):.1f} {(dw-ad[0].dispersion(asNanometers=True)):.3f}",
                  file=report_output)


def do_plots(ad):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : astrodata
    """

    output_dir = ("./plots/geminidr/f2/"
                  "test_f2_spect_ls_determine_wavelength_solution")
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grism = ad.disperser(pretty=True)
    filter = ad.filter_name(pretty=True)
    camera = ad.camera(pretty=True)

    central_wavelength = ad.central_wavelength(asNanometers=True)

    p = F2Longslit([ad])
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
    from geminidr.f2.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals, set_btypes

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():
        print(filename)

        arc_path = download_from_archive(filename)
        arc_darks_paths = [download_from_archive(f) for f in cals['arc_darks']]
        flat_darks_paths = [download_from_archive(f) for f in cals['flat_darks']]
        flat_path = [download_from_archive(f) for f in cals['flat']]

        arc_ad = astrodata.open(arc_path)
        data_label = arc_ad.data_label()

        logutils.config(file_name='log_arc_darks_{}.txt'.format(data_label))
        arc_darks_reduce = Reduce()
        arc_darks_reduce.files.extend(arc_darks_paths)
        arc_darks_reduce.runr()
        arc_darks_master = arc_darks_reduce.output_filenames.pop()
        del arc_darks_reduce

        logutils.config(file_name='log_flat_darks_{}.txt'.format(data_label))
        flat_darks_reduce = Reduce()
        flat_darks_reduce.files.extend(flat_darks_paths)
        flat_darks_reduce.runr()
        flat_darks_master = flat_darks_reduce.output_filenames.pop()
        calibration_files = ['processed_dark:{}'.format(flat_darks_master)]
        del flat_darks_reduce

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_path)
        flat_reduce.ucals = normalize_ucals(calibration_files)
        flat_reduce.uparms = [('normalizeFlat:threshold','0.01')]
        flat_reduce.runr()
        processed_flat = flat_reduce.output_filenames.pop()
        del flat_reduce

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        p = F2Longslit([arc_ad])
        p.prepare(bad_wcs="fix")
        p.addDQ()
        p.addVAR(read_noise=True)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.darkCorrect(dark=arc_darks_master)
        p.flatCorrect(flat=processed_flat, suffix="_flatCorrected")
        p.makeIRAFCompatible()

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
        p = F2Longslit([ad])
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
