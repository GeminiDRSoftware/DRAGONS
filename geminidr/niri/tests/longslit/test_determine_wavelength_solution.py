#!/usr/bin/env python
"""
Tests related to NIRI Long-slit Spectroscopy Arc primitives.

"""
import glob
import tarfile
import logging

import numpy as np
import pytest
from matplotlib import pyplot as plt
from importlib import import_module

import astrodata
import geminidr

from geminidr.niri.primitives_niri_longslit import NIRILongslit
from gempy.library import astromodels as am
from gempy.utils import logutils

import os
from geminidr.niri.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS

# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'global',
    'fwidth': None,
    'order': 3,
    'min_snr': None,
    'debug_min_lines': 15,
    'in_vacuo': True,
    'num_atran_lines': 100,
    "combine_method": "optimal",
    "wv_band": "header",
    "resolution": None
}

input_pars = [
    # Process Arcs: NIRI ---
    # (Input File, params)
    # camera: f32, grism: J
    ("N20041025S0073_flatCorrected.fits", dict()), # f32-9pix
    # camera: f32, grism: H
    ("N20090925S0304_flatCorrected.fits", dict()), # f6-2pix
    ("N20081219S0530_flatCorrected.fits", dict()), # f32-9pix
    # camera: f32, grism: K
    ("N20080530S0292_flatCorrected.fits", dict()), # f32-9pix
    # camera: f6, grism: J
    ("N20100620S0126_flatCorrected.fits", dict()), # f6-6pix
    ("N20070404S0006_flatCorrected.fits", dict()), # f6-2pixBl
    # camera: f6, grism: H
    #("N20100616S0840_flatCorrected.fits", dict()), # f6-6pix. The solution is unstable due to close lines
   # ("N20070627S0025_flatCorrected.fits", dict()), # f6-2pixBl. The solution is unstable due to close lines
    # camera: f6, grism: K
    ("N20100619S0594_flatCorrected.fits", dict(min_snr=30)), # f6-2pix, With default min_snr the solution is unstable due to close lines
    ("N20060707S0283_flatCorrected.fits", dict()), # f6-4pixBl
    # camera: f6, grism: L
    ("N20100215S0128_flatCorrected.fits", dict()), # f6-6pix, science, linelists created on-the fly
    ("N20060206S0130_flatCorrected.fits", dict()), # f6-2pixBl, science, linelists created on-the fly
    # camera: f6, grism: M
    ("N20091022S0369_flatCorrected.fits", dict(min_snr=20)), # f6-2pix, science, linelists created on-the fly. With default min_snr the solution is unstable due to faint lines
    # absorption test
    ("N20090706S0727_aperturesFound.fits", dict(absorption=True)), # science, linelists created on-the fly
    # OH emission test
    ("N20090706S0706_flatCorrected.fits", dict()) # science, linelists created on-the fly
]

associated_calibrations = {
    "N20041025S0073.fits": {
        'flat': ["N20041025S0074.fits"],
    },
    "N20090925S0304.fits": {
        'flat': ["N20090925S0305.fits"],
    },
    "N20081219S0530.fits": {
        'flat': ["N20081219S0529.fits"],
    },
    "N20080530S0292.fits": {
        'flat': ["N20080530S0291.fits"],
    },
    "N20100620S0126.fits": {
        'flat': ["N20100620S0125.fits"],
    },
    "N20070404S0006.fits": {
        'flat': ["N20070404S0005.fits"],
    },
    "N20100616S0840.fits": {
        'flat': ["N20100616S0841.fits"],
    },
    "N20070627S0025.fits": {
        'flat': ["N20070627S0024.fits"],
    },
    "N20100619S0594.fits": {
        'flat': ["N20100619S0595.fits"],
    },
    "N20060707S0283.fits": {
        'flat': ["N20060707S0282.fits"],
    },
    "N20100215S0128.fits": {
        'flat': ["N20100215S0131.fits"],
    },
    "N20060206S0130.fits": {
        'flat': ["N20060206S0140.fits"],
    },
    "N20091022S0369.fits": {
        'flat': ["N20091022S0371.fits"], # doesn't find flat, no rect model in flat
    },
    # "N20090706S0727.fits": {
    #     'flat': ["N20090706S0676.fits"],
    #     'arc': ["N20090706S0666.fits"], # make reference manually
    # },
    "N20090706S0706.fits": {
        'flat': ["N20090706S0676.fits"],
    }
}

# Tests Definitions ------------------------------------------------------------
@pytest.mark.skip
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
        p = NIRILongslit([ad])
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
    if 'ARC' in ad.tags:
        if 'Xe' in ad.object():
            linelist ='Ar_Xe.dat'
        elif "Ar" in ad.object():
            linelist = 'argon.dat'
        else:
            raise ValueError(f"No default line list found for {ad.object()}-type arc. Please provide a line list.")
    else:
        linelist = 'nearIRsky.dat'

    return linelist

# add a fixture for this to work
def do_report(ad, ref_ad, failed):
    """
    Generate text file with test details.

    """
    output_dir = ("../DRAGONS_tests/geminidr/niri/longslit/"
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

    output_dir = ("./plots/geminidr/niri/"
                  "test_niri_spect_ls_determine_wavelength_solution")
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grism = ad.disperser(pretty=True)
    filter = ad.filter_name(pretty=True)
    camera = ad.camera(pretty=True)

    central_wavelength = ad.central_wavelength(asNanometers=True)

    p = NIRILongslit([ad])
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
    from geminidr.niri.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS
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

        p = NIRILongslit([arc_ad])
        p.prepare()
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.nonlinearityCorrect()
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
        p = NIRILongslit([ad])
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
