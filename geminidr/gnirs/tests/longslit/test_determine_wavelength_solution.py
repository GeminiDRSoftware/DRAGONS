#!/usr/bin/env python
"""
Tests related to GNIRS Long-slit Spectroscopy Arc primitives. `input_files` is a
list of tuples which contains:

 - the input filename,
 - the full-width-at-half-maximum measured using IRAF's splot,
 - the wavelength solution order guessed based on residuals (usually between 2 and 4),
 - the minimum signal to noise for detection, based on splot analysis.

The input data can be cached from the archive and/or processed using the
--force-preprocess-data command line option.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.

  @bquint:
    It seems that the matching process depends heavily on the FWHM. Because of
    that, the FWHM was measured using `splot` (keys h, c) manually for each
    file. It basically tells how far the KDTreeFitter should look for a match.

    The fitting order was picked up after running the test and analysing the
    shape of the residuals.

    Finally, the min_snr was semi-arbitrary. It had an opposite effect from
    what I, expected. Sometimes, raising this number caused more peaks to be
    detected.

"""
import glob
import os
import tarfile
import logging
from copy import deepcopy

import numpy as np
import pytest
from matplotlib import pyplot as plt
from astropy import units as u
from specutils.utils.wcs_utils import air_to_vac

import astrodata
import geminidr

from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from gempy.library import astromodels as am
from gempy.utils import logutils


# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'global',
    'fwidth': None,
    'order': None,
    'min_snr': 20,
    'nbright': 0,
    'debug_min_lines': 100000,
    'in_vacuo': True
}

input_pars = [
    # Process Arcs: GNIRS ---
    # (Input File, params)
    # If a grating+camera+site combination is not present here -> no data exists
    # 10 l/mm, LongCam, GN
    ("N20210810S0833_varAdded.fits", dict()), # X, 1.100um, 0.1" slit
    ("N20210810S0835_varAdded.fits", dict()), # J, 1.250um, 0.1" slit
    ("N20140713S0193_varAdded.fits", dict()), # H, 1.650um, 0.1" slit, AO
    ("N20210803S0133_varAdded.fits", dict()), # K, 2.220um, 0.1" slit, AO
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
    ("N20210810S0841_varAdded.fits", dict()), # X, 1.100um, 0.1" slit
    ("N20210810S0843_varAdded.fits", dict()), # J, 1.250um, 0.1" slit
    ("N20210810S0845_varAdded.fits", dict()), # H, 1.650um, 0.1" slit
    ("N20191221S0157_varAdded.fits", dict()), # K, 2.350um, 0.1" slit
    ("N20210810S0847_varAdded.fits", dict()), # K, 2.200um, 0.1" slit
    # 111 l/mm, ShortCam, GN
    ("N20210722S0332_varAdded.fits", dict()), # X, 1.083um, 0.3" slit
    ("N20220222S0151_varAdded.fits", dict()), # X, 1.120um, 1" slit
    ("N20220222S0199_varAdded.fits", dict()), # X, 1.160um, 1" slit
    ("N20130808S0131_varAdded.fits", dict()), # J, 1.180um, 0.3" slit
    ("N20220222S0170_varAdded.fits", dict()), # J, 1.210um, 1" slit
    ("N20101203S0888_varAdded.fits", dict()), # J, 1.250um, 0.45" slit
    ("N20110923S0478_varAdded.fits", dict()), # J, 1.340um, 1" slit
    ("N20130808S0145_varAdded.fits", dict()), # H, 1.591um, 0.3" slit
    ("N20210613S0358_varAdded.fits", dict()), # H, 1.740um, 1" slit
    ("N20170429S0265_varAdded.fits", dict()), # K, 2.020um, 0.45" slit
    ("N20220303S0271_varAdded.fits", dict()), # K, 2.100um, 0.3" slit
    ("N20160618S0255_varAdded.fits", dict()), # K, 2.200um, 0.45" slit
    ("N20210615S0171_varAdded.fits", dict()), # K, 2.330um, 0.3" slit
    # 111 l/mm, ShortCam, GS
    ("S20060214S0406_varAdded.fits", dict()), # X, 1.083um, 0.3" slit
    ("S20051109S0095_varAdded.fits", dict()), # J, 1.207um, 0.675" slit
    ("S20040614S0367_varAdded.fits", dict()), # H, 1.475um, 0.3" slit
    ("S20050927S0054_varAdded.fits", dict()), # K, 2.360um, 0.3" slit
    ("S20040904S0321_varAdded.fits", dict()), # K, 2.400um, 0.675" slit
    # 111 l/mm, LongCam, GN
    ("N20111018S1547_varAdded.fits", dict()), # X, 1.050um, 0.2" slit
    ("N20121221S0204_varAdded.fits", dict()), # X, 1.083um, 0.1" slit
    ("N20210810S0865_varAdded.fits", dict()), # X, 1.100um, 0.1" slit
    ("N20111018S1549_varAdded.fits", dict()), # X, 1.110um, 0.2" slit
    ("N20111019S1026_varAdded.fits", dict()), # X, 1.140um, 0.68" slit
    ("N20150602S0501_varAdded.fits", dict()), # J, 1.180um, 0.1" slit
    ("N20100823S1864_varAdded.fits", dict()), # J, 1.230um, 0.3" slit
    ("N20100815S0237_varAdded.fits", dict()), # J, 1.270um, 0.1" slit
    ("N20200914S0035_varAdded.fits", dict()), # J, 1.300um, 0.15" slit
    ("N20111020S0664_varAdded.fits", dict()), # J, 1.340um, 0.45" slit
    ("N20110727S0332_varAdded.fits", dict()), # H, 1.520um, 0.1" slit
    ("N20110601S0753_varAdded.fits", dict()), # H, 1.565um, 0.1" slit
    ("N20180605S0147_varAdded.fits", dict()), # H, 1.600um, 0.1" slit
    ("N20101206S0770_varAdded.fits", dict()), # H, 1.644um, 0.1" slit
    ("N20140801S0183_varAdded.fits", dict()), # H, 1.690um, 0.1" slit
    ("N20141218S0289_varAdded.fits", dict()), # H, 1.710um, 0.1" slit
    ("N20111019S1056_varAdded.fits", dict()), # H, 1.750um, 0.45" slit
    ("N20130624S0196_varAdded.fits", dict()), # H, 1.778um, 0.1" slit
    ("N20111011S0564_varAdded.fits", dict()), # K, 1.940um, 1" slit
    ("N20110531S0510_varAdded.fits", dict()), # K, 2.030um, 0.1" slit
    ("N20110715S0523_varAdded.fits", dict()), # K, 2.090um, 0.1" slit
    ("N20160109S0118_varAdded.fits", dict()), # K, 2.115um, 0.15" slit
    ("N20150613S0087_varAdded.fits", dict()), # K, 2.166um, 0.1" slit
    ("N20110618S0031_varAdded.fits", dict()), # K, 2.208um, 0.3" slit
    ("N20110923S0569_varAdded.fits", dict()), # K, 2.260um, 0.1" slit
    ("N20201022S0051_varAdded.fits", dict()), # K, 2.310um, 0.1" slit
    ("N20160804S0155_varAdded.fits", dict()), # K, 2.350um, 0.1" slit
    ("N20100823S1874_varAdded.fits", dict()), # K, 2.400um, 0.3" slit
    ("N20121218S0198_varAdded.fits", dict()), # K, 2.420um, 0.1" slit
    ("N20161217S0001_varAdded.fits", dict()), # K, 2.500um, 0.1" slit
    # 111 l/mm, LongCam, GS
    ("S20040907S0141_varAdded.fits", dict()), # J, 1.200um, 0.1" slit
    ("S20040907S0129_varAdded.fits", dict()), # H, 1.600um, 0.1" slit
    ("S20050119S0122_varAdded.fits", dict()), # K, 1.975um, 0.1" slit
    ("S20061227S0095_varAdded.fits", dict()), # K, 2.120um, 0.1" slit
    ("S20050119S0117_varAdded.fits", dict()), # K, 2.170um, 0.1" slit
    ("S20060716S0045_varAdded.fits", dict()), # K, 2.190um, 0.1" slit
    ("S20050119S0122_varAdded.fits", dict()), # K, 2.300um, 0.1" slit
]

# Tests Definitions ------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gmosls
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
    slit_size_in_arcsec = float(wcalibrated_ad[0].focal_plane_mask().replace('arcsec', ''))
    slit_size_in_px = slit_size_in_arcsec / pixel_scale
    dispersion = abs(wcalibrated_ad[0].dispersion(asNanometers=True))  # nm / px

    # We don't care about what the wavelength solution is doing at
    # wavelengths outside where we've matched lines
    lines = ref_ad[0].WAVECAL["wavelengths"].data
    indices = np.where(np.logical_and(ref_wavelength > lines.min(),
                                      ref_wavelength < lines.max()))
    tolerance = 0.5 * (slit_size_in_px * dispersion)
    np.testing.assert_allclose(wavelength[indices], ref_wavelength[indices],
                               atol=tolerance)

    if request.config.getoption("--do-plots"):
        do_plots(wcalibrated_ad)


# We only need to test this with one input
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, params", input_pars[:1], indirect=['ad'])
def test_consistent_air_and_vacuum_solutions(ad, params):
    p = GNIRSLongslit([])
    p.viewer = geminidr.dormantViewer(p, None)

    new_params = {**determine_wavelength_solution_parameters, **params}
    ad_air = p.determineWavelengthSolution(
        [deepcopy(ad)], **new_params, in_vacuo=False).pop()
    ad_vac = p.determineWavelengthSolution(
        [ad], **new_params, in_vacuo=True).pop()
    wave_air = am.get_named_submodel(ad_air[0].wcs.forward_transform, "WAVE")
    wave_vac = am.get_named_submodel(ad_vac[0].wcs.forward_transform, "WAVE")
    x = np.arange(ad_air[0].shape[1])
    wair = wave_air(x)
    wvac = air_to_vac(wair * u.nm).to(u.nm).value
    dw = wvac - wave_vac(x)
    assert abs(dw).max() < 0.001


# We only need to test this with one input
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, params", input_pars[:1], indirect=['ad'])
@pytest.mark.parametrize("in_vacuo", (True, False))
def test_user_defined_linelist(ad, params, in_vacuo):
    p = GNIRSLongslit([])
    p.viewer = geminidr.dormantViewer(p, None)
    new_params = determine_wavelength_solution_parameters.copy()
    new_params.pop("linelist")
    new_params.update(params)

    linelist = os.path.join(os.path.dirname(geminidr.__file__),
                            "gmos", "lookups", get_linelist(ad))

    ad_out = p.determineWavelengthSolution(
        [deepcopy(ad)], in_vacuo=in_vacuo, linelist=None, **new_params).pop()
    ad_out2 = p.determineWavelengthSolution(
        [ad], in_vacuo=in_vacuo, linelist=linelist, **new_params).pop()
    wave1 = am.get_named_submodel(ad_out[0].wcs.forward_transform, "WAVE")
    wave2 = am.get_named_submodel(ad_out2[0].wcs.forward_transform, "WAVE")
    x = np.arange(ad_out[0].shape[1])
    np.testing.assert_array_equal(wave1(x), wave2(x))


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
    arc_table = os.path.join(p.inst_lookups, get_linelist(ad))
    arc_lines = np.loadtxt(arc_table, usecols=[0]) / 10.0

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
            dpi=150, num="{:s}_{:s}_{:s}_{:s}_{:.0f}".format(
                name, grism, filter, camera, central_wavelength))

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

        fig_name = os.path.join(output_dir, "{:s}_{:s}_{:s}_{:s}_{:.0f}.png".format(
            name, grism, filter, camera, central_wavelength))

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

        fig_name = os.path.join(
            output_dir, "{:s}_{:s}_{:s}_{:s}_{:.0f}_non_linear_comps.png".format(
                name, grism, filter, camera, central_wavelength))

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

        fig_name = os.path.join(
            output_dir, "{:s}_{:s}_{:s}_{:s}_{:.0f}_residuals.png".format(
                name, grism, filter, camera, central_wavelength))

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
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, _, _, _ in input_pars:
        print('Downloading files...')
        basename = filename.split("_")[0] + ".fits"
        sci_path = download_from_archive(basename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = GNIRSLongslit([sci_ad])
        p.prepare()
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


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
