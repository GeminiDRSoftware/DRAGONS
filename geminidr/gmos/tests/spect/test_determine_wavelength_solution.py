#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives. `input_files` is a
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
from importlib import import_module

import numpy as np
import pytest
from matplotlib import pyplot as plt
from astropy import units as u
from specutils.utils.wcs_utils import air_to_vac

import astrodata
import geminidr

from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit, GMOSClassicLongslit
from gempy.library import astromodels as am
from gempy.utils import logutils


# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'global',
    'fwidth': None,
    'order': 3,
    'min_snr': 10,
}

input_pars = [
    # Process Arcs: GMOS-N ---
    # (Input File, params)
    ("N20100115S0346_mosaic.fits", dict()),  # B600:0.500 EEV
    ("N20130112S0390_mosaic.fits", dict()),  # B600:0.500 E2V
    ("N20170609S0173_mosaic.fits", dict()),  # B600:0.500 HAM
    ("N20170403S0452_mosaic.fits", dict()),  # B600:0.590 HAM
    ("N20170415S0255_mosaic.fits", dict()),  # B600:0.590 HAM
    ("N20171016S0010_mosaic.fits", dict()),  # B600:0.500 HAM
    ("N20171016S0127_mosaic.fits", dict()),  # B600:0.500 HAM
    ("N20180103S0341_mosaic.fits", dict()),  # B600:0.430 HAM
    ("N20180113S0351_mosaic.fits", dict()),  # B600:0.750 HAM
    ("N20180615S0407_mosaic.fits", dict()),  # B600:0.620 HAM
    ("N20100307S0236_mosaic.fits", dict()),  # B1200:0.445 EEV
    ("N20130628S0290_mosaic.fits", dict()),  # B1200:0.420 E2V
    ("N20170904S0078_mosaic.fits", dict()),  # B1200:0.440 HAM
    # ("N20170627S0116_mosaic.fits", dict()),  # B1200:0.520 HAM  (KL passes locally, Mac and Linux, fails in Jenkins)
    ("N20100830S0594_mosaic.fits", dict()),  # R150:0.500 EEV
    ("N20100702S0321_mosaic.fits", dict()),  # R150:0.700 EEV
    ("N20130606S0291_mosaic.fits", dict()),  # R150:0.550 E2V
    ("N20130112S0574_mosaic.fits", dict()),  # R150:0.700 E2V
    #("N20130809S0337_mosaic.fits", dict()),  # R150:0.700 E2V  2" slit
    #("N20140408S0218_mosaic.fits", dict()),  # R150:0.700 E2V  2" slit
    ("N20180119S0232_mosaic.fits", dict()),  # R150:0.520 HAM
    ("N20171007S0439_mosaic.fits", dict()),  # R150:0.650 HAM
    #("N20181114S0512_mosaic.fits", dict()),  # R150:0.610 HAM  2" slit I can't identify
    ("N20180111S0155_mosaic.fits", dict()),  # R150:0.610 HAM
    ("N20171007S0441_mosaic.fits", dict()),  # R150:0.650 HAM
    ("N20101212S0213_mosaic.fits", dict()),  # R400:0.550 EEV
    ("N20100202S0214_mosaic.fits", dict()),  # R400:0.700 EEV
    ("N20130106S0194_mosaic.fits", dict(min_snr=3)),  # R400:0.500 E2V
    ("N20130422S0217_mosaic.fits", dict()),  # R400:0.700 E2V
    ("N20170108S0210_mosaic.fits", dict()),  # R400:0.660 HAM
    ("N20171113S0135_mosaic.fits", dict()),  # R400:0.750 HAM
    ("N20100427S1274_mosaic.fits", dict()),  # R600:0.475 EEV
    ("N20100427S1276_mosaic.fits", dict()),  # R600:0.675 EEV
    ("N20120615S0512_mosaic.fits", dict()),  # R600:0.750 e2v
    ("N20120615S0513_mosaic.fits", dict()),  # R600:0.950 e2v
    ("N20180120S0417_mosaic.fits", dict()),  # R600:0.865 HAM
    # actually closer to 833nm, so use "alternative_centers"
    ("N20180516S0214_mosaic.fits", dict(debug_alternative_centers=True)),  # R600:0.860 HAM
    ("N20100212S0143_mosaic.fits", dict()),  # R831:0.450 EEV
    ("N20100720S0247_mosaic.fits", dict()),  # R831:0.850 EEV
    ("N20130808S0490_mosaic.fits", dict()),  # R831:0.571 E2V
    ("N20130830S0291_mosaic.fits", dict()),  # R831:0.845 E2V
    ("N20170910S0009_mosaic.fits", dict()),  # R831:0.653 HAM
    ("N20170509S0682_mosaic.fits", dict()),  # R831:0.750 HAM
    #("N20170416S0058_mosaic.fits", dict()),  # R831:0.855 HAM
    ("N20170416S0081_mosaic.fits", dict()),  # R831:0.865 HAM
    ("N20180120S0315_mosaic.fits", dict()),  # R831:0.865 HAM
    ("N20190111S0271_mosaic.fits", dict()),  # R831:0.525 HAM
    #
    # # Process Arcs: GMOS-S ---
    ("S20130218S0126_mosaic.fits", dict()),  # B600:0.600 EEV
    ("S20130111S0278_mosaic.fits", dict()),  # B600:0.520 EEV
    ("S20130114S0120_mosaic.fits", dict()),  # B600:0.500 EEV
    ("S20130216S0243_mosaic.fits", dict()),  # B600:0.480 EEV
    ("S20130608S0182_mosaic.fits", dict()),  # B600:0.500 EEV
    ("S20131105S0105_mosaic.fits", dict()),  # B600:0.500 EEV
    ("S20140504S0008_mosaic.fits", dict()),  # B600:0.500 EEV
    ("S20170103S0152_mosaic.fits", dict(nbright=2)),  # B1200:0.440 HAM bad columns
    ("S20170108S0085_mosaic.fits", dict(nbright=2)),  # B600:0.500 HAM
    ("S20130510S0103_mosaic.fits", dict()),  # B1200:0.450 EEV
    ("S20130629S0002_mosaic.fits", dict()),  # B1200:0.525 EEV
    ("S20131123S0044_mosaic.fits", dict()),  # B1200:0.595 EEV
    ("S20170116S0189_mosaic.fits", dict(nbright=2)),  # B1200:0.440 HAM
    ("S20170908S0189_mosaic.fits", dict(nbright=1)),  # B1200:0.595 HAM bad column
    ("S20131230S0153_mosaic.fits", dict()),  # R150:0.550 EEV
    ("S20130801S0140_mosaic.fits", dict()),  # R150:0.700 EEV
    ("S20170430S0060_mosaic.fits", dict(nbright=2)),  # R150:0.717 HAM bad columns
    ("S20170430S0063_mosaic.fits", dict(nbright=2)),  # R150:0.727 HAM bad columns
    ("S20171102S0051_mosaic.fits", dict()),  # R150:0.950 HAM
    ("S20130114S0100_mosaic.fits", dict()),  # R400:0.620 EEV
    ("S20130217S0073_mosaic.fits", dict()),  # R400:0.800 EEV
    ("S20170108S0046_mosaic.fits", dict(nbright=2)),  # R400:0.550 HAM bad columns
    ("S20170129S0125_mosaic.fits", dict(nbright=1)),  # R400:0.685 HAM bad column
    ("S20170703S0199_mosaic.fits", dict()),  # R400:0.850 HAM
    ("S20170718S0420_mosaic.fits", dict()),  # R400:0.910 HAM
    #("S20101218S0139_mosaic.fits", dict()),  # R600:0.675 EEV 5-arcsec slit!
    #("S20110306S0294_mosaic.fits", dict()),  # R600:0.675 EEV 5-arcsec slit!
    ("S20110720S0236_mosaic.fits", dict()),  # R600:0.675 EEV
    ("S20101221S0090_mosaic.fits", dict()),  # R600:0.690 EEV
    ("S20120322S0122_mosaic.fits", dict()),  # R600:0.900 EEV
    ("S20130803S0011_mosaic.fits", dict()),  # R831:0.576 EEV
    ("S20130414S0040_mosaic.fits", dict()),  # R831:0.845 EEV
    ("S20170214S0059_mosaic.fits", dict(nbright=3)),  # R831:0.440 HAM
    ("S20170703S0204_mosaic.fits", dict()),  # R831:0.600 HAM
    ("S20171018S0048_mosaic.fits", dict())  # R831:0.865 HAM
]


# Tests Definitions ------------------------------------------------------------

@pytest.mark.wavecal
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
        p = GMOSLongslit([ad])
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
    p = GMOSClassicLongslit([])
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
    p = GMOSClassicLongslit([])
    p.viewer = geminidr.dormantViewer(p, None)
    new_params = determine_wavelength_solution_parameters.copy()
    new_params.pop("linelist")
    new_params.update(params)

    linelist = os.path.join(os.path.dirname(geminidr.__file__),
                            "gmos", "lookups", "CuAr_GMOS.dat")

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


def do_plots(ad):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : astrodata
    """
    output_dir = ("./plots/geminidr/gmos/"
                  "test_gmos_spect_ls_determine_wavelength_solution")
    p = GMOSClassicLongslit([])
    lookup_dir = os.path.dirname(import_module('.__init__',
                                                   p.inst_lookups).__file__)
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grating = ad.disperser(pretty=True)
    bin_x = ad.detector_x_bin()
    bin_y = ad.detector_y_bin()
    central_wavelength = ad.central_wavelength(asNanometers=True)

    p = GMOSLongslit([ad])
    arc_table = os.path.join(lookup_dir, "CuAr_GMOS.dat")
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
            dpi=150, num="{:s}_{:d}_{:s}_{:.0f}".format(
                name, ext_num, grating, central_wavelength))

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
                     "{:s}\n obtained with {:s} at {:.0f} nm".format(
                        name, grating, central_wavelength))

        if x0 > x1:
            ax.invert_xaxis()

        fig_name = os.path.join(output_dir, "{:s}_{:d}_{:s}_{:.0f}.png".format(
            name, ext_num, grating, central_wavelength))

        fig.savefig(fig_name)
        del fig, ax

        # -- Plot non-linear components ---
        fig, ax = plt.subplots(
            dpi=150, num="{:s}_{:d}_{:s}_{:.0f}_non_linear_comps".format(
                name, ext_num, grating, central_wavelength))

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
                     "{:s} obtained with {:s} at {:.0f}".format(
                        name, grating, central_wavelength))

        fig_name = os.path.join(
            output_dir, "{:s}_{:d}_{:s}_{:.0f}_non_linear_comps.png".format(
                name, ext_num, grating, central_wavelength))

        fig.savefig(fig_name)
        del fig, ax

        # -- Plot Wavelength Solution Residuals ---
        fig, ax = plt.subplots(
            dpi=150, num="{:s}_{:d}_{:s}_{:.0f}_residuals".format(
                name, ext_num, grating, central_wavelength))

        ax.plot(wavelengths, wavelengths - wavecal_model(peaks), "ko")

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Residuum [nm]")
        ax.set_title("Wavelength Calibrated Residuum for\n"
                     "{:s} obtained with {:s} at {:.0f}".format(
                        name, grating, central_wavelength))

        fig_name = os.path.join(
            output_dir, "{:s}_{:d}_{:s}_{:.0f}_residuals.png".format(
                name, ext_num, grating, central_wavelength))

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

    for filename, _ in input_pars:
        print('Downloading files...')
        basename = filename.split("_")[0] + ".fits"
        sci_path = download_from_archive(basename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.mosaicDetectors()
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
