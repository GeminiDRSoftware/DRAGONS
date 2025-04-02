#!/usr/bin/env python
"""
Tests for the `p.slitIllumCorrect` primitive.
"""
import warnings
import os

import astrodata
import gemini_instruments
import numpy as np
import pytest

from astrodata.testing import download_from_archive
from astropy.modeling import fitting, models
from cycler import cycler
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from gempy.utils import logutils
from matplotlib import pyplot as plt
from recipe_system.reduction.coreReduce import Reduce


PLOT_PATH = "plots/geminidr/gmos/longslit/test_slit_illum_correct/"


datasets = [
    "N20190103S0462.fits",  # R400 : 0.725
    "N20190327S0056.fits",  # R150 : 0.650
    "S20190204S0006.fits",  # R400 : 0.850
]

same_roi_datasets = [
    (d.split('.')[0] + "_twilight.fits", d.split('.')[0] + "_slitIllum.fits")
    for d in datasets]

different_roi_datasets = [
    # GN Ham R150 : 0.690
    ("N20200715S0059_quartz.fits", "N20190602S0306_slitIllum.fits"),
    # GS EEV B600 : 0.520
    ("S20130601S0121_quartz.fits", "S20130602S0005_slitIllum.fits"),
    # GS Ham R400 : 0.850
    ("S20190204S0081_quartz.fits", "S20190204S0006_slitIllum.fits"),
]


@pytest.mark.gmosls
def test_dont_do_slit_illum(astrofaker):
    in_ad = astrofaker.create("GMOS-S", mode="SPECT")
    p = GMOSLongslit([in_ad])
    out_ad = p.slitIllumCorrect(do_cal='skip')[0]
    for in_ext, out_ext in zip(in_ad, out_ad):
        assert np.testing.assert_equal(in_ext.data, out_ext.data)


@pytest.mark.gmosls
def test_slit_illum_correct_without_slit_illumination(astrofaker):
    in_ad = astrofaker.create("GMOS-S", mode="SPECT")
    p = GMOSLongslit([in_ad])
    with pytest.raises(NotImplementedError):
        p.slitIllumCorrect()


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_data", same_roi_datasets, indirect=True)
def test_slit_illum_correct_same_roi(change_working_dir, input_data, request):
    ad, slit_illum_ad = input_data
    p = GMOSLongslit([ad])
    ad_out = p.slitIllumCorrect(slit_illum=slit_illum_ad)[0]

    for ext_out in ad_out:

        # Create output data
        data_o = np.ma.masked_array(ext_out.data, mask=ext_out.mask)

        # Bin columns
        n_rows = data_o.shape[0]
        fitter = fitting.LevMarLSQFitter()
        model = models.Chebyshev1D(c0=0.5 * n_rows, degree=2, domain=(0, n_rows))
        nbins = 10
        rows = np.arange(n_rows)

        for i in range(nbins):

            col_start = i * data_o.shape[1] // nbins
            col_end = (i + 1) * data_o.shape[1] // nbins

            cols = np.ma.mean(data_o[:, col_start:col_end], axis=1)

            fitted_model = fitter(model, rows, cols)

            # Check column is linear
            np.testing.assert_allclose(fitted_model.c2.value, 0, atol=0.023)

            # Check if slope is (almost) horizontal (< 1.0 deg)
            slope = np.rad2deg(np.arctan(fitted_model.c1.value / fitted_model.c0.value))
            assert np.abs(slope) <= 5.1

    if request.config.getoption("--do-plots"):
        plot_slit_illum_correct_results(ad, ad_out, fname="test_same_roi_")


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_data", different_roi_datasets, indirect=True)
def test_slit_illum_correct_different_roi(change_working_dir, input_data, request):

    ad, slit_illum_ad = input_data

    assert ad.detector_roi_setting() != slit_illum_ad.detector_roi_setting()

    p = GMOSLongslit([ad])
    ad_out = p.slitIllumCorrect(slit_illum=slit_illum_ad)[0]

    for ext_out in ad_out:

        # Create output data
        data_o = np.ma.masked_array(ext_out.data, mask=ext_out.mask)

        # Bin columns
        n_rows = data_o.shape[0]
        fitter = fitting.LevMarLSQFitter()
        model = models.Chebyshev1D(c0=0.5 * n_rows, degree=2, domain=(0, n_rows))
        model.c0.fixed = True
        nbins = 10
        rows = np.arange(n_rows)

        for i in range(nbins):

            col_start = i * data_o.shape[1] // nbins
            col_end = (i + 1) * data_o.shape[1] // nbins

            cols = np.ma.mean(data_o[:, col_start:col_end], axis=1)

            fitted_model = fitter(model, rows, cols)

            # Check column is linear
            np.testing.assert_allclose(fitted_model.c2.value, 0, atol=0.001)

            # Check if slope is (almost) horizontal (< 2.5 deg)
            slope_angle = np.rad2deg(np.arctan(
                fitted_model.c1.value / (rows.size // 2)))

            assert np.abs(slope_angle) <= 1.0

    if request.config.getoption("--do-plots"):
        plot_slit_illum_correct_results(ad, ad_out, fname="test_different_roi_")


# --- Helper functions and fixtures -------------------------------------------
@pytest.fixture
def input_data(request, path_to_inputs):
    """
    Returns the pre-processed input data and the associated slit illumination
    data.

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
        Input spectrum processed up to right before the `distortionDetermine`
        primitive.
    """
    def _load_file(filename):
        path = os.path.join(path_to_inputs, filename)

        if os.path.exists(path):
            _ad = astrodata.from_file(path)
        else:
            raise FileNotFoundError(path)

        return _ad

    ad = _load_file(request.param[0])
    slit_illum_ad = _load_file(request.param[1])

    return ad, slit_illum_ad


def plot_slit_illum_correct_results(ad1, ad2, fname="", nbins=50):

    fig, (ax1, ax2, ax3) = plt.subplots(
        figsize=(6, 9), num="slitIllumCorrect: {}".format(ad1.filename),
        nrows=3, sharex='all')

    ax1.set_prop_cycle(
        cycler(color=[plt.get_cmap('cool')(i) for i in np.linspace(0, 1, nbins)]))

    ax2.set_prop_cycle(
        cycler(color=[plt.get_cmap('cool')(i) for i in np.linspace(0, 1, nbins)]))

    for ext1, ext2 in zip(ad1, ad2):

        assert ext1.shape == ext2.shape
        height, width = ext1.shape

        detsec = ext1.detector_section()
        xb = ext1.detector_x_bin()
        y1 = height // 2 - 10
        y2 = height // 2 + 10

        cols = np.arange(detsec.x1, detsec.x2, xb)

        data1 = np.ma.masked_array(ext1.data, ext1.mask)
        data1 -= np.mean(data1[y1:y2], axis=0)

        data2 = np.ma.masked_array(ext2.data, ext2.mask)
        data2 -= np.mean(data2[y1:y2], axis=0)

        for i in range(nbins):

            row_start = i * height // nbins
            row_end = (i + 1) * height // nbins

            rows1 = np.ma.mean(data1[row_start:row_end], axis=0)
            rows2 = np.ma.mean(data2[row_start:row_end], axis=0)

            ax1.plot(cols, rows1, alpha=0.2)
            ax1.set_ylabel('Non-corrected Ad\n Mean along columns [adu]')

            ax2.plot(cols, rows2, alpha=0.2)
            ax2.set_ylabel('Corrected Ad\n Mean along columns [adu]')

            ax3.plot(cols, np.ma.std(data1, axis=0), 'C0-', alpha=0.5)
            ax3.plot(cols, np.ma.std(data2, axis=0), 'C1-', alpha=0.5)
            ax3.set_ylabel('std\n along columns [adu]')
            ax3.set_xlabel('Columns [px]')

    fig.tight_layout()
    plt.savefig(fname + ad1.filename.replace(".fits", ".png"))


# -- Recipe to create pre-processed data ---------------------------------------
def create_twilight_inputs():
    """
    Creates input data for tests using pre-processed twilight flat data and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """

    associated_calibrations = {
        "S20190204S0006.fits": {
            "bias": ["S20190203S0110.fits",
                     "S20190203S0109.fits",
                     "S20190203S0108.fits",
                     "S20190203S0107.fits",
                     "S20190203S0106.fits"],
            "twilight": ["S20190204S0006.fits"],
        },
        "N20190103S0462.fits": {
            "bias": ["N20190102S0531.fits",
                     "N20190102S0530.fits",
                     "N20190102S0529.fits",
                     "N20190102S0528.fits",
                     "N20190102S0527.fits"],
            "twilight": ["N20190103S0462.fits",
                         "N20190103S0463.fits"],
        },
        "N20190327S0056.fits": {
            "bias": ["N20190327S0098.fits",
                     "N20190327S0099.fits",
                     "N20190327S0100.fits",
                     "N20190327S0101.fits",
                     "N20190327S0102.fits"],
            "twilight": ["N20190327S0056.fits"],
        },
        "S20130602S0005.fits": {
            "bias": [
                "S20130601S0161.fits",
                "S20130601S0160.fits",
                "S20130601S0159.fits",
                "S20130601S0158.fits",
                "S20130601S0157.fits",
            ],
            "twilight": ["S20130602S0005.fits"],
        },
        "N20190602S0306.fits": {
            "bias": [
                "N20190601S0648.fits",
                "N20190601S0647.fits",
                "N20190601S0646.fits",
                "N20190601S0645.fits",
                "N20190601S0644.fits",
            ],
            "twilight": ["N20190602S0306.fits"],
        }
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/longslit/test_slit_illumination_correct/inputs"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(path)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():

        print('Download raw files')
        bias_path = [download_from_archive(f) for f in cals['bias']]
        twilight_path = [download_from_archive(f) for f in cals['twilight']]

        twilight_ad = astrodata.from_file(twilight_path[0])
        data_label = twilight_ad.data_label()

        print('Reducing BIAS for {:s}'.format(data_label))
        logutils.config(file_name='log_bias_{}.txt'.format(data_label))
        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_path)
        bias_reduce.runr()
        bias_master = bias_reduce.output_filenames.pop()
        del bias_reduce

        print('Reducing twilight flat:')
        logutils.config(file_name='log_twilight_{}.txt'.format(data_label))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = GMOSLongslit(
                [astrodata.from_file(f) for f in twilight_path])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.biasCorrect(bias=bias_master)
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.stackFrames()

            # Write non-mosaicked data
            twilight = p.writeOutputs(suffix="_twilight", strip=True)[0]

            # Write mosaicked data
            p = GMOSLongslit([twilight])
            p.makeSlitIllum()
            p.writeOutputs()

    os.chdir(cwd)
    return


def create_quartz_inputs():
    """
    Creates input data for tests using pre-processed twilight flat data and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    associated_calibrations = {
        "N20200715S0059.fits": {
            "quartz": ["N20200715S0059.fits"],
        },
        "S20130601S0121.fits": {
            "quartz": ["S20130601S0121.fits"],
        },
        "S20190204S0081.fits": {
            "quartz": ["S20190204S0081.fits"],
        },
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/longslit/test_slit_illum_correct/inputs"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(path)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():

        print('Download raw files')
        quartz_path = [download_from_archive(f) for f in cals['quartz']]

        quartz_ad = astrodata.from_file(quartz_path[0])
        data_label = quartz_ad.data_label()

        print('Reducing quartz lamp:')
        logutils.config(file_name='log_quartz_{}.txt'.format(data_label))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = GMOSLongslit(
                [astrodata.from_file(f) for f in quartz_path])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            # p.biasCorrect(bias=bias_master)
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.stackFrames()

            # Write non-mosaicked data
            p.writeOutputs(suffix="_quartz", strip=True)

    os.chdir(cwd)


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_twilight_inputs()
        create_quartz_inputs()
    else:
        pytest.main()
