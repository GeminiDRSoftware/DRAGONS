#!/usr/bin/env python
"""
Tests for the `makeSlitIllum` primitive. The primitive itself is
defined in :mod:`~geminidr.core.primitives_spect` but these tests use GMOS Spect
data.
"""
import os
import pytest
import warnings

from copy import copy, deepcopy

import astrodata
import numpy as np

from astrodata.testing import download_from_archive
from astropy.modeling import fitting, models
from gempy.utils import logutils
from geminidr.gmos import primitives_gmos_longslit
from geminidr.gmos.lookups import geometry_conf as geotable
from gempy.library import transform
from gempy.gemini import gemini_tools as gt
from matplotlib import gridspec
from matplotlib import pyplot as plt
from recipe_system.reduction.coreReduce import Reduce

PLOT_PATH = "plots/geminidr/gmos/test_gmos_spect_ls_create_slit_illumination/"

datasets = [
    "N20190103S0462.fits",  # R400 : 0.725
    "N20190327S0056.fits",  # R150 : 0.650
    "S20190204S0006.fits",  # R400 : 0.850
]

mosaicked_datasets = [d.split('.')[0] + "_mosaickedTwilight.fits" for d in datasets]
multiext_datasets = [d.split('.')[0] + "_twilight.fits" for d in datasets]


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", mosaicked_datasets, indirect=True)
def test_create_slit_illumination_with_mosaicked_data(ad, change_working_dir, request):
    """
    Test that can run `makeSlitIllum` in mosaicked data. This primitive
    creates a 2D image that is used to normalize the input data.

    After normalization, the intensity along a column should be more or less
    constant.

    There are several ways of doing this but, given the noise levels, we bin
    the data, fit a polynomium, and check that the fitted polynomium has its 1st
    and 2nd coefficients almost zero.
    """
    plot = request.config.getoption("--do-plots")
    np.random.seed(0)

    with change_working_dir():

        cwd = os.getcwd()
        print("Running tests inside folder:\n  {}".format(cwd))

        assert hasattr(ad[0], "wcs")

        p = primitives_gmos_longslit.GMOSLongslit([ad])
        p.makeSlitIllum(bins=25, border=10, debug_plot=plot)
        slit_illum_ad = p.writeOutputs(
            suffix="_mosaickedSlitIllum",  strip=True)[0]

        for ext, slit_ext in zip(ad, slit_illum_ad):
            assert ext.shape == slit_ext.shape

            # Create output data
            data_o = (np.ma.masked_array(ext.data, mask=ext.mask) /
                      np.ma.masked_array(slit_ext.data, mask=slit_ext.mask))

            # Bin columns
            fitter = fitting.LinearLSQFitter()
            model = models.Polynomial1D(degree=2)
            nbins = 50
            rows = np.arange(data_o.shape[0])

            for i in range(nbins):

                col_start = i * data_o.shape[1] // nbins
                col_end = (i + 1) * data_o.shape[1] // nbins

                cols = np.ma.mean(data_o[:, col_start:col_end], axis=1)

                fitted_model = fitter(model, rows, cols)

                # Check column is linear
                np.testing.assert_allclose(fitted_model.c2.value, 0, atol=0.01)

                # Check if slope is (almost) horizontal (< 1.0 deg)
                assert np.abs(
                    np.rad2deg(
                        np.arctan(
                            fitted_model.c1.value / (rows.size // 2)))) < 1.0

    if plot:
        os.makedirs(PLOT_PATH, exist_ok=True)
        print("Renaming plots to ",
              os.path.join(PLOT_PATH, ad.filename.replace(".fits", ".png")))
        os.rename(
            os.path.join(cwd, slit_illum_ad.filename.replace(".fits", ".png")),
            os.path.join(PLOT_PATH, ad.filename.replace(".fits", ".png")))


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", multiext_datasets, indirect=True)
def test_create_slit_illumination_with_multi_extension_data(ad, change_working_dir, request):
    """
    Test that can run `makeSlitIllum` in multi-extension data.
    """
    plot = request.config.getoption("--do-plots")

    with change_working_dir():

        cwd = os.getcwd()
        print("Running tests inside folder:\n  {}".format(cwd))
        p = primitives_gmos_longslit.GMOSLongslit([ad])
        p.makeSlitIllum(bins=25, border=10, debug_plot=plot)
        slit_illum_ad = p.writeOutputs()[0]

        for ext, slit_ext in zip(ad, slit_illum_ad):
            assert ext.shape == slit_ext.shape

            # Create output data
            data_o = (np.ma.masked_array(ext.data, mask=ext.mask) /
                      np.ma.masked_array(slit_ext.data, mask=slit_ext.mask))

            # Bin columns
            fitter = fitting.LinearLSQFitter()
            model = models.Polynomial1D(degree=2)
            nbins = 10
            rows = np.arange(data_o.shape[0])

            for i in range(nbins):

                col_start = i * data_o.shape[1] // nbins
                col_end = (i + 1) * data_o.shape[1] // nbins

                cols = np.ma.mean(data_o[:, col_start:col_end], axis=1)

                fitted_model = fitter(model, rows, cols)

                # Check column is linear
                np.testing.assert_allclose(fitted_model.c2.value, 0, atol=0.01)

                # Check if slope is (almost) horizontal (< 2.0 deg)
                assert np.abs(
                    np.rad2deg(
                        np.arctan(
                            fitted_model.c1.value / (rows.size // 2)))) < 1.5

    if plot:
        os.makedirs(PLOT_PATH, exist_ok=True)
        print("Renaming plots to ",
              os.path.join(PLOT_PATH, ad.filename.replace(".fits", ".png")))
        os.rename(
            os.path.join(cwd, slit_illum_ad.filename.replace(".fits", ".png")),
            os.path.join(PLOT_PATH, ad.filename.replace(".fits", ".png")))


def test_split_mosaic_into_extensions(request):
    """
    Tests helper function that split a mosaicked data into several extensions
    based on another multi-extension file that contains gWCS.
    """
    astrofaker = pytest.importorskip("astrofaker")

    ad = astrofaker.create('GMOS-S')
    ad.init_default_extensions(binning=2)

    ad = transform.add_mosaic_wcs(ad, geotable)
    ad = gt.trim_to_data_section(
        ad, keyword_comments={'NAXIS1': "", 'NAXIS2': "", 'DATASEC': "",
                              'TRIMSEC': "", 'CRPIX1': "", 'CRPIX2': ""})

    for i, ext in enumerate(ad):
        x1 = ext.detector_section().x1
        x2 = ext.detector_section().x2
        xb = ext.detector_x_bin()

        data = np.arange(x1 // xb, x2 // xb)[np.newaxis, :]
        data = np.repeat(data, ext.data.shape[0], axis=0)
        data = data + 0.1 * (0.5 - np.random.random(data.shape))

        ext.data = data

    mosaic_ad = transform.resample_from_wcs(
        ad, "mosaic", attributes=None, interpolant="linear", process_objcat=False)

    mosaic_ad[0].data = np.pad(mosaic_ad[0].data, 10, mode='edge')

    mosaic_ad[0].hdr[mosaic_ad._keyword_for('data_section')] = \
        '[1:{},1:{}]'.format(*mosaic_ad[0].shape[::-1])

    ad2 = primitives_gmos_longslit._split_mosaic_into_extensions(
        ad, mosaic_ad, border_size=10)

    if request.config.getoption("--do-plots"):

        palette = copy(plt.cm.viridis)
        palette.set_bad('r', 1)

        fig = plt.figure(num="Test: Split Mosaic Into Extensions", figsize=(8, 6.5), dpi=120)
        fig.suptitle("Test Split Mosaic Into Extensions\n Difference between"
                     " input and mosaicked/demosaicked data")

        gs = fig.add_gridspec(nrows=4, ncols=len(ad) // 3, wspace=0.1, height_ratios=[1, 1, 1, 0.1])

        for i, (ext, ext2) in enumerate(zip(ad, ad2)):

            data1 = ext.data
            data2 = ext2.data
            diff = np.ma.masked_array(data1 - data2, mask=np.abs(data1 - data2) > 1)
            height, width = data1.shape

            row = i // 4
            col = i % 4

            ax = fig.add_subplot(gs[row, col])
            ax.set_title("Ext {}".format(i + 1))
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            _ = [ax.spines[s].set_visible(False) for s in ax.spines]

            if col == 0:
                ax.set_ylabel("Det {}".format(row + 1))

            sub_gs = gridspec.GridSpecFromSubplotSpec(2, 2, ax, wspace=0.05, hspace=0.05)

            for j in range(4):
                sx = fig.add_subplot(sub_gs[j])
                im = sx.imshow(diff, origin='lower', cmap=palette, vmin=-0.1, vmax=0.1)

                sx.set_xticks([])
                sx.set_yticks([])
                sx.set_xticklabels([])
                sx.set_yticklabels([])
                _ = [sx.spines[s].set_visible(False) for s in sx.spines]

                if j == 0:
                    sx.set_xlim(0, 25)
                    sx.set_ylim(height - 25, height)

                if j == 1:
                    sx.set_xlim(width - 25, width)
                    sx.set_ylim(height - 25, height)

                if j == 2:
                    sx.set_xlim(0, 25)
                    sx.set_ylim(0, 25)

                if j == 3:
                    sx.set_xlim(width - 25, width)
                    sx.set_ylim(0, 25)

        cax = fig.add_subplot(gs[3, :])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Difference levels")

        os.makedirs(PLOT_PATH, exist_ok=True)

        fig.savefig(
            os.path.join(PLOT_PATH, "test_split_mosaic_into_extensions.png"))

    # Actual test ----
    for i, (ext, ext2) in enumerate(zip(ad, ad2)):
        data1 = np.ma.masked_array(ext.data[1:-1, 1:-1], mask=ext.mask)
        data2 = np.ma.masked_array(ext2.data[1:-1, 1:-1], mask=ext2.mask)

        np.testing.assert_almost_equal(data1, data2, decimal=1)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("filename", datasets)
def test_split_mosaic_into_extensions_metadata(filename):
    """
    Tests that the metadata is correctly propagated to the split object.
    """
    ad = astrodata.open(download_from_archive(filename))

    p = primitives_gmos_longslit.GMOSLongslit([ad])
    p.prepare()
    p.overscanCorrect()
    mosaicked_ad = p.mosaicDetectors().pop()

    ad2 = primitives_gmos_longslit._split_mosaic_into_extensions(
        ad, mosaicked_ad, border_size=10)

    for (ext, ext2) in zip(ad, ad2):
        assert (ext.shape == ext2.shape)
        assert (ext.data_section() == ext2.data_section())
        assert (ext.detector_section() == ext2.detector_section())
        assert (ext.array_section() == ext2.array_section())


# --- Helper functions and fixtures -------------------------------------------
@pytest.fixture
def ad(request, path_to_inputs):
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
        Input spectrum processed up to right before the `distortionDetermine`
        primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad


# -- Recipe to create pre-processed data ---------------------------------------
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed twilight flat data and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    from geminidr.gmos.tests.longslit import INPUTS_ROOT_PATH

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
        }
    }

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(INPUTS_ROOT_PATH, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():

        print('Downloading files...')
        twilight_path = [download_from_archive(f) for f in cals['twilight']]
        bias_path = [download_from_archive(f) for f in cals['bias']]

        twilight_ad = astrodata.open(twilight_path[0])
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

            p = primitives_gmos_longslit.GMOSLongslit(
                [astrodata.open(f) for f in twilight_path])

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
            p = primitives_gmos_longslit.GMOSLongslit([twilight])
            p.mosaicDetectors()
            p.writeOutputs(suffix="_mosaickedTwilight", strip=True)


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
