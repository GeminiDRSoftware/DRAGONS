#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import numpy as np
import os
import pytest

from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

import astrodata
import geminidr
from astropy.modeling import models
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from gempy.library import transform, astromodels as am
from gempy.testing import assert_have_same_distortion
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory


# Test parameters --------------------------------------------------------------
fixed_parameters_for_determine_distortion = {
    "fwidth": None,
    "id_only": False,
    "max_missed": 5,
    "max_shift": 0.05,
    "min_snr": 5.,
    "nsum": 10,
    "spatial_order": 3,
    "spectral_order": 4,
}

# Each test input filename contains the original input filename with
# "_mosaic" suffix
datasets = [
    # Process Arcs: GMOS-N ---
    "N20100115S0346_mosaic.fits",  # B600:0.500 EEV
    # "N20130112S0390_mosaic.fits",  # B600:0.500 E2V
    # "N20170609S0173_mosaic.fits",  # B600:0.500 HAM
    # "N20170403S0452_mosaic.fits",  # B600:0.590 HAM Full Frame 1x1
    # "N20170415S0255_mosaic.fits",  # B600:0.590 HAM Central Spectrum 1x1
    # "N20171016S0010_mosaic.fits",  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    # "N20171016S0127_mosaic.fits",  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    # "N20100307S0236_mosaic.fits",  # B1200:0.445 EEV
    # "N20130628S0290_mosaic.fits",  # B1200:0.420 E2V
    # "N20170904S0078_mosaic.fits",  # B1200:0.440 HAM
    # "N20170627S0116_mosaic.fits",  # B1200:0.520 HAM
    # "N20100830S0594_mosaic.fits",  # R150:0.500 EEV
    # "N20100702S0321_mosaic.fits",  # R150:0.700 EEV
    # "N20130606S0291_mosaic.fits",  # R150:0.550 E2V
    # "N20130112S0574_mosaic.fits",  # R150:0.700 E2V
    # "N20130809S0337_mosaic.fits",  # R150:0.700 E2V
    # "N20140408S0218_mosaic.fits",  # R150:0.700 E2V
    # "N20180119S0232_mosaic.fits",  # R150:0.520 HAM
    # "N20180516S0214_mosaic.fits",  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2
    # "N20171007S0439_mosaic.fits",  # R150:0.650 HAM
    # "N20171007S0441_mosaic.fits",  # R150:0.650 HAM
    # "N20101212S0213_mosaic.fits",  # R400:0.550 EEV
    # "N20100202S0214_mosaic.fits",  # R400:0.700 EEV
    # "N20130106S0194_mosaic.fits",  # R400:0.500 E2V
    # "N20130422S0217_mosaic.fits",  # R400:0.700 E2V
    # "N20170108S0210_mosaic.fits",  # R400:0.660 HAM
    # "N20171113S0135_mosaic.fits",  # R400:0.750 HAM
    # "N20100427S1276_mosaic.fits",  # R600:0.675 EEV
    # "N20180120S0417_mosaic.fits",  # R600:0.860 HAM
    # "N20100212S0143_mosaic.fits",  # R831:0.450 EEV
    # "N20100720S0247_mosaic.fits",  # R831:0.850 EEV
    # "N20130808S0490_mosaic.fits",  # R831:0.571 E2V
    # "N20130830S0291_mosaic.fits",  # R831:0.845 E2V
    # "N20170910S0009_mosaic.fits",  # R831:0.653 HAM
    # "N20170509S0682_mosaic.fits",  # R831:0.750 HAM
    # "N20181114S0512_mosaic.fits",  # R831:0.865 HAM
    # "N20170416S0058_mosaic.fits",  # R831:0.865 HAM
    # "N20170416S0081_mosaic.fits",  # R831:0.865 HAM
    # "N20180120S0315_mosaic.fits",  # R831:0.865 HAM
    # # Process Arcs: GMOS-S ---
    # # "S20130218S0126_mosaic.fits",  # B600:0.500 EEV - todo: won't pass
    # "S20130111S0278_mosaic.fits",  # B600:0.520 EEV
    # "S20130114S0120_mosaic.fits",  # B600:0.500 EEV
    # "S20130216S0243_mosaic.fits",  # B600:0.480 EEV
    # "S20130608S0182_mosaic.fits",  # B600:0.500 EEV
    # "S20131105S0105_mosaic.fits",  # B600:0.500 EEV
    # "S20140504S0008_mosaic.fits",  # B600:0.500 EEV
    # "S20170103S0152_mosaic.fits",  # B600:0.600 HAM
    # "S20170108S0085_mosaic.fits",  # B600:0.500 HAM
    # "S20130510S0103_mosaic.fits",  # B1200:0.450 EEV
    # "S20130629S0002_mosaic.fits",  # B1200:0.525 EEV
    # "S20131123S0044_mosaic.fits",  # B1200:0.595 EEV
    # "S20170116S0189_mosaic.fits",  # B1200:0.440 HAM
    # "S20170103S0149_mosaic.fits",  # B1200:0.440 HAM
    # "S20170730S0155_mosaic.fits",  # B1200:0.440 HAM
    # "S20171219S0117_mosaic.fits",  # B1200:0.440 HAM
    # "S20170908S0189_mosaic.fits",  # B1200:0.550 HAM
    # "S20131230S0153_mosaic.fits",  # R150:0.550 EEV
    # "S20130801S0140_mosaic.fits",  # R150:0.700 EEV
    # "S20170430S0060_mosaic.fits",  # R150:0.717 HAM
    # # "S20170430S0063_mosaic.fits",  # R150:0.727 HAM - todo: won't pass
    # "S20171102S0051_mosaic.fits",  # R150:0.950 HAM
    # "S20130114S0100_mosaic.fits",  # R400:0.620 EEV
    # "S20130217S0073_mosaic.fits",  # R400:0.800 EEV
    # "S20170108S0046_mosaic.fits",  # R400:0.550 HAM
    # "S20170129S0125_mosaic.fits",  # R400:0.685 HAM
    # "S20170703S0199_mosaic.fits",  # R400:0.800 HAM
    # "S20170718S0420_mosaic.fits",  # R400:0.910 HAM
    # # "S20100306S0460_mosaic.fits",  # R600:0.675 EEV - todo: won't pass
    # # "S20101218S0139_mosaic.fits",  # R600:0.675 EEV - todo: won't pass
    # "S20110306S0294_mosaic.fits",  # R600:0.675 EEV
    # "S20110720S0236_mosaic.fits",  # R600:0.675 EEV
    # "S20101221S0090_mosaic.fits",  # R600:0.690 EEV
    # "S20120322S0122_mosaic.fits",  # R600:0.900 EEV
    # "S20130803S0011_mosaic.fits",  # R831:0.576 EEV
    # "S20130414S0040_mosaic.fits",  # R831:0.845 EEV
    # "S20170214S0059_mosaic.fits",  # R831:0.440 HAM
    # "S20170703S0204_mosaic.fits",  # R831:0.600 HAM
    # "S20171018S0048_mosaic.fits",  # R831:0.865 HAM
]


# Tests Definitions ------------------------------------------------------------
@pytest.mark.skip("No diagnostic value; the WCS test covers this")
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_regression_for_determine_distortion_using_models_coefficients(
        ad, change_working_dir, ref_ad_factory, request):
    """
    Runs the `determineDistortion` primitive on a preprocessed data and compare
    its model with the one in the reference file.

    Parameters
    ----------
    ad : pytest.fixture (AstroData)
        Fixture that reads the filename and loads as an AstroData object.
    change_working_dir : pytest.fixture
        Fixture that changes the working directory
        (see :mod:`astrodata.testing`).
    reference_ad : pytest.fixture
        Fixture that contains a function used to load the reference AstroData
        object (see :mod:`recipe_system.testing`).
    request : pytest.fixture
        PyTest built-in containing command line options.
    """
    with change_working_dir():
        logutils.config(file_name='log_model_{:s}.txt'.format(ad.data_label()))
        p = GMOSLongslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(distortion_determined_ad.filename)
    assert_have_same_distortion(distortion_determined_ad, ref_ad, atol=1)

    if request.config.getoption("--do-plots"):
        do_plots(distortion_determined_ad, ref_ad)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_regression_for_determine_distortion_using_wcs(
        ad, change_working_dir, ref_ad_factory):
    """
    Runs the `determineDistortion` primitive on a preprocessed data and compare
    its model with the one in the reference file. The distortion model needs to
    be reconstructed because different coefficients might return same results.

    Parameters
    ----------
    ad : pytest.fixture (AstroData)
        Fixture that reads the filename and loads as an AstroData object.
    change_working_dir : pytest.fixture
        Fixture that changes the working directory
        (see :mod:`astrodata.testing`).
    reference_ad : pytest.fixture
        Fixture that contains a function used to load the reference AstroData
        object (see :mod:`recipe_system.testing`).
    """
    with change_working_dir():
        logutils.config(file_name='log_fitcoord_{:s}.txt'.format(ad.data_label()))
        p = GMOSLongslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(distortion_determined_ad.filename)
    model = distortion_determined_ad[0].wcs.get_transform(
        "pixels", "distortion_corrected")[1]
    ref_model = ref_ad[0].wcs.get_transform("pixels", "distortion_corrected")[1]

    # Otherwise we're doing something wrong!
    assert model.__class__.__name__ == ref_model.__class__.__name__ == "Chebyshev2D"

    X, Y = np.mgrid[:ad[0].shape[0], :ad[0].shape[1]]

    np.testing.assert_allclose(model(X, Y), ref_model(X, Y), atol=0.05)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_fitcoord_table_and_gwcs_match(ad, change_working_dir):
    """
    Runs determineDistortion and checks that the model in the gWCS is the same
    as the model in the FITCOORD table. The FITCOORD table is never used by
    DRAGONS.

    Parameters
    ----------
    ad: pytest.fixture (AstroData)
        Fixture that reads the filename and loads as an AstroData object.
    change_working_dir : pytest.fixture
        Fixture that changes the working directory
        (see :mod:`astrodata.testing`).
    """
    with change_working_dir():
        logutils.config(file_name='log_match_{:s}.txt'.format(ad.data_label()))
        p = GMOSLongslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    model = distortion_determined_ad[0].wcs.get_transform(
        "pixels", "distortion_corrected")

    fitcoord = distortion_determined_ad[0].FITCOORD
    fitcoord_model = am.table_to_model(fitcoord[0])
    fitcoord_inv = am.table_to_model(fitcoord[1])

    np.testing.assert_allclose(model[1].parameters, fitcoord_model.parameters)
    np.testing.assert_allclose(model.inverse[1].parameters, fitcoord_inv.parameters)


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


def do_plots(ad, ad_ref):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : AstroData

    ad_ref : AstroData
    """
    n_hlines = 25
    n_vlines = 25

    output_dir = "./plots/geminidr/gmos/test_gmos_spect_ls_distortion_determine"
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grating = ad.disperser(pretty=True)
    bin_x = ad.detector_x_bin()
    bin_y = ad.detector_y_bin()
    central_wavelength = ad.central_wavelength() * 1e9  # in nanometers

    # -- Show distortion map ---
    for ext_num, ext in enumerate(ad):
        fname, _ = os.path.splitext(os.path.basename(ext.filename))
        n_rows, n_cols = ext.shape

        x = np.linspace(0, n_cols, n_vlines, dtype=int)
        y = np.linspace(0, n_rows, n_hlines, dtype=int)

        X, Y = np.meshgrid(x, y)

        model = ext.wcs.get_transform("pixels", "distortion_corrected")[1]
        U = X - model(X, Y)
        V = np.zeros_like(U)

        fig, ax = plt.subplots(
            num="Distortion Map {:s} #{:d}".format(fname, ext_num))

        vmin = U.min() if U.min() < 0 else -0.1 * U.ptp()
        vmax = U.max() if U.max() > 0 else +0.1 * U.ptp()
        vcen = 0

        Q = ax.quiver(
            X, Y, U, V, U, cmap="coolwarm",
            norm=colors.DivergingNorm(vcenter=vcen, vmin=vmin, vmax=vmax))

        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title(
            "Distortion Map\n{:s} #{:d}- Bin {:d}x{:d}".format(
                fname, ext_num, bin_x, bin_y))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(Q, extend="max", cax=cax, orientation="vertical")
        cbar.set_label("Distortion [px]")

        fig.tight_layout()
        fig_name = os.path.join(
            output_dir, "{:s}_{:d}_{:s}_{:.0f}_distMap.png".format(
                fname, ext_num, grating, central_wavelength))

        fig.savefig(fig_name)
        del fig, ax

    # -- Show distortion model difference ---
    for num, (ext, ext_ref) in enumerate(zip(ad, ad_ref)):
        name, _ = os.path.splitext(ext.filename)
        shape = ext.shape
        data = generate_fake_data(shape, ext.dispersion_axis() - 1)

        model_out = ext.wcs.get_transform("pixels", "distortion_corrected")
        model_ref = ext_ref.wcs.get_transform("pixels", "distortion_corrected")

        transform_out = transform.Transform(model_out)
        transform_ref = transform.Transform(model_ref)

        data_out = transform_out.apply(data, output_shape=ext.shape)
        data_ref = transform_ref.apply(data, output_shape=ext.shape)

        data_out = np.ma.masked_invalid(data_out)
        data_ref = np.ma.masked_invalid(data_ref)

        fig, ax = plt.subplots(
            dpi=150, num="Distortion Comparison: {:s} #{:d}".format(name, num))

        im = ax.imshow(data_ref - data_out)

        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title(
            "Difference between output and reference: \n {:s} #{:d} ".format(
                name, num))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(im, extend="max", cax=cax, orientation="vertical")
        cbar.set_label("Distortion [px]")

        fig_name = os.path.join(
            output_dir, "{:s}_{:d}_{:s}_{:.0f}_distDiff.png".format(
                name, num, grating, central_wavelength))

        fig.savefig(fig_name)


def generate_fake_data(shape, dispersion_axis, n_lines=100):
    """
    Helper function that generates fake arc data.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the output data with (nrows, ncols)
    dispersion_axis : {0, 1}
        Dispersion axis along rows (0) or along columns (1)
    n_lines : int
        Number of random lines to be added (default: 100)

    Returns
    -------
    :class:`~astropy.modeling.models.Model`
        2D Model that can be applied to an array.
    """
    np.random.seed(0)
    nrows, ncols = shape

    data = np.zeros((nrows, ncols))
    line_positions = np.random.random_integers(0, ncols, size=n_lines)
    line_intensities = 100 * np.random.random_sample(n_lines)

    if dispersion_axis == 0:
        data[:, line_positions] = line_intensities
        data = ndimage.gaussian_filter(data, [5, 1])
    else:
        data[line_positions, :] = line_intensities
        data = ndimage.gaussian_filter(data, [1, 5])

    data = data + (np.random.random_sample(data.shape) - 0.5) * 10

    return data


def remap_distortion_model(model, dispersion_axis):
    """
    Remaps the distortion model so it can return a 2D array.

    Parameters
    ----------
    model : :class:`~astropy.modeling.models.Model`
        A model that receives 2D data and returns 1D data.

    dispersion_axis : {0 or 1}
        Define distortion model along the rows (0) or along the columns (1).

    Returns
    -------
    :class:`~astropy.modeling.models.Model`
        A model that receives and returns 2D data.

    See also
    --------
    - https://docs.astropy.org/en/stable/modeling/compound-models.html#advanced-mappings

    """
    m = models.Identity(2)

    if dispersion_axis == 0:
        m.inverse = models.Mapping((0, 1, 1)) | (model & models.Identity(1))
    else:
        m.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & model)

    return m

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

    for filename in datasets:
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
        os.chdir("..")
        print("Wrote pre-processed file to:\n"
              "    {:s}".format(processed_ad.filename))


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
