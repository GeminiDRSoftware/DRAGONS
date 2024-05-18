#!/usr/bin/env python
"""
Tests related to F2 Long-slit Spectroscopy Arc primitives.

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
from geminidr.f2.primitives_f2_longslit import F2Longslit
from gempy.library import transform, astromodels as am
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory

from geminidr.f2.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS


# Test parameters --------------------------------------------------------------
fixed_parameters_for_determine_distortion = {
    "fwidth": None,
    "id_only": False,
    "max_missed": None,
    "max_shift": 0.05,
    "min_snr": 7.,
    "nsum": 10,
    "spatial_order": 3,
    "spectral_order": 3,
    "min_line_length": 0.3,
    "debug_reject_bad": False
}

input_pars = [
    # Process Arcs: F2 ---
    # (Input File, params)
    # grism: JH, filter: JH (1.400um)
    ("S20211216S0177_flatCorrected.fits", dict()), # 2-pix slit
    # grism: HK, filter: HK (1.870um)
    ("S20220319S0060_flatCorrected.fits", dict()), # 3-pix slit, new HK filter
    # grism: HK, filter: JH (1.100um)
    ("S20220101S0044_flatCorrected.fits", dict()), # 3-pix slit
    # grism: R3K, filter: J-low (1.100um)
    ("S20131018S0230_flatCorrected.fits", dict()), # 6-pix slit
    # grism: R3K, filter: J (1.250um)
    ("S20170715S0121_flatCorrected.fits", dict()), # 1-pix slit
    # grism: R3K, filter: H (1.650um)
    ("S20210903S0053_flatCorrected.fits", dict()), # 6-pix slit
    # grism: R3K, filter: Ks (2.200um)
    ("S20220515S0026_flatCorrected.fits", dict()), # 3-pix slit
    # grism: R3K, filter: K-long (2.200um)
    ("S20150624S0023_flatCorrected.fits", dict()), # 1-pix slit
    # from OH emission sky lines in science:
    ("S20180114S0104_flatCorrected.fits", dict())
]

associated_calibrations = {
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
    "S20180114S0104.fits": {
        'flat_darks': ["S20180120S0222.fits",
                 "S20180120S0223.fits",
                 "S20180120S0224.fits",
                 "S20180120S0225.fits",
                 "S20180120S0226.fits"],
        'flat': ["S20180114S0118.fits"], # ND2.0
        'arc_darks': ["S20180117S0033.fits",
                 "S20180117S0034.fits",
                 "S20180117S0035.fits",
                 "S20180117S0036.fits",
                 "S20180117S0037.fits"],
    },
}

# Tests Definitions ------------------------------------------------------------
@pytest.mark.f2
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad,params", input_pars, indirect=['ad'])
def test_regression_for_determine_distortion_using_wcs(
        ad, params, change_working_dir, ref_ad_factory):
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
        p = F2Longslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(distortion_determined_ad.filename)
    model = distortion_determined_ad[0].wcs.get_transform(
        "pixels", "distortion_corrected")[2]
    ref_model = ref_ad[0].wcs.get_transform("pixels", "distortion_corrected")[2]

    # Otherwise we're doing something wrong!
    assert model.__class__.__name__ == ref_model.__class__.__name__ == "Chebyshev2D"

    X, Y = np.mgrid[:ad[0].shape[0], :ad[0].shape[1]]

    np.testing.assert_allclose(model(X, Y), ref_model(X, Y), atol=0.05)


@pytest.mark.gnirs
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, params", input_pars, indirect=['ad'])
def test_fitcoord_table_and_gwcs_match(ad, params, change_working_dir):
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
        p = F2Longslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    model = distortion_determined_ad[0].wcs.get_transform(
        "pixels", "distortion_corrected")

    fitcoord = distortion_determined_ad[0].FITCOORD
    fitcoord_model = am.table_to_model(fitcoord[0])
    fitcoord_inv = am.table_to_model(fitcoord[1])

    np.testing.assert_allclose(model[2].parameters, fitcoord_model.parameters)
    np.testing.assert_allclose(model.inverse[2].parameters, fitcoord_inv.parameters)


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

    output_dir = "./plots/geminidr/f2/test_f2_spect_ls_determine_wavelength_solution"
    os.makedirs(output_dir, exist_ok=True)

    name, _ = os.path.splitext(ad.filename)
    grism = ad.disperser(pretty=True)
    filter = ad.filter_name(pretty=True)
    central_wavelength = ad.central_wavelength(asNanometers=True)  # in nanometers

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
            "Distortion Map\n{:s}_{:s}_{:s}".format(
                fname, grism, filter))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(Q, extend="max", cax=cax, orientation="vertical")
        cbar.set_label("Distortion [px]")

        fig.tight_layout()
        fig_name = os.path.join(
            output_dir, "{:s}_{:s}_{:s}_distMap.png".format(
                fname, grism, filter))

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
            output_dir, "{:s}_{:s}_{:s}_distDiff.png".format(name, grism, filter))

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
        p.prepare()
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
        p.determineDistortion(**{**fixed_parameters_for_determine_distortion,
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
