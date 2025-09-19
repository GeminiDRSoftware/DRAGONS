#!/usr/bin/env python
"""
Regression tests for GMOS LS `extractSpectra`. These tests run on real data to
ensure that the output is always the same. Further investigation is needed to
check if these outputs are scientifically relevant.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` fixture to be called and the AstroData object returned.
"""
import numpy as np
import os
import pytest

import astrodata
import gemini_instruments
import geminidr

from astropy.table import Table
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory


# Test parameters --------------------------------------------------------------
test_datasets = [
    "N20180508S0021_aperturesTraced.fits",  # B600 720
    # "N20180509S0010_aperturesTraced.fits",  # R400 900
    # "N20180516S0081_aperturesTraced.fits",  # R600 860
    # "N20190201S0163_aperturesTraced.fits",  # B600 530
    # "N20190313S0114_aperturesTraced.fits",  # B600 482
    # "N20190427S0123_aperturesTraced.fits",  # R400 525
    # "N20190427S0126_aperturesTraced.fits",  # R400 625
    # "N20190427S0127_aperturesTraced.fits",  # R400 725
    # "N20190427S0141_aperturesTraced.fits",  # R150 660
]


# Tests Definitions ------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", test_datasets, indirect=True)
def test_regression_on_extract_1d_spectra(ad, ref_ad_factory, change_working_dir):

    """
    Regression test for the :func:`~geminidr.gmos.GMOSSpect.extractSpectra`
    primitive.

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
        logutils.config(
            file_name='log_regression_{:s}.txt'.format(ad.data_label()))

        p = primitives_gmos_spect.GMOSSpect([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.extractSpectra(method="aperture", width=None, grow=10)
        extracted_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(extracted_ad.filename)

    for ext, ref_ext in zip(extracted_ad, ref_ad):
        assert ext.data.ndim == 1
        np.testing.assert_allclose(ext.data, ref_ext.data, rtol=5e-7, atol=1e-3)

    # Check the RA and DEC; the maths here is just a quick way to evaluate
    # the Chebyshev1D model
    hdr = extracted_ad[0].hdr
    x = 0.5 * (ad[0].shape[0] - 1)
    aptable = ad[0].APERTURE
    y = aptable['c0'][0]
    if 'c2' in aptable.colnames:
        y -= aptable['c2'][0]
    if 'c4' in aptable.colnames:
        y += aptable['c4'][0]
    np.testing.assert_allclose((hdr['XTRACTRA'], hdr['XTRACTDE']),
                               ad[0].wcs(x, y)[1:])


# Local Fixtures and Helper Functions ------------------------------------------
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
def _add_aperture_table(ad, center):
    """
    Adds a fake aperture table to the `AstroData` object (NB. this is
    incomplete compared with the real one, which also includes domain & model
    type information in the header).

    Parameters
    ----------
    ad : AstroData
    center : int

    Returns
    -------
    AstroData : the input data with an `.APERTURE` table attached to it.
    """
    width = ad[0].shape[1]

    aperture = Table(
        [[1],  # Number
         [center],  # c0
         [-10.],  # aper_lower
         [10.],  # aper_upper
         ],
        names=[
            'number',
            'c0',
            'aper_lower',
            'aper_upper'],
        dtype=(np.int32, np.float64, np.float64, np.float64)
    )

    ad[0].APERTURE = aperture
    return ad


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
    from recipe_system.reduction.coreReduce import Reduce
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS

    associated_info = {
        "N20180508S0021_aperturesTraced.fits": {
            "arc": ["N20180615S0409.fits"], "center": 255},
        "N20180509S0010_aperturesTraced.fits": {
            "arc": ["N20180509S0080.fits"], "center": 259},
        "N20180516S0081_aperturesTraced.fits": {
            "arc": ["N20180516S0214.fits"], "center": 255},
        "N20190201S0163_aperturesTraced.fits": {
            "arc": ["N20190201S0176.fits"], "center": 255},
        "N20190313S0114_aperturesTraced.fits": {
            "arc": ["N20190313S0132.fits"], "center": 254},
        "N20190427S0123_aperturesTraced.fits": {
            "arc": ["N20190427S0266.fits"], "center": 260},
        "N20190427S0126_aperturesTraced.fits": {
            "arc": ["N20190427S0267.fits"], "center": 259},
        "N20190427S0127_aperturesTraced.fits": {
            "arc": ["N20190427S0268.fits"], "center": 258},
        "N20190427S0141_aperturesTraced.fits": {
            "arc": ["N20190427S0270.fits"], "center": 264},
    }

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename in test_datasets:

        arcs_name = associated_info[filename]["arc"]
        aperture_center = associated_info[filename]["center"]

        print('Downloading files...')
        basename = filename.split("_")[0] + ".fits"
        sci_path = download_from_archive(basename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()
        arcs_path = [download_from_archive(f) for f in arcs_name]

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        # arc_reduce.mode = "ql"
        arc_reduce.files.extend(arcs_path)
        arc_reduce.runr()
        arc_master = arc_reduce.output_filenames.pop()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_spect.GMOSSpect([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.mosaicDetectors()
        p.attachWavelengthSolution(arc=arc_master)
        p.distortionCorrect()

        temp_ad = p.makeIRAFCompatible()[0]
        temp_ad = _add_aperture_table(temp_ad, aperture_center)

        p = primitives_gmos_spect.GMOSSpect([temp_ad])
        p.skyCorrectFromSlit(order=5, grow=0)
        p.traceApertures(order=2, nsum=20, step=10, max_shift=0.09, max_missed=5)

        os.chdir("inputs/")
        _ = p.writeOutputs()[0]
        os.chdir("../../")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()

