#!/usr/bin/env python
"""
Regression tests for GMOS LS extraction1D. These tests run on real data to ensure
that the output is always the same. Further investigation is needed to check if
these outputs are scientifically relevant.
"""

import os
import numpy as np
import pytest

from astropy import table

import astrodata
import geminidr

from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory


# Test parameters --------------------------------------------------------------
test_datasets = [
    "N20180508S0021_mosaic.fits",  # B600 720
    "N20180509S0010_mosaic.fits",  # R400 900
    "N20180516S0081_mosaic.fits",  # R600 860
    "N20190201S0163_mosaic.fits",  # B600 530
    "N20190313S0114_mosaic.fits",  # B600 482
    "N20190427S0123_mosaic.fits",  # R400 525
    "N20190427S0126_mosaic.fits",  # R400 625
    "N20190427S0127_mosaic.fits",  # R400 725
    "N20190427S0141_mosaic.fits",  # R150 660
]

fixed_test_parameters_for_determine_distortion = {
    "debug": False,
    "max_missed": 5,
    "max_shift": 0.09,
    "nsum": 20,
    "step": 10,
    "order": 2,
}


# Tests Definitions ------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", test_datasets, indirect=True)
def test_regression_trace_apertures(ad, change_working_dir, ref_ad_factory):

    with change_working_dir():
        logutils.config(file_name="log_regression_{}.txt".format(ad.data_label()))
        p = primitives_gmos_spect.GMOSSpect([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.traceApertures()
        aperture_traced_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(aperture_traced_ad.filename)

    for ext, ref_ext in zip(aperture_traced_ad, ref_ad):
        input_table = ext.APERTURE
        reference_table = ref_ext.APERTURE

        assert input_table['aper_lower'][0] <= 0
        assert input_table['aper_upper'][0] >= 0

        keys = ext.APERTURE.colnames
        actual = np.array([input_table[k] for k in keys])
        desired = np.array([reference_table[k] for k in keys])
        np.testing.assert_allclose(desired, actual, atol=0.05)


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
        Input spectrum processed up to right before the `calculateSensitivity`
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
    from gempy.utils import logutils

    input_data = [
        ("N20180508S0021.fits", 255),  # B600 720
        ("N20180509S0010.fits", 259),  # R400 900
        ("N20180516S0081.fits", 255),  # R600 860
        ("N20190201S0163.fits", 255),  # B600 530
        ("N20190313S0114.fits", 254),  # B600 482
        ("N20190427S0123.fits", 260),  # R400 525
        ("N20190427S0126.fits", 259),  # R400 625
        ("N20190427S0127.fits", 258),  # R400 725
        ("N20190427S0141.fits", 264),  # R150 660
    ]

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, center in input_data:

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

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
        _ad = p.makeIRAFCompatible()[0]

        width = _ad[0].shape[1]

        aperture = table.Table(
            [[1],  # Number
             [1],  # ndim
             [0],  # degree
             [0],  # domain_start
             [width - 1],  # domain_end
             [center],  # c0
             [-10],  # aper_lower
             [10],  # aper_upper
             ],
            names=[
                'number',
                'ndim',
                'degree',
                'domain_start',
                'domain_end',
                'c0',
                'aper_lower',
                'aper_upper']
        )

        _ad[0].APERTURE = aperture

        os.chdir("inputs/")
        _ad.write(overwrite=True)
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(_ad.filename))
        os.chdir("../")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
