#!/usr/bin/env python
"""
Regression tests for GMOS LS `skyCorrectFromSlit`. These tests run on real data
to ensure that the output is always the same. Further investigation is needed to
check if these outputs are scientifically relevant.
"""

import os
import pytest
import numpy as np

from astropy import table

import astrodata
import gemini_instruments
import geminidr

from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory


# Test parameters --------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_aperturesTraced" suffix
test_datasets = [
    "N20180508S0021_aperturesTraced.fits",  # B600 720
    "N20180509S0010_aperturesTraced.fits",  # R400 900
    "N20180516S0081_aperturesTraced.fits",  # R600 860
    "N20190201S0163_aperturesTraced.fits",  # B600 530
    "N20190313S0114_aperturesTraced.fits",  # B600 482
    "N20190427S0123_aperturesTraced.fits",  # R400 525
    "N20190427S0126_aperturesTraced.fits",  # R400 625
    "N20190427S0127_aperturesTraced.fits",  # R400 725
    "N20190427S0141_aperturesTraced.fits",  # R150 660
]


# Tests Definitions ------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", test_datasets, indirect=True)
def test_regression_extract_1d_spectra(ad, change_working_dir,
                                       ref_ad_factory):

    with change_working_dir():

        logutils.config(
            file_name='log_regression_{}.txt'.format(ad.data_label()))

        p = primitives_gmos_spect.GMOSSpect([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.skyCorrectFromSlit(order=5, grow=0)
        sky_subtracted_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(sky_subtracted_ad.filename)

    for ext, ref_ext in zip(sky_subtracted_ad, ref_ad):
        np.testing.assert_allclose(ext.data, ref_ext.data, atol=0.01)


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


def _add_aperture_table(ad, center):
    """
    Adds a fake aperture table to the `AstroData` object.

    Parameters
    ----------
    ad : AstroData
    center : int

    Returns
    -------
    AstroData : the input data with an `.APERTURE` table attached to it.
    """
    width = ad[0].shape[1]

    aperture = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [width - 1],  # domain_end
         [center],  # c0
         [-5],  # aper_lower
         [5],  # aper_upper
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

    ad[0].APERTURE = aperture
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
    from gempy.utils import logutils
    from recipe_system.reduction.coreReduce import Reduce

    input_data = {
        "N20180508S0021.fits": {"arc": "N20180615S0409.fits", "center": 244},
        "N20180509S0010.fits": {"arc": "N20180509S0080.fits", "center": 259},
        "N20180516S0081.fits": {"arc": "N20180516S0214.fits", "center": 255},
        "N20190201S0163.fits": {"arc": "N20190201S0162.fits", "center": 255},
        "N20190313S0114.fits": {"arc": "N20190313S0132.fits", "center": 254},
        "N20190427S0123.fits": {"arc": "N20190427S0266.fits", "center": 260},
        "N20190427S0126.fits": {"arc": "N20190427S0267.fits", "center": 259},
        "N20190427S0127.fits": {"arc": "N20190427S0268.fits", "center": 258},
        "N20190427S0141.fits": {"arc": "N20190427S0270.fits", "center": 264},
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/test_gmos_spect_ls_sky_correct_from_slit/"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, pars in input_data.items():

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        arc_path = download_from_archive(pars['arc'])

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend([arc_path])
        arc_reduce.runr()
        arc = arc_reduce.output_filenames.pop()

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
        p.distortionCorrect(arc=arc)
        _ad = p.makeIRAFCompatible().pop()
        _ad = _add_aperture_table(_ad, pars['center'])

        p = primitives_gmos_spect.GMOSSpect([_ad])
        p.traceApertures(trace_order=2, nsum=20, step=10, max_shift=0.09, max_missed=5)

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