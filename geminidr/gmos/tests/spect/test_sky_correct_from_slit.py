
"""
Regression tests for GMOS LS `skyCorrectFromSlit`. These tests run on real data
to ensure that the output is always the same.
"""

import os
import sys

import astrodata
import gemini_instruments  # noqa
import numpy as np
import pytest
from astrodata.testing import download_from_archive
from geminidr.gmos import primitives_gmos_longslit
from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS
from gempy.utils import logutils
from gempy.library import astrotools as at
from recipe_system.reduction.coreReduce import Reduce


# Test parameters -------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_aperturesTraced" suffix
test_datasets = [
    (
        "N20180508S0021_aperturesTraced.fits",  # B600 720
        dict(order=5, grow=0, aperture_growth=0),
        'N20180508S0021_ref_1.fits',
    ),
    (
        "N20180508S0021_aperturesTraced.fits",  # B600 720
        dict(order=8, grow=2, regions=':200,310:'),
        'N20180508S0021_ref_2.fits',
    ),
    (
        "N20180508S0021_aperturesTraced.fits",  # B600 720
        dict(order=5, grow=2, function='chebyshev', regions=':200,310:'),
        'N20180508S0021_ref_3.fits',
    ),
    (
        "N20180508S0021_aperturesTraced.fits",  # B600 720
        dict(order=5, function='chebyshev', lsigma=2, hsigma=2, niter=2),
        'N20180508S0021_ref_4.fits',
    ),
    (
        "S20190204S0079_aperturesTraced.fits",  # R400 : 0.750
        dict(order=8, lsigma=5, hsigma=5, grow=2),
        'S20190204S0079_ref_1.fits',
    ),
    (
        "S20181024S0035_aperturesTraced.fits",  # R400 : 0.656
        dict(),
        'S20181024S0035_ref_1.fits',
    ),
]

# Tests Definitions -----------------------------------------------------------

@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename, params, refname", test_datasets)
def test_regression_sky_correct_from_slit(filename, params, refname,
                                          change_working_dir, path_to_inputs,
                                          path_to_refs):

    path = os.path.join(path_to_inputs, filename)
    ad = astrodata.from_file(path)

    with change_working_dir():
        logutils.config(file_name=f'log_regression_{ad.data_label()}.txt')
        p = primitives_gmos_longslit.GMOSLongslit([ad])
        p.skyCorrectFromSlit(**params)
        sky_subtracted_ad = p.writeOutputs(outfilename=refname).pop()

    ref_ad = astrodata.from_file(os.path.join(path_to_refs, refname))

    # Require that <1% of unmasked pixels differ by >1 sigma
    for ext, ref_ext in zip(sky_subtracted_ad, ref_ad):
        mask = ext.mask | ref_ext.mask | (ext.variance == 0)
        sigma_diffs = (at.divide0(ext.data - ref_ext.data,
                                  np.sqrt(ext.variance)))[mask == 0]
        assert (abs(sigma_diffs) > 1).sum() < 0.01 * sigma_diffs.size


# -- Recipe to create pre-processed data --------------------------------------
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """

    input_data = {
        "N20180508S0021.fits": {
            "arc": "N20180615S0409.fits",
        },
        "S20190204S0079.fits": {
            "arc": "S20190206S0030.fits",
        },
        "S20181024S0035.fits" : {
            "arc": "S20181025S0012.fits",
        },
    }

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, pars in input_data.items():

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        arc_path = download_from_archive(pars['arc'])

        sci_ad = astrodata.from_file(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend([arc_path])
        arc_reduce.runr()
        arc = arc_reduce.output_filenames.pop()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.mosaicDetectors()
        p.attachWavelengthSolution(arc=arc)
        p.distortionCorrect()
        p.findApertures(max_apertures=1)
        p.traceApertures(order=2,
                         nsum=20,
                         step=10,
                         max_shift=0.09,
                         max_missed=5)

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../")
        print(f'Wrote pre-processed file to:\n    {processed_ad.filename}')


if __name__ == '__main__':
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
