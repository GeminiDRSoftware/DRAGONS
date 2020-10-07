"""
Regression tests for GMOS LS `skyCorrectFromSlit`. These tests run on real data
to ensure that the output is always the same. Further investigation is needed
to check if these outputs are scientifically relevant.
"""

import os

import astrodata
import astropy
import gemini_instruments  # noqa
import geminidr
import numpy as np
import pytest
from astrodata.testing import download_from_archive
from astropy import table
from astropy.utils import minversion
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce

ASTROPY_LT_42 = not minversion(astropy, '4.2')

# Test parameters -------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_aperturesTraced" suffix
test_datasets = [
    (
        "N20180508S0021_aperturesTraced.fits",  # B600 720
        dict(order=5, grow=0),
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
        dict(order=5, function='legendre', lsigma=2, hsigma=2, max_iters=2),
        'N20180508S0021_ref_4.fits',
    ),
]

# ("N20180509S0010_aperturesTraced.fits", dict(order=5, grow=2)),  # R400 900
# ("N20180516S0081_aperturesTraced.fits", dict(order=5, grow=2)),  # R600 860
# ("N20190427S0126_aperturesTraced.fits", dict(order=5, grow=2)),  # R400 625
# ("N20190427S0127_aperturesTraced.fits", dict(order=5, grow=2)),  # R400 725
# ("N20190427S0141_aperturesTraced.fits", dict(order=5, grow=2)),  # R150 660

# Tests Definitions -----------------------------------------------------------


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("filename, params, refname", test_datasets)
def test_regression_extract_1d_spectra(filename, params, refname,
                                       change_working_dir, path_to_inputs,
                                       path_to_refs):

    func = params.get('function', 'spline')
    if not func.startswith('spline') and ASTROPY_LT_42:
        pytest.skip('Astropy 4.2 is required to use the linear fitter '
                    'with weights')

    path = os.path.join(path_to_inputs, filename)
    ad = astrodata.open(path)

    with change_working_dir():
        logutils.config(file_name=f'log_regression_{ad.data_label()}.txt')
        p = primitives_gmos_spect.GMOSSpect([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.skyCorrectFromSlit(**params)
        sky_subtracted_ad = p.writeOutputs(outfilename=refname).pop()

    ref_ad = astrodata.open(os.path.join(path_to_refs, refname))

    for ext, ref_ext in zip(sky_subtracted_ad, ref_ad):
        np.testing.assert_allclose(ext.data, ref_ext.data, atol=0.01)


# Local Fixtures and Helper Functions -----------------------------------------


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
        [
            [1],  # Number
            [1],  # ndim
            [0],  # degree
            [0],  # domain_start
            [width - 1],  # domain_end
            [center],  # c0
            [-5],  # aper_lower
            [5],  # aper_upper
        ],
        names=[
            'number', 'ndim', 'degree', 'domain_start', 'domain_end', 'c0',
            'aper_lower', 'aper_upper'
        ])

    ad[0].APERTURE = aperture
    return ad


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
            "center": 244
        },
        "N20180509S0010.fits": {
            "arc": "N20180509S0080.fits",
            "center": 259
        },
        "N20180516S0081.fits": {
            "arc": "N20180516S0214.fits",
            "center": 255
        },
        "N20190427S0126.fits": {
            "arc": "N20190427S0267.fits",
            "center": 259
        },
        "N20190427S0127.fits": {
            "arc": "N20190427S0268.fits",
            "center": 258
        },
        "N20190427S0141.fits": {
            "arc": "N20190427S0270.fits",
            "center": 264
        },
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/spect/{}".format(__file__.split('.')[0])
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
        p.traceApertures(trace_order=2,
                         nsum=20,
                         step=10,
                         max_shift=0.09,
                         max_missed=5)

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../")
        print(f'Wrote pre-processed file to:\n    {processed_ad.filename}')


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
