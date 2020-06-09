import os
import pathlib

import numpy as np
import pytest

import astrodata
import gemini_instruments  # noqa
from astrodata.testing import download_from_archive
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.primitives_gmos_spect import GMOSSpect
from gempy.utils import logutils

# Test parameters -------------------------------------------------------------
test_datasets = [
    "S20190808S0048_distortionCorrected.fits",  # R400 at 0.740 um
]


# Tests Definitions -----------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_cosmics(adinputs, caplog):
    # add some additional fake cosmics
    ext = adinputs[0][0]
    size = 50
    np.random.seed(42)
    cr_x = np.random.randint(low=5, high=ext.shape[0] - 5, size=size)
    cr_y = np.random.randint(low=5, high=ext.shape[1] - 5, size=size)
    cr_brightnesses = np.random.uniform(low=1000, high=5000, size=size)
    ext.data[cr_x, cr_y] += cr_brightnesses

    p = GMOSSpect(adinputs)
    adout = p.flagCosmicRays()[0]
    mask = adout[0].mask
    # check some pixels with real cosmics
    for pix in [(497, 520), (138, 219), (420, 634), (297, 1871)]:
        assert (mask[pix] & DQ.cosmic_ray) == DQ.cosmic_ray
    # And check our fake cosmics. Since they are placed at random positions
    # we may miss a few ones.
    assert np.sum((mask & DQ.cosmic_ray)[cr_x, cr_y] != DQ.cosmic_ray) < 3


@pytest.fixture(scope='function')
def adinputs(path_to_inputs):
    return [astrodata.open(os.path.join(path_to_inputs, f))
            for f in test_datasets]


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

    associated_calibrations = {
        "S20190808S0048.fits": 'S20190808S0167.fits',
    }

    path = pathlib.Path('dragons_test_inputs')
    path = path / "geminidr" / "gmos" / "test_gmos_spect_ls_cosmics" / "inputs"
    path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    print('Current working directory:\n    {!s}'.format(path.cwd()))

    for fname, arc_fname in associated_calibrations.items():
        arc_ad = astrodata.open(download_from_archive(arc_fname))
        sci_ad = astrodata.open(download_from_archive(fname))
        data_label = sci_ad.data_label()

        print(f'===== Reducing ARC for {data_label} =====')
        logutils.config(file_name=f'log_arc_{data_label}.txt')
        p = GMOSSpect([arc_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.mosaicDetectors()
        p.makeIRAFCompatible()
        p.determineWavelengthSolution()
        p.determineDistortion()
        arc = p.writeOutputs().pop()

        print('===== Reducing pre-processed data =====')
        logutils.config(file_name=f'log_{data_label}.txt')
        p = GMOSSpect([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.mosaicDetectors()
        p.distortionCorrect(arc=arc)
        p.writeOutputs()


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main([__file__])
