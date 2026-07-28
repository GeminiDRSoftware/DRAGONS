import pytest

from copy import deepcopy
import numpy as np

from geminidr.gmos.primitives_gmos_image import GMOSImage


@pytest.mark.parametrize("tiled", (False, True))
@pytest.mark.parametrize("factors", ("0.5,2", [0.5,2]))
def test_qe_correct_user_factors(astrofaker, tiled, factors):
    """
    Check that the primitive correctly handles a pair of user-supplied
    scaling factors when supplied as a string or as a python list.
    This also confirms that the primitive works correctly on untiled
    *and* tiled data.
    """
    ad = astrofaker.create("GMOS-N", "IMAGE")
    ad.init_default_extensions(binning=4, overscan=False)
    ad.add(1)
    p = GMOSImage([ad])
    if tiled:
        p.tileArrays(tile_all=False)
    ad = p.QECorrect(factors=factors).pop()
    for i, ext in enumerate(ad):
        np.testing.assert_allclose(ext.data, 2 ** ((i - 1) if tiled
                                                   else (i // 4 - 1)))


@pytest.mark.parametrize("common", (False, True))
def test_qe_correct_calculate_and_apply_scaling(adinputs, common):
    """
    Check that scalings are calculated and applied correctly
    """
    correct_scaling = {False: [(2, 1, 2/3), (1.5, 1, 0.75), (4/3, 1, 0.8)],
                       True: [(36/23, 1, 36/49)] * 3}

    safe_adinputs = [deepcopy(ad) for ad in adinputs]
    p = GMOSImage(adinputs)
    p.QECorrect(factors=None, common=common)

    for i, (adin, adout) in enumerate(zip(safe_adinputs, p.streams['main'])):
        for j, (extin, extout) in enumerate(zip(adin, adout)):
            scaling = correct_scaling[common][i][j // 4]
            np.testing.assert_allclose(extin.data * scaling, extout.data,
                                       rtol=2e-7)


@pytest.fixture(scope="function")
def adinputs(astrofaker):
    # input 1 has sky levels 1, 2, 3
    # input 2 has sky levels 2, 3, 4
    # input 3 has sky levels 3, 4, 5
    adinputs = []
    for i in range(3):
        ad = astrofaker.create("GMOS-N", "IMAGE")
        ad.init_default_extensions(binning=4, overscan=False)
        # add stars to check it's a multiplicative, not additive, change
        for j, ext in enumerate(ad):
            ext.add(i + j // 4 + 1)
            ext.add_star(amplitude=1, fwhm=3, x=ext.shape[1] // 2,
                         y=ext.shape[0] // 2)
        adinputs.append(ad)

    return adinputs
