import pytest

import numpy as np
from astropy.table import Table

from geminidr.niri.primitives_niri_image import NIRIImage


@pytest.mark.parametrize("tolerance", (0, 1))
@pytest.mark.parametrize("use_common", (False, True))
def test_scale_by_object_flux_image(p_niri_image, tolerance, use_common):
    expected_results = {(0, False): (1, 1/2, 1/3),
                        (0, True): (1, 1/2, 1/3),
                        (1, False): (1, 170100/170201, 100200/100404),  # 0.9994054, 0.00796915
                        (1, True): (1, 100/101, 100/102)  # 0.990985, 0.9803924
                        }

    p_niri_image.scaleByObjectFlux(tolerance=tolerance, use_common=use_common)
    adoutputs = p_niri_image.streams['main']
    scalings = [ad[0].data.mean() for ad in adoutputs]
    np.testing.assert_allclose(scalings, expected_results[tolerance, use_common],
                               rtol=1e-5)


@pytest.fixture
def p_niri_image(astrofaker):
    full_objcat = Table([[100.0, 100.1, 100.2, 100.3],
                         [5.0, 5.1, 5.2, 5.3],
                         [100., 200., 300., 400.],
                         [2., 2., 2., 2.]],
                        names=('X_WORLD', 'Y_WORLD', 'FLUX_AUTO', 'FLUXERR_AUTO'))
    adinputs = []
    for i in range(1, 4):
        ad = astrofaker.create('NIRI', 'IMAGE', filename=f'N20010101S{i:04d}.fits')
        ad.init_default_extensions()
        ad.add(1)
        ad.exposure_time = i * 10
        rows = np.ones(len(full_objcat), dtype=bool)
        rows[i] = False
        ad[0].OBJCAT = full_objcat[rows]
        ad[0].OBJCAT['FLUX_AUTO'][0] += i - 1
        adinputs.append(ad)

    return NIRIImage(adinputs)

