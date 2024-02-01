import pytest

import numpy as np

from geminidr.ghost.primitives_ghost_slit import _total_obj_flux

from . import ad_slit


@pytest.mark.ghostslit
def test__total_obj_flux(ad_slit):
    """
    Checks to make

    - Compare against already-known total flux?

    Measured flux needs to be within -2%/+1% of actual
    (There are slit losses due to restricted width of extraction)
    """
    sums = [ext.data.sum() for ext in ad_slit]
    fluxes = np.array([_total_obj_flux(None, ad_slit.res_mode(), ad_slit.ut_date(),
                                       ad_slit.filename, ext.data, None,
                                       binning=ad_slit.detector_x_bin())
                       for ext in ad_slit])
    for actual, measured in zip(sums, fluxes):
        assert 0.98 < measured / actual < 1.01


