import pytest

import random

import numpy as np

from geminidr.ghost.primitives_ghost_slit import GHOSTSlit

from . import ad_slit


CRFLUX = 50000


@pytest.mark.ghostslit
def test_fix_cosmic_rays(ad_slit):
    """
    Checks to make:

    - Check that all simulated CR are removed from test data
    - Check shape of output data matches shape of input
    """
    random.seed(0)
    modded_coords, sums = [], []
    shapes = ad_slit.shape
    for ext in ad_slit:
        # Switch on a '1' in the data plane of each slit. With all other
        # values being 0., that should trigger _mad detection
        # Ensure that a different coord pixel is flagged in each ext.
        success = False
        while not success:
            attempt_coord = tuple(random.randint(0, length - 1)
                                  for length in ext.shape)
            if attempt_coord not in modded_coords:
                ext.data[attempt_coord] += CRFLUX
                modded_coords.append(attempt_coord)
                success = True
        sums.append(ext.data.sum())

    p = GHOSTSlit([ad_slit])
    p.fixCosmicRays()
    # Check CR replacement. Need a large-ish tolerance because
    # a CR in the slit region will affect the obj_flux and so
    # cause a scaling of the affected image
    np.testing.assert_allclose(sums, [ext.data.sum() + CRFLUX for ext in ad_slit],
                               atol=20), 'fixCosmicRays failed to remove all dummy cosmic rays'
    # Check for replacement header keyword
    np.testing.assert_array_equal(ad_slit.hdr['CRPIXREJ'], 1), \
        'Incorrect number of rejected pixels recorded in CRPIXREJ'
    np.testing.assert_array_equal(ad_slit.shape, shapes), \
        'fixCosmicRays has mangled data shapes'


