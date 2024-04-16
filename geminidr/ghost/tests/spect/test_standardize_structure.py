import pytest

from copy import deepcopy

import numpy as np

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect

from . import ad_min  # minimum AD fixture


@pytest.mark.ghostspect
def test_standardizeStructure(ad_min):
    """
    Checks to make:

    - This is a no-op primitive - ensure no change is made (the Gemini-level
      primitive would try to add an MDF)
    """
    ad_orig = deepcopy(ad_min)
    gs = GHOSTSpect([])

    ad_out = gs.standardizeStructure([ad_min])[0]
    assert np.all([
        ad_orig.info() == ad_out.info(),
        ad_orig.phu == ad_out.phu,
        ad_orig[0].hdr == ad_out[0].hdr,
        len(ad_orig) == len(ad_out),
    ]), "standardizeStructure is no longer a no-op primitive"

