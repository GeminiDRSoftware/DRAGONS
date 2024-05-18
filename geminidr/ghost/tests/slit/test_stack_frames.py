import pytest

import numpy as np

from geminidr.ghost.primitives_ghost_slit import GHOSTSlit

from . import ad_slit


@pytest.mark.ghostslit
def test_stackFrames_outputs(ad_slit):
    """
    Checks to make:

    - Only one file comes out
    - Dimensions of the output image match those of the input image
    """
    p = GHOSTSlit([ad_slit])
    p.prepare()
    output = p.stackFrames()
    assert len(output) == 1, 'Output length not 1'
    assert len(output[0]) == 1, 'Output frame does not have 1 slice'
    assert np.all([output[0][0].data.shape ==
                   _.data.shape for _ in ad_slit]), "Stacked frame shape " \
                                               "does not match inputs"

