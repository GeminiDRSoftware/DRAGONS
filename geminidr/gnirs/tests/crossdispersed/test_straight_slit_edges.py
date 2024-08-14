"""
This test works with several primitives to trace the edges of the GNIRS XD
slits in a flatfield, cut and mask outside them, and then use this flatfield
to cut another file (actually, the pre-cut version of itself) and confirm
that the transformed edges are vertical and round-trip back to their original
coordinates.
"""
import pytest
import os

import numpy as np

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed


@pytest.mark.preprocessed_data
@pytest.mark.gnirsxd
@pytest.mark.parametrize('filename', ["N20130821S0308_stack.fits"])
def test_edges_and_slit_centers(filename, path_to_inputs):
    ad = astrodata.open(os.path.join(path_to_inputs, filename))
    # Clear the mask so only pixels beyond the edge are masked
    for ext in ad:
        ext.mask = None
    p = GNIRSCrossDispersed([ad])
    p.determineSlitEdges(search_radius=30)
    p.cutSlits()
    ad_masked = p.maskBeyondSlit().pop()

    ad = astrodata.open(os.path.join(path_to_inputs, filename))
    p = GNIRSCrossDispersed([ad])
    ad = p.flatCorrect(flat=ad_masked).pop()
    for ext in ad:
        y = np.arange(ext.shape[0])
        # First and last unmasked pixels in each row
        x1 = ext.mask.argmin(axis=1)
        x2 = ext.shape[1] - ext.mask[:, ::-1].argmin(axis=1) - 1
        x1_on = x1 > 0
        x2_on = x2 < ext.shape[1] - 1
        t = ext.wcs.get_transform('pixels', 'rectified')
        left_transformed = t(x1[x1_on], y[x1_on])
        right_transformed = t(x2[x2_on], y[x2_on])

        # Check that the transformed edges are vertical. The stddev will
        # be fairly large, since we are transforming integer pixels only
        assert np.std(left_transformed[0]) < 0.5
        assert np.std(right_transformed[0]) < 0.5

        # And confirm that performing the round-trip gets up back to the
        # original coordinates
        left_round_trip = t.inverse(*left_transformed)
        right_round_trip = t.inverse(*right_transformed)
        assert np.allclose(left_round_trip[0], x1[x1_on], atol=0.2)
        assert np.allclose(right_round_trip[0], x2[x2_on], atol=0.2)
