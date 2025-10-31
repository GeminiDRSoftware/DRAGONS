"""
Compare
"""
import pytest
import os
from pytest import approx
from copy import deepcopy

import numpy as np

import astrodata, gemini_instruments
from erfa.ufunc import ecm06
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed

@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_apply_slit_model(path_to_inputs):
    preprocessed_ad = astrodata.open(os.path.join(path_to_inputs, 'N20190613S0173_varAdded.fits'))
    preprocessed_flat = astrodata.open(os.path.join(path_to_inputs, 'N20190613S0165_flat.fits'))

    p = GNIRSCrossDispersed([deepcopy(preprocessed_ad)])
    ad_with_slit_model = p.applySlitModel(flat=preprocessed_flat).pop()
    p1 = GNIRSCrossDispersed([preprocessed_ad])
    ad_flat_corrected = p1.flatCorrect(flat=preprocessed_flat).pop()

    for ext_slit, ext_flat,  in zip(ad_with_slit_model, ad_flat_corrected):
        # Compare DQ and VAR planes of the flatCorrected and slitModelAttached images (should be the same)
        assert np.array_equal(ext_slit.mask, ext_flat.mask), "DQ planes differ"

        # Now check the WCS of flatCorrect-ed and slitModelApplied images (should be the same)
        mid_x = ext_slit.shape[1] / 2.
        mid_y = ext_slit.shape[0] / 2.
        # Check the midpoint of the extension
        assert np.allclose(ext_slit.wcs.forward_transform(mid_x, mid_y), ext_flat.wcs.forward_transform(mid_x, mid_y),
                                                                atol=1e-6), "WCS differ at midpoint"
        # Check the corners (need 'abs' for this since we're comparing to zero)
        for i in (0, ext_slit.shape[0]-1):
            for j in (0, ext_slit.shape[1]-1):
                assert np.allclose(ext_slit.wcs.forward_transform(i, j), ext_flat.wcs.forward_transform(i, j),
                                                                atol=1e-6), "WCS differ at corner"

        # Check that data and VAR plane remain unchanged after running the applySlitModel primitive,
        # and that the same sections of the data are present in the slitModelApplied image as in flatCorrected
        for ext_flat_detsec, ext_slit in zip(ad_flat_corrected.detector_section(),ad_with_slit_model):
            x1 = ext_flat_detsec.x1
            x2 = ext_flat_detsec.x2
            assert np.allclose(ext_slit.data, preprocessed_ad[0].data[:, x1:x2], atol=1e-6), "Data planes differ"
            assert np.allclose(ext_slit.variance, preprocessed_ad[0].variance[:, x1:x2], atol=1e-6), "VAR planes differ"
