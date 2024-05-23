"""
Tests applied to primitives_image.py
"""

import pytest
import os

from numpy.testing import assert_array_equal

import astrodata, gemini_instruments
from geminidr.niri.primitives_niri_image import NIRIImage


object_mask_datasets = {"N20210512S0018_sourcesDetected.fits": "N20210512S0077_flatCorrected.fits"}


@pytest.mark.regression
@pytest.mark.parametrize("dataset", object_mask_datasets.keys())
def test_transfer_object_mask(path_to_inputs, path_to_refs, dataset):
    """
    Test the transferObjectMask primitive
    """
    ad_donor = astrodata.open(os.path.join(path_to_inputs, dataset))
    ad_target = astrodata.open(os.path.join(path_to_inputs, object_mask_datasets[dataset]))
    p = NIRIImage([ad_target])
    p.streams['donor'] = [ad_donor]
    adout = p.transferObjectMask(source="donor", dq_threshold=0.01, dilation=1.5,
                                 interpolant="linear").pop()
    adout.write(overwrite=True)
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))

    assert_array_equal(adout[0].OBJMASK, adref[0].OBJMASK)
