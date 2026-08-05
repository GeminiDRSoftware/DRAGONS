import pytest

import os

from numpy.testing import assert_allclose

import astrodata, gemini_instruments

from recipe_system.mappers.primitiveMapper import PrimitiveMapper


@pytest.fixture
def ad(path_to_inputs, request):
    return astrodata.open(os.path.join(path_to_inputs, request.param))


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,result",
                         [("S20240112S0130_varAdded.fits", "446:13 581:13 1153:13 1548:13")],
                         indirect=["ad"])
def test_find_acquistion_slits(ad, result):
    """Check that the findAcquisitionPrimitive returns the correct answer"""
    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.findAcquisitionSlits().pop()
    slits_out = adout.phu['ACQSLITS'].split(" ")
    slits_ref = result.split(" ")

    assert len(slits_out) == len(slits_ref)
    for slit_out, slit_ref in zip(slits_out, slits_ref):
        coords_out = [int(x) for x in slit_out.split(":")]
        coords_ref = [int(x) for x in slit_ref.split(":")]
        assert_allclose(coords_out, coords_ref, atol=1)