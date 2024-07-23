import pytest

import os

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from recipe_system.mappers.primitiveMapper import PrimitiveMapper


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("filename", ["hip93667_109_ad.fits"])
def test_fit_telluric(path_to_inputs, path_to_refs, filename):
    ad = astrodata.open(os.path.join(path_to_inputs, filename))

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.fitTelluric(magnitude="K=5.241", bbtemp=9650, shift_tolerance=None).pop()

    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    assert ad_compare(adout, adref)
