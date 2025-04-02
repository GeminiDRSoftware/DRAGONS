import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive

from ..decorators import insert_descriptor_values

def test_insert_descriptor_values(ad):

    @insert_descriptor_values()
    def fn(data, exposure_time=5):
        return exposure_time

    assert fn(ad) == ad.exposure_time()
    assert fn(ad, exposure_time=1) == 1
    assert fn(None) == 5


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad():
    ad_path = download_from_archive("N20150123S0337.fits")
    return astrodata.from_file(ad_path)
