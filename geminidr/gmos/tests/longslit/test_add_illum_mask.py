import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

# The location is the lowest row (0-indexed) of the first bridge
datasets_and_locations = [("S20200122S0020.fits", 747),
                          ("N20200909S0259.fits", 719),
                          ("N20201010S0122.fits", 718),
                          ("S20201112S0034.fits", 751),
                          ("S20201208S0021.fits", 748),
                          ("S20100108S0052.fits", 1611)]


@pytest.fixture(scope='module')
def ad(request):
    file_on_disk = download_from_archive(request.param)
    return astrodata.open(file_on_disk)


@pytest.fixture(scope='module')
def start_row(request):
    return request.param


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("ad,start_row", datasets_and_locations, indirect=True)
def test_add_illum_mask_position(ad, start_row):
    p = GMOSLongslit([ad])
    p.prepare()
    ad = p.addIllumMaskToDQ().pop()

    # Chop off the bottom 200 rows because of the bias issue
    # and the bottom of the longslit mask for pre-Hamamatsu data
    # (larger CCDs) and  choose a middle column in case of edge effects
    actual_start_row = ad[0].mask[200:,100].argmax() + 200
    assert abs(start_row - actual_start_row) <= 2
