import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

# The location is the lowest row (0-indexed) of the first bridge
datasets_and_locations = [("S20200122S0020.fits", 747),
                          ("N20190601S0285.fits", 721),
                          ("N20190325S0388.fits", 721),
                          ("S20191202S0064.fits", 748),
                          ("S20190604S0111.fits", 755),
                          ("N20180908S0020.fits", 1420),
                          ]


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("filename,start_row", datasets_and_locations)
def test_add_illum_mask_position(filename, start_row):
    file_on_disk = download_from_archive(filename)
    ad = astrodata.open(file_on_disk)

    p = GMOSLongslit([ad])
    p.prepare(bad_wcs="ignore")
    ad = p.addIllumMaskToDQ().pop()

    start_row -= 2 ## because we are growing the mask by 2 rows

    # Chop off the bottom 200 rows because of the bias issue
    # and the bottom of the longslit mask for pre-Hamamatsu data
    # (larger CCDs) and  choose a middle column in case of edge effects
    actual_start_row = (ad[0].mask[200:,100] & 64).argmax() + 200
    assert abs(start_row - actual_start_row) <= 2
