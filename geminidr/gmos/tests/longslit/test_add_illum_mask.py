import pytest

import os
import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

# The location is the lowest row (0-indexed) of the first bridge
datasets_and_locations = [("S20200122S0020.fits", 747),
                          ("N20190601S0285.fits", 721),
                          ("N20190325S0388.fits", 721),
                          ("S20191202S0064.fits", 748),
                          ("S20190604S0111.fits", 755),
                          ("N20051127S0041.fits", 764),  # N&S
                          ("N20120516S0139.fits", 375),  # N&S
                          ("N20180908S0020.fits", 1420),  # N&S
                          ]


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("filename,start_row", datasets_and_locations)
def test_add_illum_mask_position(filename, start_row):
    file_on_disk = download_from_archive(filename)
    ad = astrodata.from_file(file_on_disk)

    p = GMOSLongslit([ad])
    p.prepare()
    ad = p.addIllumMaskToDQ().pop()

    start_row -= 2 ## because we are growing the mask by 2 rows

    # Chop off the bottom 200 rows because of the bias issue
    # and the bottom of the longslit mask for pre-Hamamatsu data
    # (larger CCDs) and  choose a middle column in case of edge effects
    actual_start_row = (ad[0].mask[200:,100] & 64).argmax() + 200
    assert abs(start_row - actual_start_row) <= 2


@pytest.mark.preprocessed_data
@pytest.mark.gmosls
def test_add_illum_mask_position_amp5(path_to_inputs, path_to_common_inputs):
    """Test of bad-amp5 GMOS-S data"""
    adinputs = [astrodata.from_file(os.path.join(
        path_to_inputs, f"S20220927S{i:04d}_prepared.fits"))
        for i in (190, 191)]
    bpmfile = os.path.join(path_to_common_inputs,
                           "bpm_20220128_gmos-s_Ham_22_full_12amp.fits")

    p = GMOSLongslit(adinputs)
    # A full addDQ is needed to mask out amp5
    p.addDQ(static_bpm=bpmfile)

    start_row = 731

    # Chop off the bottom 200 rows because of the bias issue
    # and the bottom of the longslit mask for pre-Hamamatsu data
    # (larger CCDs) and  choose a middle column in case of edge effects
    for ad in p.streams['main']:
        actual_start_row = (ad[0].mask[200:,100] & 64).argmax() + 200
        assert abs(start_row - actual_start_row) <= 2
