from datetime import datetime, timedelta

import pytest

import astrodata
import astrodata.testing
import gemini_instruments


test_files = [
    "S20221209S0007.fits",
    "S20221209S0009.fits",
    "S20221209S0010.fits",
    "S20221209S0011.fits",
    "S20221209S0012.fits",
    "S20221209S0049.fits",
    "S20221209S0054.fits",
    "S20221209S0057.fits",
    "S20221209S0061.fits",
]


@pytest.fixture(params=test_files)
def ad(request):
    filename = request.param
    path = astrodata.testing.download_from_archive(filename)
    return astrodata.open(path)


@pytest.mark.dragons_remote_data
def test_is_right_instance(ad):
    assert isinstance(ad, gemini_instruments.ghost.adclass.AstroDataGhost)


@pytest.mark.dragons_remote_data
def test_can_return_ad_length(ad):
    assert len(ad)


@pytest.mark.dragons_remote_data
def test_instrument():
    path = astrodata.testing.download_from_archive("S20221209S0007.fits")
    ad = astrodata.open(path)
    assert ad.phu['INSTRUME'] == 'GHOST'
    assert ad.instrument() == 'GHOST'


@pytest.mark.dragons_remote_data
def test_various_tags():
    path = astrodata.testing.download_from_archive("S20221209S0007.fits")
    ad = astrodata.open(path)
    # assert 'STD' in ad.tags  #STD is no longer a tag
    assert 'GHOST' in ad.tags
    assert 'BUNDLE' in ad.tags


@pytest.mark.dragons_remote_data
def test_detector_x_bin():
    path = astrodata.testing.download_from_archive("S20221209S0007.fits")
    ad = astrodata.open(path)
    xbin = ad.detector_x_bin()
    # should be a dict, since we are a bundle
    assert(isinstance(xbin, dict))


@pytest.mark.dragons_remote_data
def test_ut_datetime():
    path = astrodata.testing.download_from_archive("S20221209S0007.fits")
    ad = astrodata.open(path)
    udt = ad.ut_datetime()
    # Check against expected UT Datetime, this descriptor also exercises the nascent PHU logic
    assert(abs(udt - datetime(2022, 12, 8, 20, 52, 22)) < timedelta(seconds=1))


@pytest.mark.dragons_remote_data
def test_data_label():
    path = astrodata.testing.download_from_archive("S20221209S0007.fits")
    ad = astrodata.open(path)
    dl = ad.data_label()
    # Check against expected UT Datetime, this descriptor also exercises the nascent PHU logic
    assert(dl == 'GS-ENG-GHOST-COM-3-123-001')


@pytest.mark.dragons_remote_data
def test_tab_bias():
    path = astrodata.testing.download_from_archive("S20221208S0089.fits")
    ad = astrodata.open(path)
    assert('BIAS' in ad.tags)


@pytest.mark.dragons_remote_data
def test_tab_flat():
    path = astrodata.testing.download_from_archive("S20221209S0026.fits")
    ad = astrodata.open(path)
    assert('FLAT' in ad.tags)


@pytest.mark.dragons_remote_data
def test_tab_arc():
    path = astrodata.testing.download_from_archive("S20221208S0064.fits")
    ad = astrodata.open(path)
    assert('ARC' in ad.tags)


if __name__ == '__main__':
    pytest.main()
