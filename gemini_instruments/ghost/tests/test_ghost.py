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
    assert 'STD' in ad.tags
    assert 'GHOST' in ad.tags
    assert 'BUNDLE' in ad.tags


if __name__ == '__main__':
    pytest.main()
