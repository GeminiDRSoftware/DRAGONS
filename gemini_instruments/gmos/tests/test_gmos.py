import pytest

import astrodata
import astrodata.testing
import gemini_instruments

import numpy as np


GMOS_DESCRIPTORS_TYPES = [
    ('detector_x_offset', float),
    ('detector_y_offset', float),
    ('nod_count', tuple),
    ('nod_offsets', tuple),
    ('pixel_scale', float),
    ('shuffle_pixels', int),
]

test_files = [
    "N20110826S0336.fits",
    "N20150624S0106.fits",
    "N20160524S0119.fits",
    "N20170529S0168.fits",
]


@pytest.fixture(params=test_files)
def ad(request):
    filename = request.param
    path = astrodata.testing.download_from_archive(filename)
    return astrodata.open(path)


@pytest.mark.dragons_remote_data
def test_is_right_instance(ad):
    assert isinstance(ad, gemini_instruments.gmos.adclass.AstroDataGmos)


@pytest.mark.dragons_remote_data
def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] in ['GMOS-N', 'GMOS-S']
    assert ad.instrument() == ad.phu['INSTRUME']


@pytest.mark.dragons_remote_data
def test_can_return_ad_length(ad):
    assert len(ad)


@pytest.mark.parametrize("descriptor,expected_type", GMOS_DESCRIPTORS_TYPES)
@pytest.mark.dragons_remote_data
def test_descriptor_matches_type(ad, descriptor, expected_type):
    value = getattr(ad, descriptor)()
    assert isinstance(value, expected_type) or value is None, \
        "Assertion failed for file: {}".format(ad.filename)


def test_tag_as_standard_fake(astrofaker):
    # LTT4363 (a high proper motion specphot) on Jan 1, 2021
    ad = astrofaker.create('GMOS-S', ['SPECT'],
                           extra_keywords={'RA': 176.46534847,
                                           'DEC': -64.84352513,
                                           'DATE-OBS': '2021-01-01T12:00:00.000',
                                           'OBSTYPE': 'OBJECT'}
                           )
    assert 'STANDARD' in ad.tags


@pytest.mark.dragons_remote_data
def test_tag_as_standard_real():
    path = astrodata.testing.download_from_archive("S20190215S0188.fits")
    ad = astrodata.open(path)
    assert 'STANDARD' in ad.tags


def test_ra_dec_from_text(astrofaker):
    ad = astrofaker.create('GMOS-S', ['SPECT'],
                           extra_keywords={'RA': '03:48:30.113',
                                           'DEC': '+24:20:43.00',
                                           'DATE-OBS': '2021-01-01T12:00:00.000'}
                           )
    assert ad.ra() == pytest.approx(57.12547083333333)
    assert ad.dec() == pytest.approx(24.345277777777778)

    # test bad RA/DEC values, just doing this for GMOS but it's testing the base
    ad = astrofaker.create('GMOS-S', ['SPECT'],
                           extra_keywords={'RA': 'Fail',
                                           'DEC': 'Fail',
                                           'DATE-OBS': '2021-01-01T12:00:00.000'}
                           )
    assert ad.ra() is None
    assert ad.dec() is None


if __name__ == '__main__':

    pytest.main()
