#!/usr/bin/env python
import pytest

import astrodata
import astrodata.testing
import gemini_instruments
from gemini_instruments.f2 import AstroDataF2

import numpy as np


test_files = [
    "S20131121S0094.fits",
    "S20131126S1111.fits",
    "S20131126S1112.fits",
    "S20131126S1113.fits",
    "S20160112S0080.fits",
    "S20170103S0032.fits",
]

F2_DESCRIPTORS_TYPES = [
    ('detector_x_offset', float),
    ('detector_y_offset', float),
    ('pixel_scale', float),
]


@pytest.fixture(scope='module', params=test_files)
def ad(request):
    filename = request.param
    file_path = astrodata.testing.download_from_archive(filename)
    return astrodata.open(file_path)


@pytest.mark.dragons_remote_data
@pytest.mark.xfail(reason="AstroFaker changes the AstroData factory")
def test_is_right_type(ad):
    assert type(ad) == gemini_instruments.f2.adclass.AstroDataF2


@pytest.mark.dragons_remote_data
def test_is_right_instance(ad):
    assert isinstance(ad, gemini_instruments.f2.adclass.AstroDataF2)


@pytest.mark.dragons_remote_data
def test_extension_data_shape(ad):
    data = ad[0].data
    assert data.shape == (1, 2048, 2048)


@pytest.mark.dragons_remote_data
def test_tags(ad):
    tags = ad.tags
    expected_tags = {'F2', 'SOUTH', 'GEMINI'}
    assert expected_tags.issubset(tags)


@pytest.mark.dragons_remote_data
def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] == 'F2'
    assert ad.instrument() == ad.phu['INSTRUME']


@pytest.mark.dragons_remote_data
def test_can_return_ad_length(ad):
    assert len(ad) == 1


@pytest.mark.dragons_remote_data
def test_slice_range(ad):
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1:]
    assert len(slc) == 0

    for ext, md in zip(slc, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


@pytest.mark.dragons_remote_data
def test_read_a_keyword_from_phu(ad):
    assert ad.phu['INSTRUME'].strip() == 'F2'


@pytest.mark.dragons_remote_data
def test_read_a_keyword_from_hdr(ad):
    try:
        assert ad.hdr['CCDNAME'] == 'F2'
    except KeyError:
        # KeyError only accepted if it's because headers out of range
        assert len(ad) == 1


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("descriptor,expected_type", F2_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(descriptor, expected_type, ad):
    value = getattr(ad, descriptor)()
    assert isinstance(value, expected_type) or value is None, \
        "Assertion failed for file: {}".format(ad)


def test_ra_dec_from_text(astrofaker):
    ad = AstroDataF2()
    ad.phu['RA'] = '03:48:30.113'
    ad.phu['DEC'] = '+24:20:43.00'
    assert ad.target_ra() == pytest.approx(57.12547083333333)
    assert ad.target_dec() == pytest.approx(24.345277777777778)


if __name__ == "__main__":
    pytest.main()
