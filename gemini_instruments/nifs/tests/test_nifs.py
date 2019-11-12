import pytest

import astrodata
import gemini_instruments
from astrodata.testing import download_from_archive

filename = 'N20160727S0077.fits'


@pytest.mark.remote_data
def test_is_right_instance():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))
    # YES, this *can* be different from test_is_right_type. Metaclasses!
    assert isinstance(ad, gemini_instruments.nifs.adclass.AstroDataNifs)


@pytest.mark.remote_data
def test_extension_data_shape():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))
    data = ad[0].data

    assert data.shape == (2048, 2048)


@pytest.mark.remote_data
def test_tags():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))
    tags = ad.tags
    expected = {'DARK', 'RAW', 'AT_ZENITH', 'NORTH', 'AZEL_TARGET',
                'CAL', 'UNPREPARED', 'NIFS', 'GEMINI', 'NON_SIDEREAL'}

    assert expected.issubset(tags)


@pytest.mark.remote_data
def test_can_return_instrument():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))
    assert ad.phu['INSTRUME'] == 'NIFS'
    assert ad.instrument() == ad.phu['INSTRUME']


@pytest.mark.remote_data
def test_can_return_ad_length():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))
    assert len(ad) == 1


@pytest.mark.remote_data
def test_slice_range():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1:]

    assert len(slc) == 0

    for ext, md in zip(slc, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


@pytest.mark.remote_data
def test_read_a_keyword_from_hdr():
    ad = astrodata.open(download_from_archive(filename, path='NIFS'))

    try:
        assert ad.hdr['CCDNAME'] == 'NIFS'
    except KeyError:
        # KeyError only accepted if it's because headers out of range
        assert len(ad) == 1
