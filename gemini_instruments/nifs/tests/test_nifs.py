import pytest

import astrodata
import astrodata.testing
import gemini_instruments

test_files = [
    'N20160727S0077.fits',
]


@pytest.fixture(params=test_files)
def ad(request):
    filename = request.param
    path = astrodata.testing.download_from_archive(filename)
    return astrodata.from_file(path)


@pytest.mark.dragons_remote_data
def test_is_right_instance(ad):
    # YES, this *can* be different from test_is_right_type. Metaclasses!
    assert isinstance(ad, gemini_instruments.nifs.adclass.AstroDataNifs)


@pytest.mark.dragons_remote_data
def test_extension_data_shape(ad):
    data = ad[0].data
    assert data.shape == (2048, 2048)


@pytest.mark.dragons_remote_data
def test_tags(ad):
    tags = ad.tags
    expected = {'DARK', 'RAW', 'AT_ZENITH', 'NORTH', 'AZEL_TARGET',
                'CAL', 'UNPREPARED', 'NIFS', 'GEMINI', 'NON_SIDEREAL'}
    assert expected.issubset(tags)


@pytest.mark.dragons_remote_data
def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] == 'NIFS'
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
def test_read_a_keyword_from_hdr(ad):
    try:
        assert ad.hdr['CCDNAME'] == 'NIFS'
    except KeyError:
        # KeyError only accepted if it's because headers out of range
        assert len(ad) == 1
