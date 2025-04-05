
import pytest

import astrodata
import astrodata.testing
import gemini_instruments

filename = 'N20190120S0287.fits'


@pytest.fixture()
def ad():
    path = astrodata.testing.download_from_archive(filename)
    return astrodata.from_file(path)


@pytest.mark.xfail(reason="AstroFaker changes the AstroData factory")
@pytest.mark.dragons_remote_data
def test_is_right_type(ad):
    assert type(ad) == gemini_instruments.niri.adclass.AstroDataNiri


@pytest.mark.dragons_remote_data
def test_is_right_instance(ad):
    # YES, this *can* be different from test_is_right_type. Metaclasses!
    assert isinstance(ad, gemini_instruments.niri.adclass.AstroDataNiri)


@pytest.mark.dragons_remote_data
def test_extension_data_shape(ad):
    assert ad[0].data.shape == (1024, 1024)


@pytest.mark.dragons_remote_data
def test_tags(ad):
    tags = ad.tags
    expected = {'RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'UNPREPARED',
                'IMAGE', 'NIRI'}
    assert expected.issubset(tags)


@pytest.mark.dragons_remote_data
def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] == 'NIRI'
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
        assert ad.hdr['CCDNAME'] == 'NIRI'
    except KeyError:
        # KeyError only accepted if it's because headers out of range
        assert len(ad) == 1


def test_ra_dec_from_text(astrofaker):
    ad = astrofaker.create('NIRI', ['SPECT'],
                           extra_keywords={'RA': '03:48:30.113',
                                           'DEC': '+24:20:43.00',
                                           'DATE-OBS': '2021-01-01T12:00:00.000'}
                           )
    assert ad.target_ra() == pytest.approx(57.12547083333333)
    assert ad.target_dec() == pytest.approx(24.345277777777778)

    from astropy import units as u

