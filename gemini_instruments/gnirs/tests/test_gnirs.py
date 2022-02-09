import pytest

import astrodata
import astrodata.testing
import gemini_instruments

test_files = [
    "N20190206S0279.fits",
    "N20190214S0058.fits",
    "N20190216S0060.fits",
    "N20190216S0092.fits",
    "N20190221S0032.fits",
]


GNIRS_DESCRIPTORS_TYPES = [
    ('detector_x_offset', float),
    ('detector_y_offset', float),
    ('pixel_scale', float),
]


@pytest.fixture(params=test_files)
def ad(request):
    """
    Fixture that will download a file from specified as a test parameter,
    open it as an AstroData object and return it.

    Parameters
    ----------
    request : fixture
        Pytest built-in fixture containing information about the parent test.

    Returns
    -------
    AstroData
        Raw file downloaded from the archive and cached locally.
    """
    filename = request.param
    path = astrodata.testing.download_from_archive(filename)
    return astrodata.open(path)


@pytest.mark.xfail(reason="AstroFaker changes the AstroData factory")
@pytest.mark.dragons_remote_data
def test_is_right_type(ad):
    assert type(ad) == gemini_instruments.gnirs.adclass.AstroDataGnirs


@pytest.mark.dragons_remote_data
def test_is_right_instance(ad):
    # YES, this *can* be different from test_is_right_type. Metaclasses!
    assert isinstance(ad, gemini_instruments.gnirs.adclass.AstroDataGnirs)


@pytest.mark.dragons_remote_data
def test_extension_data_shape(ad):
    data = ad[0].data
    assert data.shape == (1022, 1024)


expected_tags = {'RAW', 'GEMINI', 'NORTH', 'GNIRS', 'UNPREPARED', }


@pytest.mark.parametrize("tag", expected_tags)
@pytest.mark.dragons_remote_data
def test_tags(ad, tag):
    assert tag in ad.tags


@pytest.mark.dragons_remote_data
def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] == 'GNIRS'
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


# def test_read_a_keyword_from_phu(path_to_inputs):
#
#     ad = astrodata.open(os.path.join(path_to_inputs, filename))
#     assert ad.phu['DETECTOR'] == 'GNIRS'

@pytest.mark.dragons_remote_data
def test_read_a_keyword_from_hdr(ad):
    try:
        assert ad.hdr['CCDNAME'] == ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']
    except KeyError:
        # KeyError only accepted if it's because headers out of range
        assert len(ad) == 1

    # with pytest.raises(AssertionError):
    #     ad.phu.DETECTOR = 'FooBar'
    #     ad.phu.ARBTRARY = 'BarBaz'
    #     assert ad.phu.DETECTOR == 'FooBar'
    #     assert ad.phu.ARBTRARY == 'BarBaz'
    #     assert ad.phu['DETECTOR'] == 'FooBar'


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("descriptor,expected_type", GNIRS_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(ad, descriptor, expected_type):
    value = getattr(ad, descriptor)()
    assert isinstance(value, expected_type) or value is None, \
        "Assertion failed for file: {}".format(ad.filename)


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("ad", ["N20190101S0102.fits"], indirect=True)
def test_ra_and_dec_always_returns_float(ad, monkeypatch):
    """
    Tests that the get the RA/DEC coordinates using descriptors.

    Parameters
    ----------
    ad : fixture
        Custom fixture that downloads and opens the input file.
    """
    if isinstance(ad.wcs_ra(), float) or ad.wcs_ra() is None:
        assert isinstance(ad.ra(), float)

    if isinstance(ad.wcs_dec(), float) or ad.wcs_dec() is None:
        assert isinstance(ad.dec(), float)


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("ad", ["N20190101S0102.fits"], indirect=True)
def test_ra_and_dec_wcs_fallback(ad, monkeypatch):
    """
    Tests that the get the RA/DEC coordinates using descriptors when the WCS coordinate mapping fails

    Parameters
    ----------
    ad : fixture
        Custom fixture that downloads and opens the input file.
    """
    def _fake_wcs_call():
        return None

    monkeypatch.setattr(ad, 'wcs_ra', _fake_wcs_call)
    monkeypatch.setattr(ad, 'wcs_dec', _fake_wcs_call)
    assert(ad.ra() == ad.phu.get('RA', None))
    assert(ad.dec() == ad.phu.get('DEC', None))


def test_ra_dec_from_text(astrofaker):
    ad = astrofaker.create('GNIRS', ['IMAGE'],
                           extra_keywords={'RA': '03:48:30.113',
                                           'DEC': '+24:20:43.00',
                                           'DATE-OBS': '2021-01-01T12:00:00.000'}
                           )
    assert ad.ra() == 57.12547083333333
    assert ad.dec() == 24.345277777777778


if __name__ == "__main__":
    pytest.main()
