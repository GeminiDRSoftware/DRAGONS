import pytest
import astrodata
import gemini_instruments

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
def ad(cache_file_from_archive, request):
    filename = request.param
    path = cache_file_from_archive(filename)
    return astrodata.open(path)


def test_is_right_instance(ad):
    assert isinstance(ad, gemini_instruments.gmos.adclass.AstroDataGmos)


def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] in ['GMOS-N', 'GMOS-S']
    assert ad.instrument() == ad.phu['INSTRUME']


def test_can_return_ad_length(ad):
    assert len(ad)


@pytest.mark.parametrize("descriptor,expected_type", GMOS_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(ad, descriptor, expected_type, gmos_files):
    value = getattr(ad, descriptor)()
    assert isinstance(value, expected_type) or value is None, \
        "Assertion failed for file: {}".format(ad.filename)


if __name__ == '__main__':

    pytest.main()
