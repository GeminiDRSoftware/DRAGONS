import os
import pytest

import astrodata
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
def ad(cache_file_from_archive, request):
    filename = request.param
    path = cache_file_from_archive(filename)
    return astrodata.open(path)


@pytest.mark.xfail(reason="AstroFaker changes the AstroData factory")
def test_is_right_type(ad):
    assert type(ad) == gemini_instruments.gnirs.adclass.AstroDataGnirs


def test_is_right_instance(ad):
    # YES, this *can* be different from test_is_right_type. Metaclasses!
    assert isinstance(ad, gemini_instruments.gnirs.adclass.AstroDataGnirs)


def test_extension_data_shape(ad):
    data = ad[0].data
    assert data.shape == (1022, 1024)


expected_tags = {'RAW', 'GEMINI', 'NORTH', 'GNIRS', 'UNPREPARED', }


@pytest.mark.parametrize("tag", expected_tags)
def test_tags(ad, tag):
    assert tag in ad.tags


def test_can_return_instrument(ad):
    assert ad.phu['INSTRUME'] == 'GNIRS'
    assert ad.instrument() == ad.phu['INSTRUME']


def test_can_return_ad_length(ad):
    assert len(ad) == 1


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


@pytest.mark.parametrize("descriptor,expected_type", GNIRS_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(ad, descriptor, expected_type):
    value = getattr(ad, descriptor)()
    assert isinstance(value, expected_type) or value is None, \
        "Assertion failed for file: {}".format(ad.filename)
