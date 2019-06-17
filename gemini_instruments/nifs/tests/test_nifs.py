import os

import astrodata
import gemini_instruments

filename = 'N20160727S0077.fits'


def test_is_right_instance(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))
    # YES, this *can* be different from test_is_right_type. Metaclasses!
    assert isinstance(ad, gemini_instruments.nifs.adclass.AstroDataNifs)


def test_extension_data_shape(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))
    data = ad[0].data

    assert data.shape == (2048, 2048)


def test_tags(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))
    tags = ad.tags
    expected = {'DARK', 'RAW', 'AT_ZENITH', 'NORTH', 'AZEL_TARGET',
                'CAL', 'UNPREPARED', 'NIFS', 'GEMINI', 'NON_SIDEREAL'}

    assert expected.issubset(tags)


def test_can_return_instrument(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))
    assert ad.phu['INSTRUME'] == 'NIFS'
    assert ad.instrument() == ad.phu['INSTRUME']


def test_can_return_ad_length(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))
    assert len(ad) == 1


def test_slice_range(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1:]

    assert len(slc) == 0

    for ext, md in zip(slc, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


def test_read_a_keyword_from_hdr(input_test_path):
    ad = astrodata.open(os.path.join(input_test_path, filename))

    try:
        assert ad.hdr['CCDNAME'] == 'NIFS'
    except KeyError:
        # KeyError only accepted if it's because headers out of range
        assert len(ad) == 1
