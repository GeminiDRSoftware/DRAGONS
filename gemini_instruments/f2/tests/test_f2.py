import os
import pytest
import warnings

import astrodata
import gemini_instruments
from astrodata.test.conftest import test_path


try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''


filename = 'S20190213S0084.fits'


@pytest.fixture(scope='class')
def setup_f2(request):
    print('setup TestF2')

    def fin():
        print('\nteardown TestF2')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_f2')
class TestF2:

    def test_is_right_type(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert type(ad) ==  gemini_instruments.f2.adclass.AstroDataF2

    def test_is_right_instance(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        # YES, this *can* be different from test_is_right_type. Metaclasses!
        assert isinstance(ad, gemini_instruments.f2.adclass.AstroDataF2)

    def test_extension_data_shape(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        data = ad[0].data

        assert data.shape == (1, 2048, 2048)

    def test_tags(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        tags = ad.tags
        expected = {'IMAGE', 'F2', 'RAW', 'SOUTH',
                    'SIDEREAL', 'UNPREPARED',
                    'GEMINI', 'ACQUISITION'}

        assert expected.issubset(tags)

    def test_can_return_instrument(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert ad.phu['INSTRUME'] == 'F2'
        assert ad.instrument() == ad.phu['INSTRUME']

    def test_can_return_ad_length(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert len(ad) == 1


    def test_slice_range(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        metadata = ('SCI', 2), ('SCI', 3)
        slc = ad[1:]

        assert len(slc) == 0

        for ext, md in zip(slc, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


    # def test_read_a_keyword_from_phu(self):
    #
    #     ad = astrodata.open(os.path.join(test_path(), filename))
    #     assert ad.phu['DETECTOR'] == 'F2'

    def test_read_a_keyword_from_hdr(self):

        ad = astrodata.open(os.path.join(test_path(), filename))

        try:
            assert ad.hdr['CCDNAME'] == 'F2'
        except KeyError:
            # KeyError only accepted if it's because headers out of range
            assert len(ad) == 1

        # with pytest.raises(AssertionError):
        #     ad.phu.DETECTOR = 'FooBar'
        #     ad.phu.ARBTRARY = 'BarBaz'
        #     assert ad.phu.DETECTOR == 'FooBar'
        #     assert ad.phu.ARBTRARY == 'BarBaz'
        #     assert ad.phu['DETECTOR'] == 'FooBar'
