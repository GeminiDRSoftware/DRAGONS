import os
import pytest
import warnings

import astrodata
import gemini_instruments
from astrodata.test.conftest import test_path


try:
    path = os.environ['TEST_PATH']
except KeyError:
    path = ''

if not os.path.exists(path):
    path = ''


filename = 'N20110826S0336.fits'


@pytest.fixture(scope='class')
def setup_gmos(request):
    print('setup TestGMOS')

    def fin():
        print('\nteardown TestGMOS')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_gmos')
class TestF2:

    def test_is_right_type(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        assert type(ad) ==  gemini_instruments.gmos.adclass.AstroDataGmos

    def test_is_right_instance(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        # YES, this *can* be different from test_is_right_type. Metaclasses!
        assert isinstance(ad, gemini_instruments.gmos.adclass.AstroDataGmos)

    def test_extension_data_shape(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data

        assert data.shape == (2304, 1056)

    def test_tags(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        tags = ad.tags
        expected = {'RAW', 'GMOS', 'GEMINI', 'NORTH',
                    'SIDEREAL', 'UNPREPARED', 'SPECT', 'MOS'}

        assert expected.issubset(tags)

    def test_can_return_instrument(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        assert ad.phu['INSTRUME'] == 'GMOS-N'
        assert ad.instrument() == ad.phu['INSTRUME']

    def test_can_return_ad_length(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        assert len(ad) == 3


    def test_slice_range(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        metadata = ('SCI', 2), ('SCI', 3)
        slc = ad[1:]

        assert len(slc) == 2

        for ext, md in zip(slc, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


    # def test_read_a_keyword_from_phu(self, test_path):
    #
    #     ad = astrodata.open(os.path.join(test_path, filename))
    #     assert ad.phu['DETECTOR'] == 'F2'

    def test_read_a_keyword_from_hdr(self, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))

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
