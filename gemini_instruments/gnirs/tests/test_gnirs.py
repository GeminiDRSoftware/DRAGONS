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


filename = 'N20190206S0279.fits'


@pytest.fixture(scope='class')
def setup_gnirs(request):
    print('setup TestGNIRS')

    def fin():
        print('\nteardown TestGNIRS')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_gnirs')
class TestF2:

    def test_is_right_type(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert type(ad) ==  gemini_instruments.gnirs.adclass.AstroDataGnirs

    def test_is_right_instance(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        # YES, this *can* be different from test_is_right_type. Metaclasses!
        assert isinstance(ad, gemini_instruments.gnirs.adclass.AstroDataGnirs)

    def test_extension_data_shape(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        data = ad[0].data

        assert data.shape == (1022, 1024)

    def test_tags(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        tags = ad.tags
        expected = {'RAW', 'GEMINI', 'NORTH',
                    'SIDEREAL', 'GNIRS',
                    'UNPREPARED', 'SPECT', 'XD'}

        assert expected.issubset(tags)

    def test_can_return_instrument(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert ad.phu['INSTRUME'] == 'GNIRS'
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
    #     assert ad.phu['DETECTOR'] == 'GNIRS'

    def test_read_a_keyword_from_hdr(self):

        ad = astrodata.open(os.path.join(test_path(), filename))

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
