import os
import pytest
import glob

import astrodata
import gemini_instruments
from astrodata.test.conftest import test_path

# Cleans up a fake file created in the tests in case it's still there
cleanup = os.path.join(test_path(), 'created_fits_file.fits')
if os.path.exists(cleanup):
    os.remove(cleanup)

# test_path() needs to be exported as the path with
# all the reuired files you wish to test
testfiles = glob.glob(os.path.join(test_path(), "*.fits"))
filename = 'N20160727S0077.fits'


@pytest.fixture(scope='class')
def setup_nifs(request):
    print('setup Test_GRACES')

    def fin():
        print('\nteardown Test_GRACES')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_nifs')
class Test_GRACES:

    def test_is_right_type(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert type(ad) ==  gemini_instruments.nifs.adclass.AstroDataNifs

    def test_is_right_instance(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        # YES, this *can* be different from test_is_right_type. Metaclasses!
        assert isinstance(ad, gemini_instruments.nifs.adclass.AstroDataNifs)

    def test_extension_data_shape(self):

        testfiles = glob.glob(os.path.join(test_path(), "*.fits"))
        ad = astrodata.open(os.path.join(test_path(), filename))
        data = ad[0].data

        assert data.shape == (2048, 2048)

    def test_tags(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        tags = ad.tags
        expected = {'DARK', 'RAW', 'AT_ZENITH', 'NORTH', 'AZEL_TARGET',
                 'CAL', 'UNPREPARED', 'NIFS', 'GEMINI', 'NON_SIDEREAL'}

        assert expected.issubset(tags)

    def test_can_return_instrument(self):

        ad = astrodata.open(os.path.join(test_path(), filename))
        assert ad.phu['INSTRUME'] == 'NIFS'
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


    def test_read_a_keyword_from_hdr(self):

        ad = astrodata.open(os.path.join(test_path(), filename))

        try:
            assert ad.hdr['CCDNAME'] == 'NIFS'
        except KeyError:
            # KeyError only accepted if it's because headers out of range
            assert len(ad) == 1