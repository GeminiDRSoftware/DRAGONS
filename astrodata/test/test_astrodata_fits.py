
import os
import numpy as np
import pytest
import tempfile
import glob

import astrodata
import gemini_instruments
from .conftest import test_path

from astropy.io import fits
from astropy.table import Table


# naming all fits files for easier legibility in code
# GRACES = "N20190116G0054i.fits"
# GNIRS = 'N20190206S0279.fits'
# GMOSN = "N20110826S0336.fits"
# GMOSS = "S20180223S0229.fits"
# NIFS = "N20160727S0077.fits"
# NIRI = 'N20190120S0287.fits'
# F2 = 'S20190213S0084.fits'
# GSAOI = 'S20170505S0188.fits'

#testfiles = [hello, GRACES, GNIRS, GMOSN, GMOSS, NIFS, NIRI, F2]
#
# buffer = open('../../.jenkins/test_files.txt', 'r')
# list_of_files = buffer.readlines()
# # glob.glob(.jenkins/test_files.txt, )

cleanup = os.path.join(test_path(), 'created_fits_file.fits')
if os.path.exists(cleanup):
    os.remove(cleanup)

testfiles = glob.glob(os.path.join(test_path(), "*.fits"))




# Fixtures for module and class
@pytest.fixture(scope='module')
def setup_module(request):
    print('setup test_astrodata_fits module')

    def fin():
        print('\nteardown test_astrodata_fits module')
    request.addfinalizer(fin)
    return


@pytest.fixture(scope='class')
def setup_astrodatafits(request):
    print('setup TestAstrodataFits')

    def fin():
        print('\nteardown TestAstrodataFits')
    request.addfinalizer(fin)
    return

#

@pytest.mark.usefixtures('setup_astrodatafits')
class TestAstrodataFits:

    @pytest.mark.parametrize("filename", testfiles)
    def test_can_read_data(self, filename):


        assert os.path.exists(filename)

    @pytest.mark.parametrize("filename", testfiles)
    def test_can_open_data(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        assert isinstance(ad, astrodata.fits.AstroDataFits)

    @pytest.mark.parametrize("filename", testfiles)
    def test_filename_recognized(self, filename, test_path):

        ad = astrodata.open(filename)
        filename = os.path.split(filename)[-1]
        assert ad.filename == filename

    @pytest.mark.parametrize("filename", testfiles)
    def test_can_add_and_del_extension(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        ourarray = np.array([(1, 2, 3),
                             (11, 12, 13),
                             (21, 22, 23)])
        original_index = len(ad)
        ad.append(ourarray)
        assert len(ad) == (original_index + 1)
        del ad[original_index]
        assert len(ad) == original_index


    @pytest.mark.parametrize("filename", testfiles)
    def test_extension_data_type(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data
        assert type(data) == np.ndarray


    @pytest.mark.parametrize("filename", testfiles)
    def test_can_add_and_del_extension(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data
        assert type(data) == np.ndarray

    @pytest.mark.parametrize("filename", testfiles)
    def test_iterate_over_extensions(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))
        for ext, md in zip(ad, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    @pytest.mark.parametrize("filename", testfiles)
    def test_iterate_over_extensions(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        np.random.rand(50, 50)

    @pytest.mark.parametrize("filename", testfiles)
    def test_slice_multiple(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        metadata = ('SCI', 2), ('SCI', 3)

        try:
            slc = ad[1, 2]
        except IndexError:
            assert len(ad) == 1
        else:
            assert len(slc) == 2
            for ext, md in zip(slc, metadata):
                assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    @pytest.mark.parametrize("filename", testfiles)
    def test_slice_single(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        try:
            metadata = ('SCI', 2)
            ext = ad[1]
        except IndexError:
            # Make sure IndexError is due to ad being to short
            assert len(ad) == 1
        else:
            assert ext.is_single
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    @pytest.mark.parametrize("filename", testfiles)
    def test_iterate_over_single_slice(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = ('SCI', 1)

        for ext in ad[0]:
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    @pytest.mark.parametrize("filename", testfiles)
    def test_slice_negative(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        assert ad.data[-1] is ad[-1].data



    @pytest.mark.parametrize("filename", testfiles)
    def test_set_a_keyword_on_phu(self, filename, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))

        ad.phu['DETECTOR'] = 'FooBar'
        ad.phu['ARBTRARY'] = 'BarBaz'

        assert ad.phu['DETECTOR'] == 'FooBar'
        assert ad.phu['ARBTRARY'] == 'BarBaz'


    @pytest.mark.parametrize("filename", testfiles)
    def test_remove_a_keyword_from_phu(self, filename, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        exceptions = ['GNIRS', 'NIRI', 'F2']

        try:
            del ad.phu['DETECTOR']
            assert 'DETECTOR' not in ad.phu

        except KeyError:
            if ad.instrument() in exceptions:
                pass
            else:
                raise KeyError


    @pytest.mark.parametrize("filename", testfiles)
    def test_writes_to_new_fits(self, filename, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        test_file_location = os.path.join(test_path,
                                          'write_to_new_fits_test_file.fits')
        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)

        assert os.path.exists(test_file_location)

        os.remove(test_file_location)


    @pytest.mark.parametrize("filename", testfiles)
    def test_can_overwrite_existing_file(self, filename, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        test_file_location = os.path.join(test_path,
                                          'test_fits_overwrite.fits')
        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)

        assert os.path.exists(test_file_location)

        adnew = astrodata.open(test_file_location)
        adnew.write(overwrite=True)

        # erasing file for cleanup
        os.remove(test_file_location)

    def test_can_make_and_write_ad_object(self, test_path):

        # Creates data and ad object
        phu = fits.PrimaryHDU()
        pixel_data = np.random.rand(100, 100)

        hdu = fits.ImageHDU()
        hdu.data = pixel_data

        ad = astrodata.create(phu)
        ad.append(hdu, name='SCI')

        # Write file and test it exists properly
        test_file_location = os.path.join(test_path,
                                          'created_fits_file.fits')
        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)

        assert os.path.exists(test_file_location)
        # Opens file again and tests data is same as above

        adnew = astrodata.open(test_file_location)
        assert np.array_equal(adnew[0].data, pixel_data)

    def test_can_append_table_and_access_data(self):

        my_astropy_table = Table(list(np.random.rand(2, 100)),
                                 names=['col1', 'col2'])

        phu = fits.PrimaryHDU()
        ad = astrodata.create(phu)
        astrodata.add_header_to_table(my_astropy_table)

        ad.append(my_astropy_table, name='BOB')

        print(ad.info())

    # Access to headers: DEPRECATED METHODS
    # These should fail at some point
    @pytest.mark.skip(reason="Deprecated methods")
    def test_read_a_keyword_from_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AttributeError):
            assert ad.phu.DETECTOR == 'GMOS + Red1'

    @pytest.mark.skip(reason="Deprecated methods")
    def test_read_a_keyword_from_hdr_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AttributeError):
            assert ad.hdr.CCDNAME == ['EEV 9273-16-03',
                                      'EEV 9273-20-04',
                                      'EEV 9273-20-03']

    @pytest.mark.skip(reason="Deprecated methods")
    def test_set_a_keyword_on_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')

        with pytest.raises(AssertionError):
            ad.phu.DETECTOR = 'FooBar'
            ad.phu.ARBTRARY = 'BarBaz'

            assert ad.phu.DETECTOR == 'FooBar'
            assert ad.phu.ARBTRARY == 'BarBaz'
            assert ad.phu['DETECTOR'] == 'FooBar'


    # Regression:
    # Make sure that references to associated
    # extension objects are copied across
    @pytest.mark.parametrize("filename", testfiles)
    def test_do_arith_and_retain_features(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
        ad2 = ad * 5

        np.testing.assert_array_almost_equal(ad[0].NEW_FEATURE,
                                             ad2[0].NEW_FEATURE)

    # Trying to access a missing attribute in the data provider should raise an
    # AttributeError
        test_data_name = "N20110826S0336.fits"
    @pytest.mark.skip(reason="uses chara")
    def test_raise_attribute_error_when_accessing_missing_extenions(self):
        ad = from_chara('N20131215S0202_refcatAdded.fits')
        with pytest.raises(AttributeError) as excinfo:
            ad.ABC

    # Some times, internal changes break the writing capability. Make sure that
    # this is the case, always
    @pytest.mark.skip(reason="uses chara")
    def test_write_without_exceptions(self):
        # Use an image that we know contains complex structure
        ad = from_chara('N20131215S0202_refcatAdded.fits')
        with tempfile.TemporaryFile() as tf:
            ad.write(tf)

    @pytest.mark.parametrize("filename", testfiles)
    def test_descriptor_is_int(self, filename, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        ad_int = ["group_id"]
        # ad_int = ['id', 'diskfile_id', 'ut_datetime_secs', 'ra', 'dec',
        #           'azimuth', 'elevation', 'cass_rotator_pa', 'airmass',
        #           'exposure_time', 'central_wavelength', 'coadds',
        #           'raw_iq', 'raw_cc', 'raw_wv', 'raw_bg', 'requested_iq',
        #           'requested_cc', 'requested_wv', 'requested_bg']


        for integer in ad_int:
            assert type(getattr(ad, integer)()) == str
