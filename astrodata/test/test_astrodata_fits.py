
import os
import numpy as np
import pytest
import tempfile
import glob
import warnings

import astrodata
import gemini_instruments
from .conftest import test_path

from astropy.io import fits
from astropy.table import Table


try:
    path = os.environ['TEST_PATH']
except KeyError:
    path = ''

if not os.path.exists(path):
    path = ''

# Returns list of all files in the TEST_PATH directory
files_with_directory = glob.glob(os.path.join(path, "*fits"))

# Separates the directory from the list, helps cleanup code
fits_files = [os.path.split(_file)[-1] for _file in files_with_directory]

# Cleans up a fake file created in the tests in case it's still there
cleanup = os.path.join(path, 'created_fits_file.fits')
if os.path.exists(cleanup):
    os.remove(cleanup)


# Fixtures for module and class
@pytest.fixture(scope='class')
def setup_astrodatafits(request):
    print('setup TestAstrodataFits')

    def fin():
        print('\nteardown TestAstrodataFits')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_astrodatafits')
class TestAstrodataFits:

    @pytest.mark.parametrize("filename", fits_files)
    def test_filename_exists(self, test_path, filename):
        assert os.path.exists(os.path.join(test_path, filename))

    # TODO: Replace this so tests reads data, not just checks if exists
    @pytest.mark.parametrize("filename", fits_files)
    def test_can_read_data(self, test_path, filename):
        assert os.path.exists(os.path.join(test_path, filename))

    @pytest.mark.parametrize("filename", fits_files)
    def test_can_open_data(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))

        assert isinstance(ad, astrodata.fits.AstroDataFits)

    @pytest.mark.parametrize("filename", fits_files)
    def test_filename_recognized(self, test_path, filename):

        ad = astrodata.open(os.path.join(test_path, filename))
        filename = os.path.split(filename)[-1]
        assert ad.filename == filename

    @pytest.mark.parametrize("filename", fits_files)
    def test_can_add_and_del_extension(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))
        ourarray = np.array([(1, 2, 3),
                             (11, 12, 13),
                             (21, 22, 23)])

        original_index = len(ad)
        ad.append(ourarray)

        assert len(ad) == (original_index + 1)

        del ad[original_index]
        assert len(ad) == original_index

    @pytest.mark.parametrize("filename", fits_files)
    def test_extension_data_type(self, test_path, filename):

        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data

        assert type(data) == np.ndarray

    @pytest.mark.parametrize("filename", fits_files)
    def test_can_add_and_del_extension(self, test_path, filename):

        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data

        assert type(data) == np.ndarray

    @pytest.mark.parametrize("filename", fits_files)
    def test_iterate_over_extensions(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))
        for ext, md in zip(ad, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    # @pytest.mark.parametrize("filename", fits_files)
    # def test_iterate_over_extensions(self, test_path, filename):
    #     ad = astrodata.open(os.path.join(test_path, filename))
    #     np.random.rand(50, 50)

    @pytest.mark.parametrize("filename", fits_files)
    def test_slice_multiple(self, test_path, filename):
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

    @pytest.mark.parametrize("filename", fits_files)
    def test_slice_single(self, test_path, filename):
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

    @pytest.mark.parametrize("filename", fits_files)
    def test_iterate_over_single_slice(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = ('SCI', 1)

        for ext in ad[0]:
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    @pytest.mark.parametrize("filename", fits_files)
    def test_slice_negative(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))

        assert ad.data[-1] is ad[-1].data

    @pytest.mark.parametrize("filename", fits_files)
    def test_set_a_keyword_on_phu(self, test_path, filename):

        ad = astrodata.open(os.path.join(test_path, filename))

        ad.phu['DETECTOR'] = 'FooBar'
        ad.phu['ARBTRARY'] = 'BarBaz'

        assert ad.phu['DETECTOR'] == 'FooBar'
        assert ad.phu['ARBTRARY'] == 'BarBaz'

    @pytest.mark.parametrize("filename", fits_files)
    def test_remove_a_keyword_from_phu(self, test_path, filename):

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

    @pytest.mark.parametrize("filename", fits_files)
    def test_writes_to_new_fits(self, test_path, filename):

        ad = astrodata.open(os.path.join(test_path, filename))
        test_file_location = os.path.join(test_path,
                                          'write_to_new_fits_test_file.fits')
        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)

        assert os.path.exists(test_file_location)

        os.remove(test_file_location)

    @pytest.mark.parametrize("filename", fits_files)
    def test_can_overwrite_existing_file(self, test_path, filename):

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
        test_file_location = os.path.join(
            test_path, 'created_fits_file.fits')

        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)

        assert os.path.exists(test_file_location)
        # Opens file again and tests data is same as above

        adnew = astrodata.open(test_file_location)
        assert np.array_equal(adnew[0].data, pixel_data)
        os.remove(test_file_location)

    def test_can_append_table_and_access_data(self):

        my_astropy_table = Table(list(np.random.rand(2, 100)),
                                 names=['col1', 'col2'])

        phu = fits.PrimaryHDU()
        ad = astrodata.create(phu)
        astrodata.add_header_to_table(my_astropy_table)

        ad.append(my_astropy_table, name='BOB')

        print(ad.info())

    @pytest.mark.parametrize("filename", fits_files)
    def test_set_a_keyword_on_phu_deprecated(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))

        try:
            with pytest.raises(AssertionError):
                ad.phu.DETECTOR = 'FooBar'
                ad.phu.ARBTRARY = 'BarBaz'

                assert ad.phu.DETECTOR == 'FooBar'
                assert ad.phu.ARBTRARY == 'BarBaz'
                assert ad.phu['DETECTOR'] == 'FooBar'

        except KeyError as e:

            # Some instruments don't have DETECTOR as a keyword
            if e.args[0] == "Keyword 'DETECTOR' not found.":
                pass
            else:
                raise KeyError

    # Regression:
    # Make sure that references to associated
    # extension objects are copied across
    @pytest.mark.parametrize("filename", fits_files)
    def test_do_arith_and_retain_features(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, filename))

        ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
        ad2 = ad * 5

        np.testing.assert_array_almost_equal(ad[0].NEW_FEATURE,
                                             ad2[0].NEW_FEATURE)

    #

    # ########################################################################################333
    @pytest.mark.skip(reason="Deprecated methods")
    def test_remove_a_keyword_from_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AttributeError):
            del ad.phu.DETECTOR
            assert 'DETECTOR' not in ad.phu

        # Regression:
        # Make sure that references to associated
        # extension objects are copied across
        # Trying to access a missing attribute in
        # the data provider should raise an
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
            assert ad.hdr.CCDNAME == [
                'EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03'
            ]

