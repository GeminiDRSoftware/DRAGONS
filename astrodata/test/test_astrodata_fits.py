
import os
import numpy as np
import pytest
import tempfile

import astrodata
import gemini_instruments

from astropy.io import fits
from astropy.table import Table


# naming all fits files for easier legibility in code
GRACES = "N20190116G0054i.fits"
GNIRS = 'N20190206S0279.fits'
GMOSN = "N20110826S0336.fits"
GMOSS = "S20180223S0229.fits"
NIFS = "N20160727S0077.fits"
NIRI = 'N20190120S0287.fits'
F2 = 'S20190213S0084.fits'
# GSAOI = 'S20170505S0188.fits'

TestFiles = [GRACES, GNIRS, GMOSN, GMOSS, NIFS, NIRI, F2]

# input_args = 1
# @pytest.mark.parametrize("filename, args_in, expected", [
#     GMOSN,  input_args, [False, True]),
#     (GMOSS, input_args, [False, True]),
#     (GNIRS, input_args, [False, True]),
#     (GSAOI, input_args, [False, True]),
#     (NIRI, input_args, [False, True]),
#     (F2, input_args, [False, True])
# ])
# def test_check_for_mos_mode(self, base_handler, filename, args_in, expected):
#     # Calling fixture with initial/default args input (inst_const.ONE)
#     ad, acq, args, handler = base_handler(filename, args_in[0])
#     # Since inst_const.ONE is the default args_in value, we don't actually
#     # have to pass it in the above base_handler, but for consistancy we will
#     try:
#         handler.check_for_mos_mode()
#     except:
#         raise Exception("Unknown exception raised")
#     else:
#         answer = handler.check_for_mos_mode()
#         assert (answer == expected[0])
#
#     # Re-call the fixture, this time with a different args input
#     ad, acq, args, handler = base_handler(filename, args_in[1])
#     try:
#         handler.check_for_mos_mode()
#     except:
#         raise Exception("Unknown exception raised")
#     else:
#         answer = handler.check_for_mos_mode()
#         assert (answer == expected[1])
#         # Notice how all these return expected[1] (True), while
#         # the ones above returned expected[0] (False)?




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
# @pytest.fixture()
# def base_handler(request):
#     print('setup create astrodata object')
#     def make_ad_object(filename, args=inst_const.ONE):
#         ad = astrodata.open(os.path.join(test_path, filename))
#         # Each instrument has it's own handler, so the long if statment
#         # just to easily distinguish them
#         if (filename == GMOSN) or (filename == GMOSS):
#             acq = GmosAcquisition(ad)
#         elif (filename == GNIRS):
#             acq = GnirsAcquisiton(ad)
#         elif (filename == GSAOI):
#             acq = GsaoiAcquisiton(ad)
#         elif (filename == NIRI):
#             acq = NiriAcquisiton(ad)
#         elif (filename == F2):
#             acq = F2Acquisiton(ad)
#
#
#         # creation of the handler to be passed back
#         handler = base.InstrumentHandler(args, acq)
#         def fin():
#             print('\nteardown after test')
#         request.addfinalizer(fin)
#         return  ad, acq, args, handler
#     return make_ad_object


@pytest.mark.usefixtures('setup_astrodatafits')
class TestAstrodataFits:

    @pytest.mark.parametrize("filename", TestFiles)
    def test_can_read_data(self, filename, test_path):
        test_data_full_name = os.path.join(test_path, filename)

        assert os.path.exists(test_data_full_name)

    @pytest.mark.parametrize("filename", TestFiles)
    def test_can_open_data(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        assert isinstance(ad, astrodata.fits.AstroDataFits)

    @pytest.mark.parametrize("filename", TestFiles)
    def test_filename_recognized(self, filename, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))
        adfilename = ad.filename
        assert adfilename == filename

    @pytest.mark.parametrize("filename", TestFiles)
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


    @pytest.mark.parametrize("filename", TestFiles)
    def test_extension_data_type(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data
        assert type(data) == np.ndarray


    @pytest.mark.parametrize("filename", TestFiles)
    def test_can_add_and_del_extension(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data
        assert type(data) == np.ndarray

    @pytest.mark.parametrize("filename", TestFiles)
    def test_iterate_over_extensions(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))
        for ext, md in zip(ad, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    @pytest.mark.parametrize("filename", TestFiles)
    def test_iterate_over_extensions(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        np.random.rand(50, 50)

    @pytest.mark.parametrize("filename", TestFiles)
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

    @pytest.mark.parametrize("filename", TestFiles)
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

    @pytest.mark.parametrize("filename", TestFiles)
    def test_iterate_over_single_slice(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = ('SCI', 1)

        for ext in ad[0]:
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    @pytest.mark.parametrize("filename", TestFiles)
    def test_slice_negative(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        assert ad.data[-1] is ad[-1].data



    @pytest.mark.parametrize("filename", TestFiles)
    def test_set_a_keyword_on_phu(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        ad.phu['DETECTOR'] = 'FooBar'
        ad.phu['ARBTRARY'] = 'BarBaz'

        assert ad.phu['DETECTOR'] == 'FooBar'
        assert ad.phu['ARBTRARY'] == 'BarBaz'

    @pytest.mark.parametrize("filename", [
        GRACES, GMOSN, GMOSS, NIFS   #Todo - Note: GNIRS/NIRI/F@ has no ['DETECTOR']
    ])
    
    def test_remove_a_keyword_from_phu(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        del ad.phu['DETECTOR']

        assert 'DETECTOR' not in ad.phu

    @pytest.mark.parametrize("filename", TestFiles)
    def test_writes_to_new_fits(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        test_file_location = os.path.join(test_path, 'write_to_new_fits_test_file.fits')
        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)
        assert os.path.exists(test_file_location)
        os.remove(test_file_location)


    @pytest.mark.parametrize("filename", TestFiles)
    def test_can_overwrite_existing_file(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        test_file_location = os.path.join(test_path, 'test_fits_overwrite.fits')
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
        test_file_location = os.path.join(test_path, 'created_fits_file.fits')
        if os.path.exists(test_file_location):
            os.remove(test_file_location)
        ad.write(test_file_location)
        assert os.path.exists(test_file_location)
        # Opens file again and tests data is same as above
        adnew = astrodata.open(test_file_location)
        assert np.array_equal((adnew[0].data), pixel_data)

    def test_can_append_table_and_access_data(self):

        my_astropy_table = Table(list(np.random.rand(2,100)),
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
            assert ad.hdr.CCDNAME == ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']

    @pytest.mark.skip(reason="Deprecated methods")
    def test_set_a_keyword_on_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AssertionError):
            ad.phu.DETECTOR = 'FooBar'
            ad.phu.ARBTRARY = 'BarBaz'
            assert ad.phu.DETECTOR == 'FooBar'
            assert ad.phu.ARBTRARY == 'BarBaz'
            assert ad.phu['DETECTOR'] == 'FooBar'

    @pytest.mark.skip(reason="Deprecated methods")
    def test_remove_a_keyword_from_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AttributeError):
            del ad.phu.DETECTOR
            assert 'DETECTOR' not in ad.phu

    # Regression:
    # Make sure that references to associated extension objects are copied across
    @pytest.mark.parametrize("filename", [
        (NIFS)
    ])
    
    def test_do_arith_and_retain_features(self, filename, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))

        ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
        ad2 = ad * 5

        np.testing.assert_array_almost_equal(ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)

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
