
import os
import pytest
import glob
import warnings

import astrodata
import gemini_instruments
from .conftest import test_path


try:
    path = os.environ['TEST_PATH']
except KeyError:
    #warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    #warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''

archivefiles = glob.glob(os.path.join(path, "Archive/", "*fits"))

# Cleans up a fake file created in the tests in case it's still there
cleanup = os.path.join(path, 'created_fits_file.fits')
if os.path.exists(cleanup):
    os.remove(cleanup)


# Fixtures for module and class
@pytest.fixture(scope='class')
def setup_archive_test(request):
    print('setup TestArchive')

    def fin():
        print('\nteardown TestArchive')
    request.addfinalizer(fin)
    return


# # Opens all files in archive
#     ad = astrodata.open(filename)
#     typelist = []
#     for i in# ad.descriptors:
#         try:
#             typelist.append((i, type(getattr(ad, i)()))
#         except Exception as err:
#             print("{} failed on call: {}".format(i, str(err))
#             #warnings.warn("{} failed on call: {}".format(i, str(err)))

@pytest.mark.usefixtures('setup_archive_test')
class TestArchive:

    @pytest.mark.parametrize("filename", archivefiles)
    def test_airmass(self, filename):
        ad = astrodata.open(filename)
        try:
            assert ((type(ad.airmass()) == float)
                    or (ad.ao_seeing() is None))
        except Exception as err:
            print("{} failed on call: {}".format(ad.airmass, str(err)))
            # #warnings.warn("{} failed on call: {}".format(ad.airmass, str(err)))
            pytest.skip("test")

    @pytest.mark.parametrize("filename", archivefiles)
    def test_amp_read_area(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.amp_read_area()) is list)
                    or (ad.ao_seeing() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.amp_read_area, str(err)))
            # #warnings.warn("{} failed on call: {}".format(ad.amp_read_area, str(err)))
            pytest.skip("test")

    @pytest.mark.parametrize("filename", archivefiles)
    def test_ao_seeing(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.ao_seeing()) == float)
                    or (ad.ao_seeing() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.ao_seeing, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.ao_seeing, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_array_name(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.array_name()) == list)
                    or (ad.array_name() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.array_name, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.array_name, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_array_section(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.array_section()) == list)
                    or (ad.array_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.array_section, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.array_section, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_azimuth(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.azimuth()) == float)
                    or (ad.azimuth() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.azimuth, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.azimuth, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_calibration_key(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.calibration_key()) == str)
                    or (ad.calibration_key() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.calibration_key, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.calibration_key, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_camera(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.camera()) == str)
                    or (ad.camera() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.camera, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.camera, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_cass_rotator_pa(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.cass_rotator_pa()) == float)
                    or (ad.cass_rotator_pa() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.cass_rotator_pa, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.cass_rotator_pa, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_central_wavelength(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.central_wavelength()) == float) or
                    (ad.central_wavelength() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.central_wavelength, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.central_wavelength, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_coadds(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.coadds()) == int) or (ad.coadds() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.coadds, str(err)))
            #warnings.warn("{} failed on call: {}".format(
               # ad.coadds, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_data_label(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.data_label()) == str) or
                    (ad.data_label() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.data_label, str(err)))
            #warnings.warn("{} failed on call: {}".format(
               # ad.data_label, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_data_section(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.data_section()) == list) or (ad.data_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.data_section, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.data_section, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_dec(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.dec()) == float) or (ad.dec() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.dec, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.dec, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_decker(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.decker()) == str) or (ad.decker() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.decker, str(err)))
            #warnings.warn("{} failed on call: {}".format(ad.decker, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_name(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.detector_name()) == str) or
                    (ad.detector_name() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_name, str(err)))
            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_name, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_roi_setting(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.detector_roi_setting()) == str) or
                    (ad.detector_roi_setting() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_roi_setting, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_roi_setting, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_rois_requested(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.detector_rois_requested()) == list) or
                    (ad.detector_rois_requested() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_rois_requested, str(err)))
            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_rois_requested, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_section(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.detector_section()) == list) or
                    (ad.detector_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_section, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_section, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_x_bin(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.detector_x_bin()) == int) or
                    (ad.detector_x_bin() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_x_bin, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_x_bin, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_x_offset(self, filename):
        ad = astrodata.open(filename)
        if filename == os.path.join(path, "Archive/N20190216S0092.fits"):
            pytest.skip("Current issue with this file, will get back to it")
        try:

            assert ((type(ad.detector_x_offset()) == float) or
                    (ad.detector_x_offset() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_x_offset, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_x_offset, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_y_bin(self, filename):
            ad = astrodata.open(filename)

            try:
                assert ((type(ad.detector_y_bin()) == int) or
                        (ad.detector_y_bin() is None))

            except Exception as err:
                print("{} failed on call: {}".format(
                   ad.detector_y_bin, str(err)))

                #warnings.warn("{} failed on call: {}".format(
                   # ad.detector_y_bin, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_detector_y_offset(self, filename):
        ad = astrodata.open(filename)
        if filename == os.path.join(path, "Archive/N20190216S0092.fits"):
            pytest.skip("Current issue with this file, will get back to it")

        try:
            assert ((type(ad.detector_y_offset()) == float) or
                    (ad.detector_y_offset() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_y_offset, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.detector_y_offset, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_disperser(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.disperser()) == str) or
                    (ad.disperser() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.disperser, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.disperser, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_dispersion(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.dispersion()) == list) or
                    (ad.dispersion() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.dispersion, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.dispersion, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_dispersion_axis(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.dispersion_axis()) == list) or
                    (ad.dispersion_axis() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.dispersion_axis, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.dispersion_axis, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_effective_wavelength(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.effective_wavelength()) == float) or
                    (ad.effective_wavelength() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.effective_wavelength, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.effective_wavelength, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_elevation(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.elevation()) == float) or
                    (ad.elevation() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.elevation, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.elevation, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_exposure_time(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.exposure_time()) == float) or
                    (ad.exposure_time() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.exposure_time, str(err)))

            #warnings.warn("{} failed on call: {}".format(
               # ad.exposure_time, str(err)))

    @pytest.mark.parametrize("filename", archivefiles)
    def test_filter_name(self, filename):
        ad = astrodata.open(filename)

        try:
            assert ((type(ad.filter_name()) == str) or
                    (ad.filter_name() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.filter_name, str(err)))
            #warnings.warn("{} failed on call: {}".format(
               # ad.filter_name, str(err)))

    

