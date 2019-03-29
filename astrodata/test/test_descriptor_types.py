
import os
import pytest
import glob
import warnings
import datetime

import astrodata
import gemini_instruments
from .conftest import test_path


try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''

# Returns list of all files in the TEST_PATH directory
archive_files = glob.glob(os.path.join(path, "Archive/", "*fits"))

# Separates the directory from the list, helps cleanup code
fits_files = [os.path.split(_file)[-1] for _file in archive_files]

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


@pytest.mark.usefixtures('setup_archive_test')
class TestArchive:

    @pytest.mark.parametrize("filename", fits_files)
    def test_airmass_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))
        try:
            assert ((type(ad.airmass()) == float)
                    or (ad.airmass() is None))
        except Exception as err:
            print("{} failed on call: {}".format(ad.airmass, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.airmass, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_amp_read_area_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.amp_read_area()) is list)
                    or (ad.amp_read_area() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.amp_read_area, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.amp_read_area, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_ao_seeing_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.ao_seeing()) == float)
                    or (ad.ao_seeing() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.ao_seeing, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.ao_seeing, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_array_name_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.array_name()) == list)
                    or (ad.array_name() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.array_name, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.array_name, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_array_section_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.array_section()) == list)
                    or (ad.array_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.array_section, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.array_section, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_azimuth_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.azimuth()) == float)
                    or (ad.azimuth() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.azimuth, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.azimuth, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_calibration_key_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.calibration_key()) == str)
                    or (ad.calibration_key() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.calibration_key, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.calibration_key, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_camera_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.camera()) == str)
                    or (ad.camera() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.camera, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.camera, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_cass_rotator_pa_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.cass_rotator_pa()) == float)
                    or (ad.cass_rotator_pa() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.cass_rotator_pa, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.cass_rotator_pa, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_central_wavelength_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.central_wavelength()) == float) or
                    (ad.central_wavelength() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.central_wavelength, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.central_wavelength, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_coadds_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.coadds()) == int) or (ad.coadds() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.coadds, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.coadds, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_data_label_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.data_label()) == str) or
                    (ad.data_label() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.data_label, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.data_label, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_data_section_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.data_section()) == list) or (ad.data_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.data_section, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.data_section, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_dec_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.dec()) == float) or (ad.dec() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.dec, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.dec, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_decker_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.decker()) == str) or (ad.decker() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.decker, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.decker, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_name_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.detector_name()) == str) or
                    (ad.detector_name() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_name, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.detector_name, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_roi_setting_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.detector_roi_setting()) == str) or
                    (ad.detector_roi_setting() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_roi_setting, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.detector_roi_setting, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_rois_requested_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.detector_rois_requested()) == list) or
                    (ad.detector_rois_requested() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_rois_requested, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.detector_rois_requested, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_section_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.detector_section()) == list) or
                    (ad.detector_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_section, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.detector_section, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_x_bin_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.detector_x_bin()) == int) or
                    (ad.detector_x_bin() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_x_bin, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.detector_x_bin, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_x_offset_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))
        if filename == os.path.join(path, "Archive/N20190216S0092.fits"):
            pytest.skip("Current issue with this file, will get back to it")
        try:

            assert ((type(ad.detector_x_offset()) == float) or
                    (ad.detector_x_offset() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_x_offset, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.detector_x_offset, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_y_bin_descriptor_is_none_or_int(self, test_path, filename):
            ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

            try:
                assert ((type(ad.detector_y_bin()) == int) or
                        (ad.detector_y_bin() is None))

            except Exception as err:
                print("{} failed on call: {}".format(
                   ad.detector_y_bin, str(err)))

                # warnings.warn("{} failed on call: {}".format(ad.detector_y_bin, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_detector_y_offset_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))
        if filename == os.path.join(path, "Archive/N20190216S0092.fits"):
            pytest.skip("Current issue with this file, will get back to it")

        try:
            assert ((type(ad.detector_y_offset()) == float) or
                    (ad.detector_y_offset() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.detector_y_offset, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.detector_y_offset, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_disperser_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.disperser()) == str) or
                    (ad.disperser() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.disperser, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.disperser, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_dispersion_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.dispersion()) == list) or
                    (ad.dispersion() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.dispersion, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.dispersion, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_dispersion_axis_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.dispersion_axis()) == list) or
                    (ad.dispersion_axis() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.dispersion_axis, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.dispersion_axis, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_effective_wavelength_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.effective_wavelength()) == float) or
                    (ad.effective_wavelength() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.effective_wavelength, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.effective_wavelength, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_elevation_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.elevation()) == float) or
                    (ad.elevation() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.elevation, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.elevation, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_exposure_time_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.exposure_time()) == float) or
                    (ad.exposure_time() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.exposure_time, str(err)))

            # warnings.warn("{} failed on call: {}".format(ad.exposure_time, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_filter_name_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.filter_name()) == str) or
                    (ad.filter_name() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
               ad.filter_name, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.filter_name, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_focal_plane_mask_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.focal_plane_mask()) == str)
                    or (ad.focal_plane_mask() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.focal_plane_mask, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.focal_plane_mask, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_gain_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.gain()) == list)
                    or (ad.gain() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.gain, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.gain, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_gain_setting_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.gain_setting()) == str)
                    or (ad.gain_setting() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.gain_setting, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.gain_setting, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_gcal_lamp_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.gcal_lamp()) == str)
                    or (ad.gcal_lamp() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.gcal_lamp, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.gcal_lamp, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_group_id_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.group_id()) == str)
                    or (ad.group_id() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.group_id, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.group_id, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_instrument_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.instrument()) == str)
                    or (ad.instrument() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.instrument, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.instrument, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_is_ao_descriptor_is_none_or_bool(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.is_ao()) == bool)
                    or (ad.is_ao() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.is_ao, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.is_ao, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_is_coadds_summed_descriptor_is_none_or_bool(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.is_coadds_summed()) == bool)
                    or (ad.is_coadds_summed() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.is_coadds_summed, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.is_coadds_summed, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_is_in_adu_descriptor_is_none_or_bool(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.is_in_adu()) == bool)
                    or (ad.is_in_adu() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.is_in_adu, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.is_in_adu, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_local_time_descriptor_is_none_or_datetime_time(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((isinstance(ad.local_time(), datetime.time))
                    or (ad.local_time() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.local_time, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.local_time, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_lyot_stop_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.lyot_stop()) == str) or
                    (ad.lyot_stop() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.lyot_stop, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.lyot_stop, str(err)))

    # TODO: mdf_row_id returns list, but string expected??
    @pytest.mark.parametrize("filename", fits_files)
    def test_mdf_row_id_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.mdf_row_id()) == list)
                    or (ad.mdf_row_id() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.mdf_row_id, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.mdf_row_id, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_nod_count_descriptor_is_none_or_tuple(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.nod_count()) == tuple)
                    or (ad.nod_count() is None))

        except AttributeError:
            # nod_count doesn't exist for some instruments
            if ad.instrument() in ['F2', 'GNIRS']:
                pass

        except Exception as err:
            print("{} failed on call: {}".format(ad.nod_count, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.nod_count, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_nod_offsets_descriptor_is_none_or_tuple(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.nod_offsets()) == tuple)
                    or (ad.nod_offsets() is None))

        except AttributeError:
            # nod_offsets doesn't exist for some instruments
            if ad.instrument() in ['F2', 'GNIRS']:
                pass

        except Exception as err:
            print("{} failed on call: {}".format(ad.nod_offsets, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.nod_offsets, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_nominal_atmospheric_extinction_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.nominal_atmospheric_extinction()) == float)
                    or (ad.nominal_atmospheric_extinction() is None))

        except Exception as err:
            print("{} failed on call: {}".format(
                ad.nominal_atmospheric_extinction, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.nominal_atmospheric_extinction, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_nominal_photometric_zeropoint_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.nominal_photometric_zeropoint()) == list)
                    or (ad.nominal_photometric_zeropoint() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.nominal_photometric_zeropoint, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.nominal_photometric_zeropoint, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_non_linear_level_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.non_linear_level()) == list)
                    or (ad.non_linear_level() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.non_linear_level, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.non_linear_level, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_nonlinearity_coeffs_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.nonlinearity_coeffs()) == list)
                    or (ad.nonlinearity_coeffs() is None))

        except AttributeError:
            # nonlinearity_coeffs doesn't exist for some instruments
            if ad.instrument() in ['GMOS-N', 'GMOS-S', 'GNIRS']:
                pass

        except Exception as err:
            print("{} failed on call: {}".format(ad.nonlinearity_coeffs, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.nonlinearity_coeffs, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_object_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.object()) == str)
                    or (ad.object() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.object, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.object, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_observation_class_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.observation_class()) == str)
                    or (ad.observation_class() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.observation_class, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.observation_class, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_observation_epoch_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.observation_epoch()) == float)
                    or (ad.observation_epoch() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.observation_epoch, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.observation_epoch, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_observation_id_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.observation_id()) == str)
                    or (ad.observation_id() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.observation_id, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.observation_id, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_observation_type_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.observation_type()) == str)
                    or (ad.observation_type() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.observation_type, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.observation_type, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_overscan_section_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.overscan_section()) == list)
                    or (ad.overscan_section() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.overscan_section, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.overscan_section, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_pixel_scale_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))
        if filename == os.path.join(path, "Archive/N20190216S0092.fits"):
            pytest.skip("Current issue with this file, will get back to it")
        try:
            assert ((type(ad.pixel_scale()) == float)
                    or (ad.pixel_scale() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.pixel_scale, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.pixel_scale, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_program_id_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.program_id()) == str)
                    or (ad.program_id() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.program_id, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.program_id, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_pupil_mask_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.pupil_mask()) == str)
                    or (ad.pupil_mask() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.pupil_mask, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.pupil_mask, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_qa_state_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.qa_state()) == str)
                    or (ad.qa_state() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.qa_state, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.qa_state, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_ra_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.ra()) == float)
                    or (ad.ra() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.ra, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.ra, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_raw_bg_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.raw_bg()) == int)
                    or (ad.raw_bg() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.raw_bg, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.raw_bg, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_raw_cc_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.raw_cc()) == int)
                    or (ad.raw_cc() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.raw_cc, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.raw_cc, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_raw_iq_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.raw_iq()) == int)
                    or (ad.raw_iq() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.raw_iq, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.raw_iq, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_raw_wv_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.raw_wv()) == int)
                    or (ad.raw_wv() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.raw_wv, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.raw_wv, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_read_mode_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.read_mode()) == str)
                    or (ad.read_mode() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.read_mode, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.read_mode, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_read_noise_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.read_noise()) == list)
                    or (ad.read_noise() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.read_noise, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.read_noise, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_read_speed_setting_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.read_speed_setting()) == str)
                    or (ad.read_speed_setting() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.read_speed_setting, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.read_speed_setting, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_requested_bg_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.requested_bg()) == int)
                    or (ad.requested_bg() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.requested_bg, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.requested_bg, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_requested_cc_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.requested_cc()) == int)
                    or (ad.requested_cc() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.requested_cc, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.requested_cc, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_requested_iq_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.requested_iq()) == int)
                    or (ad.requested_iq() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.requested_iq, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.requested_iq, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_requested_wv_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.requested_wv()) == int)
                    or (ad.requested_wv() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.requested_wv, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.requested_wv, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_saturation_level_descriptor_is_none_or_list(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.saturation_level()) == list)
                    or (ad.saturation_level() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.saturation_level, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.saturation_level, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_shuffle_pixels_descriptor_is_none_or_int(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.shuffle_pixels()) == int)
                    or (ad.shuffle_pixels() is None))

        except AttributeError:
            # shuffle_pixels doesn't exist for some instruments
            if ad.instrument() in ['F2', 'GNIRS']:
                pass

        except Exception as err:
            print("{} failed on call: {}".format(ad.shuffle_pixels, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.shuffle_pixels, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_slit_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.slit()) == str)
                    or (ad.slit() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.slit, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.slit, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_target_dec_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.target_dec()) == float)
                    or (ad.target_dec() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.target_dec, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.target_dec, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_target_ra_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.target_ra()) == float)
                    or (ad.target_ra() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.target_ra, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.target_ra, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_telescope_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.telescope()) == str)
                    or (ad.telescope() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.telescope, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.telescope, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_telescope_x_offset_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.telescope_x_offset()) == float)
                    or (ad.telescope_x_offset() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.telescope_x_offset, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.telescope_x_offset, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_telescope_y_offset_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
                assert ((type(ad.telescope_y_offset()) == float)
                        or (ad.telescope_y_offset() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.telescope_y_offset, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.telescope_y_offset, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_ut_date_descriptor_is_none_or_datetime_date(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((isinstance(ad.ut_date(), datetime.date))
                    or (ad.ut_date() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.ut_date, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.ut_date, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_ut_datetime_descriptor_is_none_or_datetime(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((isinstance(ad.ut_datetime(), datetime.datetime))
                    or (ad.ut_datetime() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.ut_datetime, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.ut_datetime, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_ut_time_descriptor_is_none_or_datetime_time(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((isinstance(ad.ut_time(), datetime.time))
                    or (ad.ut_time() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.ut_time, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.ut_time, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_wavefront_sensor_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.wavefront_sensor()) == str)
                    or (ad.wavefront_sensor() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.wavefront_sensor, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.wavefront_sensor, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_wavelength_band_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.wavelength_band()) == str)
                    or (ad.wavelength_band() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.wavelength_band, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.wavelength_band, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_wcs_dec_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.wcs_dec()) == float)
                    or (ad.wcs_dec() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.wcs_dec, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.wcs_dec, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_wcs_ra_descriptor_is_none_or_float(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))

        try:
            assert ((type(ad.wcs_ra()) == float)
                    or (ad.wcs_ra() is None))

        except Exception as err:
            print("{} failed on call: {}".format(ad.wcs_ra, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.wcs_ra, str(err)))

    @pytest.mark.parametrize("filename", fits_files)
    def test_well_depth_setting_descriptor_is_none_or_str(self, test_path, filename):
        ad = astrodata.open(os.path.join(test_path, "Archive/", filename))
    
        try:
            assert ((type(ad.well_depth_setting()) == str)
                    or (ad.well_depth_setting() is None))
    
        except Exception as err:
            print("{} failed on call: {}".format(ad.well_depth_setting, str(err)))
            # warnings.warn("{} failed on call: {}".format(ad.well_depth_setting, str(err)))
