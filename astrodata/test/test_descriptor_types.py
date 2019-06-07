
import datetime
import glob
import os

import pytest

import astrodata
from .conftest import test_path


DESCRIPTORS_TYPES = [
    ('airmass', float),
    ('amp_read_area', list),
    ('ao_seeing', float),
    ('array_name', list),
    ('array_section', list),
    ('azimuth', float),
    ('calibration_key', str),
    ('camera', str),
    ('cass_rotator_pa', float),
    ('central_wavelength', float),
    ('coadds', int),
    ('data_label', str),
    ('data_section', list),
    ('dec', float),
    ('decker', str),
    ('detector_name', str),
    ('detector_roi_setting', str),
    ('detector_rois_requested', list),
    ('detector_section', list),
    ('detector_x_bin', int),
    ('detector_x_offset', float),
    ('detector_y_bin', int),
    ('detector_y_offset', float),
    ('disperser', str),
    ('dispersion', list),
    ('dispersion_axis', list),
    ('effective_wavelength', float),
    ('elevation', float),
    ('exposure_time', float),
    ('filter_name', str),
    ('focal_plane_mask', str),
    ('gain', list),
    ('gain_setting', str),
    ('gcal_lamp', str),
    ('group_id', str),
    ('instrument', str),
    ('is_ao', bool),
    ('is_coadds_summed', bool),
    ('is_in_adu', bool),
    ('local_time', datetime.time),
    ('lyot_stop', str),
    ('mdf_row_id', list),  # TODO: mdf_row_id returns list, but string expected??
    ('nod_count', tuple),
    ('nod_offsets', tuple),
    ('nominal_atmospheric_extinction', float),
    ('nominal_photometric_zeropoint', list),
    ('non_linear_level', list),
    ('nonlinearity_coeffs', list),
    ('object', str),
    ('observation_class', str),
    ('observation_epoch', float),
    ('observation_id', str),
    ('observation_type', str),
    ('overscan_section', list),
    ('pixel_scale', float),
    ('program_id', str),
    ('pupil_mask', str),
    ('qa_state', str),
    ('ra', float),
    ('raw_bg', int),
    ('raw_cc', int),
    ('raw_iq', int),
    ('raw_wv', int),
    ('read_mode', str),
    ('read_noise', list),
    ('read_speed_setting', str),
    ('requested_bg', int),
    ('requested_cc', int),
    ('requested_iq', int),
    ('requested_wv', int),
    ('saturation_level', list),
    ('shuffle_pixels', int),
    ('slit', str),
    ('target_dec', float),
    ('target_ra', float),
    ('telescope', str),
    ('telescope_x_offset', float),
    ('telescope_y_offset', float),
    ('ut_date', datetime.date),
    ('ut_datetime', datetime.datetime),
    ('ut_time', datetime.time),
    ('wavefront_sensor', str),
    ('wavelength_band', str),
    ('wcs_dec', float),
    ('wcs_ra', float),
    ('well_depth_setting', str),
]


@pytest.fixture(scope="module")
def archive_files(test_path):

    yield glob.glob(os.path.join(test_path, "Archive/", "*fits"))


@pytest.mark.parametrize("descriptor,expected_type", DESCRIPTORS_TYPES)
def test_descriptor_matches_type(descriptor, expected_type, archive_files):

    for _file in archive_files:

        ad = astrodata.open(_file)

        try:
            value = getattr(ad, descriptor)()
        except AttributeError as e:
            raise Exception('Error triggered in file: {}'.format(_file)).with_traceback(e.__traceback__)

        assert isinstance(value, expected_type) or value is None, \
            "Assertion failed for file: {}".format(_file)


# def test_nonlinearity_coeffs_descriptor_is_none_or_list(archive_files):
#
#     for _file in archive_files:
#         ad = astrodata.open(_file)
#
#         try:
#             assert ((type(ad.nonlinearity_coeffs()) == list)
#                     or (ad.nonlinearity_coeffs() is None))
#
#         except AttributeError:
#             # nonlinearity_coeffs doesn't exist for some instruments
#             if ad.instrument() in ['GMOS-N', 'GMOS-S', 'GNIRS']:
#                 pass
#
#         except Exception as err:
#             print("{} failed on call: {}".format(ad.nonlinearity_coeffs, str(err)))
#             # warnings.warn("{} failed on call: {}".format(ad.nonlinearity_coeffs, str(err)))
#
#
# def test_shuffle_pixels_descriptor_is_none_or_int(archive_files):
#
#     for _file in archive_files:
#         ad = astrodata.open(_file)
#
#         try:
#             assert ((type(ad.shuffle_pixels()) == int)
#                     or (ad.shuffle_pixels() is None))
#
#         except AttributeError:
#             # shuffle_pixels doesn't exist for some instruments
#             if ad.instrument() in ['F2', 'GNIRS']:
#                 pass
#
#         except Exception as err:
#             print("{} failed on call: {}".format(ad.shuffle_pixels, str(err)))
#             # warnings.warn("{} failed on call: {}".format(ad.shuffle_pixels, str(err)))