#!/usr/bin/env python

import datetime
import pytest
import astrodata
import astrodata.testing
import gemini_instruments

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
    ('detector_y_bin', int),
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
    ('nominal_atmospheric_extinction', float),
    ('nominal_photometric_zeropoint', list),
    ('non_linear_level', list),
    ('object', str),
    ('observation_class', str),
    ('observation_epoch', float),
    ('observation_id', str),
    ('observation_type', str),
    ('overscan_section', list),
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

test_files = [
    "gS20190219S0207_bias.fits",
    "N20190216S0092.fits",
    "S20180211S0294.fits",
    "N20190112S0005.fits",
    "N20190219S0072.fits",
    "S20190203S0320.fits",
    "N20190214S0058.fits",
    "N20190221S0001.fits",
    "S20190214S0576.fits",
    "N20190216S0039.fits",
    "N20190221S0032.fits",
    "S20190217S0211.fits",
    "N20190216S0060.fits",
    "S20160905S0003.fits",
    "S20190221S0251.fits",
]


@pytest.fixture(scope='module')
def ad(request):
    filename = request.param
    path = astrodata.testing.download_from_archive(filename)
    return astrodata.open(path)


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("ad", test_files, indirect=True)
def test_descriptor_matches_type(ad):
    for descriptor, expected_type in DESCRIPTORS_TYPES:
        value = getattr(ad, descriptor)()
        assert isinstance(value, expected_type) or value is None, \
            f"Assertion failed for file: {ad.filename} descriptor {descriptor}"


if __name__ == "__main__":
    pytest.main()
