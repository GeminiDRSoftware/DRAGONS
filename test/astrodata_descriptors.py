import pytest
import datetime
import astrodata
import instruments

fixture_data = {
    ('gnirs', 'N20160523S0191.fits'): (
        ('airmass', 1.035),
        ('ao_seeing', None),
        ('array_name', None),
        ('array_section', [0, 1024, 0, 1024]),
        ('azimuth', -54.6193152777778),
        ('camera', 'ShortBlue_G5540'),
        ('cass_rotator_pa', 82.4391422591155),
        ('central_wavelength', 1.6498999999999999e-06),
        ('coadds', 1),
        ('data_label', 'GN-2016A-Q-74-14-007'),
        ('data_section', [0, 1024, 0, 1024]),
        ('dec', 27.928468501322637),
        ('decker', 'SCXD_G5531'),
        ('detector_name', None),
        ('detector_roi_setting', 'Fixed'),
        ('detector_rois_requested', None),
        ('detector_section', [0, 1024, 0, 1024]),
        ('detector_x_bin', 1),
        ('detector_y_bin', 1),
        ('disperser', '32/mm_G5533&SXD_G5536'),
        ('dispersion', None),
        ('dispersion_axis', None),
        ('effective_wavelength', 1.6498999999999999e-06),
        ('elevation', 75.1844152777778),
        ('exposure_time', 115.0),
        ('filter_name', 'Open&XD_G0526'),
        ('focal_plane_mask', '1.00arcsec_G5530&SCXD_G5531'),
        ('gain', 13.5),
        ('gain_setting', None),
        ('gcal_lamp', 'None'),
        ('grating', '32/mm_G5533'),
        ('group_id', 'GN-2016A-Q-74-14_XD_ShortBlue_G5540_Very Faint Objects_Shallow_[0, 1024, 0, 1024]_32/mm&SXD_1.00arcsecXD'),
        ('instrument', 'GNIRS'),
        ('is_ao', False),
        ('is_coadds_summed', True),
        ('local_time', datetime.time(22, 12, 51, 600000)),
        ('lyot_stop', None),
        ('mdf_row_id', None),
        ('nod_count', None),
        ('nod_pixels', None),
        ('nominal_atmospheric_extinction', 0.0),
        ('nominal_photometric_zeropoint', None),
        ('non_linear_level', 4761),
        ('object', 'NGP_1301+2755'),
        ('observation_class', 'science'),
        ('observation_epoch', '2016.39'),
        ('observation_id', 'GN-2016A-Q-74-14'),
        ('observation_type', 'OBJECT'),
        ('overscan_section', None),
        ('pixel_scale', 0.15),
        ('prism', 'SXD_G5536'),
        ('program_id', 'GN-2016A-Q-74'),
        ('pupil_mask', None),
        ('qa_state', 'Pass'),
        ('ra', 195.29060316484365),
        ('raw_bg', 100),
        ('raw_cc', 50),
        ('raw_iq', 70),
        ('raw_wv', 50),
        ('read_mode', 'Very Faint Objects'),
        ('read_noise', 7.0),
        ('read_speed_setting', None),
        ('requested_bg', 100),
        ('requested_cc', 70),
        ('requested_iq', 85),
        ('requested_wv', 100),
        ('saturation_level', 6666),
        ('slit', '1.00arcsec_G5530'),
        ('target_dec', 27.93144167),
        ('target_ra', 195.291925),
        ('telescope', 'Gemini-North'),
        ('ut_date', datetime.date(2016, 5, 23)),
        ('ut_datetime', datetime.datetime(2016, 5, 23, 8, 12, 52, 100000)),
        ('ut_time', datetime.time(8, 12, 52, 100000)),
        ('wavefront_sensor', 'PWFS2'),
        ('wavelength_band', 'H'),
        ('wavelength_reference_pixel', None),
        ('wcs_dec', 27.928468501322637),
        ('wcs_ra', 195.29060316484365),
        ('well_depth_setting', 'Shallow'),
        ('x_offset', 1.48489831646263),
        ('y_offset', 1.39924758216777),
        )
}

class FixtureIterator(object):
    def __init__(self, data_dict):
        self._data = data_dict

    def __iter__(self):
        for (instr, filename) in sorted(self._data.keys()):
            ad = astrodata.open('data/{}/{}'.format(instr, filename))
            for desc, value in self._data[(instr, filename)]:
                yield ad, getattr(ad, desc), value

@pytest.mark.parametrize("ad,descriptor,value", FixtureIterator(fixture_data))
def test_descriptor(ad,descriptor,value):
    assert descriptor() == value
