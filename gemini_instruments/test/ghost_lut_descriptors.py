import datetime
from gemini_instruments.common import Section


# This has been modified from a dict to a hideous nested list so
# path_to_inputs can be used
fixture_data = {
    ('GHOST', "S20230214S0025.fits"): [  # BUNDLE
        ('airmass', 1.0907455772702386),
        ('amp_read_area', {'blue': ["'EEV231-84, E':[1:2048,1:2056]", "'EEV231-84, F':[2049:4096,1:2056]",
                                    "'EEV231-84, G':[2049:4096,2057:4112]", "'EEV231-84, H':[1:2048,2057:4112]"],
                           'red': ["'EEV231-C6, E':[1:3072,1:3080]", "'EEV231-C6, F':[3073:6144,1:3080]",
                                   "'EEV231-C6, G':[3073:6144,3081:6160]", "'EEV231-C6, H':[1:3072,3081:6160]"],
                           'slitv': "'Ghost BigEye Sony ICX674, A':[801:1100,681:940]"}),
        ('arm', None),
        ('array_name', {'blue': ['EEV231-84, E', 'EEV231-84, F', 'EEV231-84, G', 'EEV231-84, H'],
                        'red': ['EEV231-C6, E', 'EEV231-C6, F', 'EEV231-C6, G', 'EEV231-C6, H'],
                        'slitv': 'Ghost BigEye Sony ICX674, A'}),
        ('azimuth', 34.61275138888889),
        ('binning', {'blue': '1x4', 'red': '1x4', 'slitv': '2x2'}),
        ('calibration_key', ('GS-ENG-GHOST-COM-3-560-001', None)),
        ('camera', None),
        ('cass_rotator_pa', -29.592426692554707),
        ('central_wavelength', None),
        ('coadds', 1),
        ('data_label', 'GS-ENG-GHOST-COM-3-560-001'),
        ('data_section', [Section(x1=0, x2=150, y1=0, y2=130), Section(x1=0, x2=150, y1=0, y2=130),
                          Section(x1=0, x2=150, y1=0, y2=130), Section(x1=0, x2=150, y1=0, y2=130),
                          Section(x1=0, x2=150, y1=0, y2=130), None, Section(x1=0, x2=3072, y1=0, y2=770),
                          Section(x1=32, x2=3104, y1=0, y2=770), Section(x1=32, x2=3104, y1=0, y2=770),
                          Section(x1=0, x2=3072, y1=0, y2=770), None, Section(x1=0, x2=3072, y1=0, y2=770),
                          Section(x1=32, x2=3104, y1=0, y2=770), Section(x1=32, x2=3104, y1=0, y2=770),
                          Section(x1=0, x2=3072, y1=0, y2=770), None, Section(x1=0, x2=2048, y1=0, y2=514),
                          Section(x1=32, x2=2080, y1=0, y2=514), Section(x1=32, x2=2080, y1=0, y2=514),
                          Section(x1=0, x2=2048, y1=0, y2=514), None, Section(x1=0, x2=3072, y1=0, y2=770),
                          Section(x1=32, x2=3104, y1=0, y2=770), Section(x1=32, x2=3104, y1=0, y2=770),
                          Section(x1=0, x2=3072, y1=0, y2=770)]),
        ('dec', -7.88625),
        ('detector_name', {'blue': 'EEV231-84', 'red': 'EEV231-C6', 'slitv': 'Ghost BigEye Sony ICX674'}),
        ('detector_x_bin', {'blue': 1, 'red': 1, 'slitv': 2}),
        ('detector_x_offset', None),
        ('detector_y_bin', {'blue': 4, 'red': 4, 'slitv': 2}),
        ('detector_y_offset', None),
        ('disperser', None),
        ('dispersion', None),
        ('dispersion_axis', None),
        ('effective_wavelength', None),
        ('elevation', 63.87784305555556),
        ('exposure_time', {'blue': 3600.0, 'red': 1200.0, 'slitv': 240.0}),
        ('filter_name', None),
        ('focal_plane_mask', 'SR'),
        ('gain', [1.0, 1.0, 1.0, 1.0, 1.0,
                  None, 0.51, 0.5, 0.52, 0.55,
                  None, 0.51, 0.5, 0.52, 0.55,
                  None, 0.59, 0.52, 0.54, 0.58,
                  None, 0.51, 0.5, 0.52, 0.55]),
        ('gain_setting', {'blue': 'low', 'red': 'low', 'slitv': 'standard'}),
        ('instrument', 'GHOST'),
        ('number_of_exposures', {'blue': 1, 'red': 3, 'slitv': 5}),
        ('object', 'Gaia-EDR3-3063607731380726272'),
        ('observation_class', 'science'),
        ('observation_id', 'GS-ENG-GHOST-COM-3-560'),
        ('observation_type', 'OBJECT'),
        ('program_id', 'GS-ENG-GHOST-COM-3'),
        ('ra', 122.30433333333333),
        ('raw_bg', 20),
        ('raw_cc', 50),
        ('raw_iq', 20),
        ('raw_wv', 20),
        ('read_mode', {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}),
        ('read_noise', [1.0, 1.0, 1.0, 1.0, 1.0,
                        None, 2.3, 2.3, 2.4, 2.6,
                        None, 2.3, 2.3, 2.4, 2.6,
                        None, 2.1, 2.0, 2.1, 2.1,
                        None, 2.3, 2.3, 2.4, 2.6]),
        ('read_speed_setting', {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}),
        ('target_dec', -7.88625),
        ('target_ra', 122.3043333333333),
        ],
    ('GHOST', 'S20230214S0025_blue001.fits'): [
        ('airmass', 1.090745577270239),
        ('amp_read_area',
         ["'EEV231-84, E':[1:2048,1:2056]", "'EEV231-84, F':[2049:4096,1:2056]", "'EEV231-84, G':[2049:4096,2057:4112]",
          "'EEV231-84, H':[1:2048,2057:4112]"]),
        ('arm', 'blue'),
        ('array_name', ['EEV231-84, E', 'EEV231-84, F', 'EEV231-84, G', 'EEV231-84, H']),
        ('array_section', [Section(x1=0, x2=2048, y1=0, y2=2056), Section(x1=2048, x2=4096, y1=0, y2=2056),
                           Section(x1=2048, x2=4096, y1=2056, y2=4112), Section(x1=0, x2=2048, y1=2056, y2=4112)]),
        ('azimuth', 34.61275138888889),
        ('binning', '1x4'),
        ('calibration_key', ('GS-ENG-GHOST-COM-3-560-001-BLUE-001', 'blue')),
        ('camera', 'BLUE'),
        ('data_label', 'GS-ENG-GHOST-COM-3-560-001-BLUE-001'),
        ('data_section', [Section(x1=0, x2=2048, y1=0, y2=514), Section(x1=32, x2=2080, y1=0, y2=514),
                          Section(x1=32, x2=2080, y1=0, y2=514), Section(x1=0, x2=2048, y1=0, y2=514)]),
        ('dec', -7.88625),
        ('detector_name', 'EEV231-84'),
        ('detector_section', [Section(x1=0, x2=2048, y1=0, y2=2056), Section(x1=2048, x2=4096, y1=0, y2=2056),
                              Section(x1=2048, x2=4096, y1=2056, y2=4112), Section(x1=0, x2=2048, y1=2056, y2=4112)]),
        ('detector_x_bin', 1),
        ('detector_y_bin', 4),
        ('elevation', 63.87784305555556),
        ('exposure_time', 3600.0),
        ('focal_plane_mask', 'SR'),
        ('gain', [0.59, 0.52, 0.54, 0.58]),
        ('gain_setting', 'low'),
        ('instrument', 'GHOST'),
        ('non_linear_level', [65535, 65535, 65535, 65535]),
        ('number_of_exposures', 1),
        ('object', 'Gaia-EDR3-3063607731380726272'),
        ('observation_class', 'science'),
        ('observation_id', 'GS-ENG-GHOST-COM-3-560'),
        ('observation_type', 'OBJECT'),
        ('overscan_section', [Section(x1=2048, x2=2080, y1=0, y2=514), Section(x1=0, x2=32, y1=0, y2=514),
                              Section(x1=0, x2=32, y1=0, y2=514), Section(x1=2048, x2=2080, y1=0, y2=514)]),
        ('program_id', 'GS-ENG-GHOST-COM-3'),
        ('ra', 122.3043333333333),
        ('raw_bg', 20),
        ('raw_cc', 50),
        ('raw_iq', 20),
        ('raw_wv', 20),
        ('read_mode', 'slow'),
        ('read_noise', [2.1, 2.0, 2.1, 2.1]),
        ('read_speed_setting', 'slow'),
        ('saturation_level', [65535, 65535, 65535, 65535]),
        ('target_dec', -7.88625),
        ('target_ra', 122.3043333333333),
        ('ut_datetime', datetime.datetime(2023, 2, 14, 2, 19, 12))
        ],
    ('GHOST', 'S20230214S0025_red002.fits'): [
        ('airmass', 1.090745577270239),
        ('amp_read_area',
         ["'EEV231-C6, E':[1:3072,1:3080]", "'EEV231-C6, F':[3073:6144,1:3080]", "'EEV231-C6, G':[3073:6144,3081:6160]",
          "'EEV231-C6, H':[1:3072,3081:6160]"]),
        ('arm', 'red'),
        ('array_name', ['EEV231-C6, E', 'EEV231-C6, F', 'EEV231-C6, G', 'EEV231-C6, H']),
        ('array_section', [Section(x1=0, x2=3072, y1=0, y2=3080), Section(x1=3072, x2=6144, y1=0, y2=3080),
                           Section(x1=3072, x2=6144, y1=3080, y2=6160), Section(x1=0, x2=3072, y1=3080, y2=6160)]),
        ('azimuth', 34.61275138888889),
        ('binning', '1x4'),
        ('calibration_key', ('GS-ENG-GHOST-COM-3-560-001-RED-002', 'red')),
        ('camera', 'RED'),
        ('data_label', 'GS-ENG-GHOST-COM-3-560-001-RED-002'),
        ('data_section', [Section(x1=0, x2=3072, y1=0, y2=770), Section(x1=32, x2=3104, y1=0, y2=770),
                          Section(x1=32, x2=3104, y1=0, y2=770), Section(x1=0, x2=3072, y1=0, y2=770)]),
        ('dec', -7.88625),
        ('detector_name', 'EEV231-C6'),
        ('detector_section', [Section(x1=0, x2=3072, y1=0, y2=3080), Section(x1=3072, x2=6144, y1=0, y2=3080),
                              Section(x1=3072, x2=6144, y1=3080, y2=6160), Section(x1=0, x2=3072, y1=3080, y2=6160)]),
        ('detector_x_bin', 1),
        ('detector_y_bin', 4),
        ('elevation', 63.87784305555556),
        ('exposure_time', 1200.0),
        ('focal_plane_mask', 'SR'),
        ('gain', [0.51, 0.5, 0.52, 0.55]),
        ('gain_setting', 'low'),
        ('instrument', 'GHOST'),
        ('non_linear_level', [65535, 65535, 65535, 65535]),
        ('number_of_exposures', 1),
        ('object', 'Gaia-EDR3-3063607731380726272'),
        ('observation_class', 'science'),
        ('observation_id', 'GS-ENG-GHOST-COM-3-560'),
        ('observation_type', 'OBJECT'),
        ('overscan_section', [Section(x1=3072, x2=3104, y1=0, y2=770), Section(x1=0, x2=32, y1=0, y2=770),
                              Section(x1=0, x2=32, y1=0, y2=770), Section(x1=3072, x2=3104, y1=0, y2=770)]),
        ('program_id', 'GS-ENG-GHOST-COM-3'),
        ('ra', 122.3043333333333),
        ('raw_bg', 20),
        ('raw_cc', 50),
        ('raw_iq', 20),
        ('raw_wv', 20),
        ('read_mode', 'medium'),
        ('read_noise', [2.3, 2.3, 2.4, 2.6]),
        ('read_speed_setting', 'medium'),
        ('saturation_level', [65535, 65535, 65535, 65535]),
        ('target_dec', -7.88625),
        ('target_ra', 122.3043333333333),
        ('ut_datetime', datetime.datetime(2023, 2, 14, 2, 39, 27))
        ],
    ('GHOST', 'S20230214S0025_slit.fits'): [
        ('airmass', 1.0907455772702386),
        ('amp_read_area', ["'Ghost BigEye Sony ICX674, A':[801:1100,681:940]"]),
        ('ao_seeing', None),
        ('arm', 'slitv'),
        ('array_name', ['Ghost BigEye Sony ICX674, A'] * 5),
        ('array_section', [Section(x1=800, x2=1100, y1=680, y2=940)] * 5),
        ('azimuth', 34.61275138888889),
        ('binning', '2x2'),
        ('calibration_key', ('GS-ENG-GHOST-COM-3-560-001-SLITV', 'slitv')),
        ('camera', 'SLITV'),
        ('data_label', 'GS-ENG-GHOST-COM-3-560-001-SLITV'),
        ('data_section', [Section(x1=0, x2=150, y1=0, y2=130)] * 5),
        ('dec', -7.88625),
        ('detector_name', 'Ghost BigEye Sony ICX674'),
        ('detector_section', [Section(x1=800, x2=1100, y1=680, y2=940)] * 5),
        ('detector_x_bin', 2),
        ('detector_y_bin', 2),
        ('exposure_time', 240.0),
        ('focal_plane_mask', 'SR'),
        ('gain', [1.0, 1.0, 1.0, 1.0, 1.0]),
        ('gain_setting', 'standard'),
        ('instrument', 'GHOST'),
        ('number_of_exposures', 5),
        ('object', 'Gaia-EDR3-3063607731380726272'),
        ('observation_class', 'science'),
        ('observation_epoch', None),
        ('observation_id', 'GS-ENG-GHOST-COM-3-560'),
        ('observation_type', 'OBJECT'),
        ('program_id', 'GS-ENG-GHOST-COM-3'),
        ('ra', 122.3043333333333),
        ('raw_bg', 20),
        ('raw_cc', 50),
        ('raw_iq', 20),
        ('raw_wv', 20),
        ('read_mode', 'standard'),
        ('read_noise', [1.0, 1.0, 1.0, 1.0, 1.0]),
        ('read_speed_setting', 'standard'),
        ('res_mode', 'std'),
        ('saturation_level', [16383, 16383, 16383, 16383, 16383]),
        ('target_dec', -7.88625),
        ('target_ra', 122.3043333333333),
        ('ut_datetime', datetime.datetime(2023, 2, 14, 2, 19, 12))
        ]
}