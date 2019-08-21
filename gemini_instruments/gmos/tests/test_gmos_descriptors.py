#!/usr/bin/env python

import pytest

import astrodata
import gemini_instruments

GMOS_DESCRIPTORS_TYPES = [
    ('detector_x_offset', float),
    ('detector_y_offset', float),
    ('nod_count', tuple),
    ('nod_offsets', tuple),
    ('pixel_scale', float),
    ('shuffle_pixels', int),
]


@pytest.mark.parametrize("descriptor,expected_type", GMOS_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(descriptor, expected_type, gmos_files):
    for _file in gmos_files:
        ad = astrodata.open(_file)

        value = getattr(ad, descriptor)()

        assert isinstance(value, expected_type) or value is None, \
            "Assertion failed for file: {}".format(_file)


if __name__ == '__main__':
    pytest.main()
