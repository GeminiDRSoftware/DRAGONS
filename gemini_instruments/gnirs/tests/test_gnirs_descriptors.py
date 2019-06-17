#!/usr/bin/env python

import pytest

import astrodata
import gemini_instruments

from gempy.adlibrary import dataselect


GNIRS_DESCRIPTORS_TYPES = [
    ('detector_x_offset', float),
    ('detector_y_offset', float),
    ('pixel_scale', float),
]


@pytest.mark.parametrize("descriptor,expected_type", GNIRS_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(descriptor, expected_type, gemini_files):

    gnirs_files = dataselect.select_data(gemini_files, tags=['GNIRS'])

    for _file in gnirs_files:

        ad = astrodata.open(_file)

        value = getattr(ad, descriptor)()

        assert isinstance(value, expected_type) or value is None, \
            "Assertion failed for file: {}".format(_file)


if __name__ == '__main__':
    pytest.main()
