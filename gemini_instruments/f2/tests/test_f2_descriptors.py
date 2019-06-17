#!/usr/bin/env python

import pytest

import astrodata
import gemini_instruments


F2_DESCRIPTORS_TYPES = [
    ('detector_x_offset', float),
    ('detector_y_offset', float),
    ('nonlinearity_coeffs', list),
    ('pixel_scale', float),
]


@pytest.mark.parametrize("descriptor,expected_type", F2_DESCRIPTORS_TYPES)
def test_descriptor_matches_type(descriptor, expected_type, f2_files):

    for _file in f2_files:

        ad = astrodata.open(_file)

        value = getattr(ad, descriptor)()

        assert isinstance(value, expected_type) or value is None, \
            "Assertion failed for file: {}".format(_file)


if __name__ == '__main__':
    pytest.main()
