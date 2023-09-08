"""
Perform a series of regression tests across GHOST-specific AstroData tags.
"""

import pytest
from pytest_dragons.fixtures import *
import astrodata, gemini_instruments
import os

THIS_DIR = os.path.dirname(__file__)

from .ghost_lut_tags import fixture_data as tags_fixture_data
#tags_fixture_data = {}

# ---
# REGRESSION TESTING
# ---


@pytest.mark.ghostunit
@pytest.mark.parametrize("data, tags", tags_fixture_data)
def test_descriptor(data, tags, path_to_inputs):
    """
    Ensure that the values returned by AstroData descriptors are as expected.
    """
    instrument, filename = data
    ad = astrodata.open(os.path.join(path_to_inputs, instrument, filename))

    assert set(tags) == set(ad.tags)
