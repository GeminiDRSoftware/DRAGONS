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


# Can only run first test (on a bundle) since other tests require non-archived inputs
@pytest.mark.ghost
@pytest.mark.parametrize("data, tags", tags_fixture_data[:1])
def test_descriptor(data, tags, path_to_inputs):
    """
    Ensure that the values returned by AstroData descriptors are as expected.
    """
    instrument, filename = data
    ad = astrodata.open(os.path.join(path_to_inputs, instrument, filename))

    assert set(tags) == set(ad.tags)
