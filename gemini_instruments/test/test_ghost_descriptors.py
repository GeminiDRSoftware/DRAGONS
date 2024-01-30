"""
Perform a series of regression tests across GHOST-specific AstroData
descriptors.
"""

import pytest
from pytest_dragons.fixtures import *
import astrodata, gemini_instruments
import astrodata.testing
import os

THIS_DIR = os.path.dirname(__file__)

from .ghost_lut_descriptors import fixture_data as descriptors_fixture_data

# ---
# REGRESSION TESTING
# ---


# Can only run first test (on a bundle) since other tests require non-archived inputs
@pytest.mark.parametrize("data, descriptors", descriptors_fixture_data[:1])
def test_descriptor(data, descriptors):
    """
    Ensure that the values returned by AstroData descriptors are as expected.
    """
    instrument, filename = data
    ad = astrodata.open(astrodata.testing.download_from_archive(filename))
    conflicts = []
    for descriptor, value in descriptors:
        method = getattr(ad, descriptor)
    if isinstance(value, type) and issubclass(value, BaseException):
        try:
            mvalue = method()
        except value:
            pass
        else:
            conflicts.append(f"{descriptor} failed to raise a {value.__name__}")
    else:
        mvalue = method()
        if float in (type(value), type(mvalue)) and value is not None and mvalue is not None:
            if not mvalue == pytest.approx(value, 0.0001):
                conflicts.append(f"{descriptor} returned {mvalue} (expected {value})")
        elif not mvalue == value:
            conflicts.append(f"{descriptor} returned {mvalue} (expected {value})")

    assert not conflicts, "\n".join(conflicts)
