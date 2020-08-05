#!/usr/bin/env python
"""
Tests for the p.addOIWFStoDQ primitive.
"""

import pytest

import astrodata
import gemini_instruments

from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_image import GMOSImage


@pytest.mark.parametrize("filename", ["N20190102S0162.fits"])
def test_oiwfs_not_used_in_observation(caplog, filename):
    """
    Test that nothing happens when the input file does not use the OIWFS.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addOIWFSToDQ()

    assert any("OIWFS not used for image" in r.message for r in caplog.records)


@pytest.mark.parametrize("filename", ["N20190101S0051.fits"])
def test_warn_if_dq_does_not_exist(caplog, filename):
    """
    Test that the primitive does not run if the input file does not have a DQ
    plan.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addOIWFSToDQ()

    assert any("No DQ plane for" in r.message for r in caplog.records)


@pytest.mark.parametrize("filename", ["N20190101S0051.fits"])
def test_add_oiwfs(caplog, filename):
    """
    Test that the primitive does not run if the input file does not have a DQ
    plan.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addDQ()
    p.addOIWFSToDQ()


def create_inputs():
    pass


if __name__ == '__main__':
    from sys import argv
    if '--create-inputs' in argv:
        create_inputs()
    else:
        pytest.main()
