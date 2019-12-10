#!/usr/bin/env python
"""
Configuration file for tests in `geminidr.gmos.tests`
"""
import pytest
import os

import astrodata
import gemini_instruments

from astrodata.testing import path_to_refs


@pytest.fixture(scope='module')
def ad_ref(request, path_to_refs):
    """
    Loads existing reference FITS files as AstroData objects.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    path_to_refs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached reference files.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.

    Raises
    ------
    IOError
        If the reference file does not exist. It should be created and verified
        manually.
    """
    fname = os.path.join(path_to_refs, request.param)

    if not os.path.exists(fname):
        raise IOError(" Cannot find reference file:\n {:s}".format(fname))

    return astrodata.open(fname)
