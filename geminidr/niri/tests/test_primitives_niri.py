# pytest suite
"""
Tests for primitives_niri

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""
import os
import pytest

import astrodata
from geminidr.core.tests import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage
from gempy.utils import logutils

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_niri.log'


class TestNIRI:
    """
    Suite of tests for the functions in the primitives_niri module.
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        if os.path.exists(logfilename):
            os.remove(logfilename)
        log = logutils.get_logger(__name__)
        log.root.handlers = []
        logutils.config(mode='standard', file_name=logfilename)

    # noinspection PyPep8Naming
    @pytest.mark.skip("File not available")
    def test_nonlinearityCorrect(self):

        ad = astrodata.open(
            os.path.join(TESTDATAPATH, 'NIRI', 'N20070819S0104_varAdded.fits'))

        p = NIRIImage([ad])
        ad = p.nonlinearityCorrect()[0]

        assert ad_compare(ad, os.path.join(
            TESTDATAPATH, 'NIRI', 'N20070819S0104_nonlinearityCorrected.fits'))
