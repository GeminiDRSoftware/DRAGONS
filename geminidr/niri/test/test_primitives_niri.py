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
import numpy as np
import astrodata
import gemini_instruments
from gempy.utils import logutils

from geminidr.core.test import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage

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
        logutils.config(mode='standard', console_lvl='stdinfo',
                        file_name=logfilename)

    @classmethod
    def teardown_class(cls):
        """Run once at the end."""
        os.remove(logfilename)

    def setup_method(self, method):
        """Run once before every test."""
        pass

    def teardown_method(self, method):
        """Run once after every test."""
        pass

    def test_nonlinearityCorrect(self):
        ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_varAdded.fits'))
        p = NIRIImage([ad])
        ad = p.nonlinearityCorrect()[0]
        assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_nonlinearityCorrected.fits'))