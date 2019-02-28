# pytest suite
"""
Tests for primitives_stack.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""
import os
import astrodata
import gemini_instruments
from gempy.utils import logutils

from . import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_stack.log'

class TestStack:
    """
    Suite of tests for the functions in the primitives_standardize module.
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

    def test_stackFrames(self):
        pass

    def test_stackSkyFrames(self):
        pass