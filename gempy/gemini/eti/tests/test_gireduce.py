# pytest suite
"""
Tests for the gireduce eti.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): py.test -v --capture=no
"""

import os
import os.path
import astrodata
import gemini_instruments
from gempy.gemini import eti

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')

TESTGMOSFILE = os.path.join('GMOS', 'N20110524S0358_varAdded.fits')

TESTDEFAULTPARAMS = {
    "suffix" : "_overscanSubtracted",
    "overscan_section" : None
}

class TestGireduce(object):
    """
    Suite of tests for the gireduce eti
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        TestGireduce.gmos_file = os.path.join(TESTDATAPATH, TESTGMOSFILE)

    @classmethod
    def teardown_class(cls):
        """Run once at the end."""
        pass

    def setup_method(self, method):
        """Run once before every test."""
        pass

    def teardown_method(self, method):
        """Run once after every test."""
        pass

    def test_gmosoverscan_default(self):
        """
        Test the overscan subtraction in a GMOS image using default
        parameters.
        """
        # where is the fits diff tool?
        ad = astrodata.open(TestGireduce.gmos_file)
        inputs = []
        parameters = TESTDEFAULTPARAMS
        gireduce_task = \
            eti.gireduceeti.GireduceETI(inputs, parameters, ad)
        ad_oversub = gireduce_task.run()
        ad_oversub.write(clobber=True)
        del ad
        del ad_oversub
        ##  NEED TO ADD A FITS DIFF.  Then remove clobber and delete
        ##  the output fits once the diff is completed.
