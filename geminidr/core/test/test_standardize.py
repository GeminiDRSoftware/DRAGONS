# pytest suite
"""
Tests for primitives_standardize.

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

from . import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_standardize.log'

class TestStandardize:
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

    def test_addDQ(self):
        ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_prepared.fits'))
        p = NIRIImage([ad])
        ad = p.addDQ()[0]
        assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_dqAdded.fits'))

    def test_addIllumMaskToDQ(self):
        pass

    def test_addMDF(self):
        pass

    def test_validateData(self):
        # This is taken care of by prepare
        pass

    def test_addVAR(self):
        ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_ADUToElectrons.fits'))
        p = NIRIImage([ad])
        ad = p.addVAR(read_noise=True, poisson_noise=True)[0]
        assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_varAdded.fits'))

    def test_prepare(self):
        ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104.fits'))
        p = NIRIImage([ad])
        ad = p.prepare()[0]
        assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
                                'N20070819S0104_prepared.fits'))