# pytest suite
"""
Tests for primitives_photometry.

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
from geminidr.gmos.primitives_gmos_image import GMOSImage

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_photometry.log'

class TestPhotometry:
    """
    Suite of tests for the functions in the primitives_photometry module.
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

    def test_addReferenceCatalog(self):
        filename = os.path.join(TESTDATAPATH, 'GMOS',
                                'N20150624S0106_refcatAdded.fits')
        ad = astrodata.open(filename)
        # Delete REFCAT and timestamp keyword so we can run again
        del ad.REFCAT
        del ad.phu['ADDRECAT']
        p = GMOSImage([ad])
        ad = p.addReferenceCatalog([ad])[0]
        assert ad_compare(ad, filename)

    def test_detectSources(self):
        filename = os.path.join(TESTDATAPATH, 'GMOS',
                                'N20150624S0106_refcatAdded.fits')
        ad = astrodata.open(filename)
        del ad.phu['DETECSRC']
        for ext in ad:
            del ext.OBJCAT
            del ext.OBJMASK
        p = GMOSImage([ad])
        ad = p.detectSources([ad], suffix='_refcatAdded')[0]
        assert ad_compare(ad, filename)