# pytest suite

"""
Tests for the fitsverify module.

This is a suite of tests to be run with pytest.

To run:
   1) Set the environment variable GEMPYTHON_TESTDATA to the path that contains
      the gempython test data files.
      This suite uses the file N20130510S0178_forStack.fits.
      Eg. /net/chara/data2/pub/gempython_testdata
   2) From the ??? (location): py.test -v
"""
import pytest
import os
import os.path
from gempy.library import fitsverify

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
TESTFITS = os.path.join('GMOS','N20130510S0178_forStack.fits')

class TestFitsverify:
    """
    Suite of tests for the functions in the fitsverify module.
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        TestFitsverify.fits_file = os.path.join(TESTDATAPATH, TESTFITS)

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

    @pytest.mark.skip(reason='Uses local data')
    def test_fitsverify(self):
        """
        Test the return values of fitsverify on our test file.
        """
        returned_values = fitsverify.fitsverify(TestFitsverify.fits_file)
        assert returned_values[:3] == [1, '21', '0']