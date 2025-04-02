# pytest suite
"""
Tests for the gmosaic eti.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): py.test -v --capture=no
"""

import os
import os.path
import pytest
import astrodata
import gemini_instruments


TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')

TESTGMOSFILE = os.path.join('GMOS', 'N20110524S0358_varAdded.fits')

TESTDEFAULTPARAMS = {
    "suffix" : "_mosaicked",
    "tile" : False,
    "interpolate_gaps" : False,
    "interpolator" : "linear"
}

class TestGmosaic:
    """
    Suite of tests for the gmosaic eti
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        TestGmosaic.gmos_file = os.path.join(TESTDATAPATH, TESTGMOSFILE)

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

    @pytest.mark.skip("Test requires IRAF/PyRAF")
    def test_gmos_default(self):
        """
        Test the mosaic of a GMOS dataset with var dq but that has
        not be overscan subtracted or trimmed.  Use default parameters.
        """
        from gempy.gemini.eti import gmosaiceti

        # where is the fits diff tool?
        ad = astrodata.from_file(TestGmosaic.gmos_file)
        inputs = []
        parameters = TESTDEFAULTPARAMS
        gmosaic_task = \
            gmosaiceti.GmosaicETI(inputs, parameters, ad)
        ad_mosaic = gmosaic_task.run()
        ad_mosaic.write(overwrite=True)
        del ad
        del ad_mosaic
        ##  NEED TO ADD A FITS DIFF.  Then remove overwrite and delete
        ##  the output fits once the diff is completed.
