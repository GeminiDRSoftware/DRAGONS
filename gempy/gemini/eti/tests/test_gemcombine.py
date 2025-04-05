# pytest suite
"""
Tests for the gemcombine eti.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories that contains the files ....
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): py.test -v --capture=no
"""

import os
import os.path
import pytest

import astrodata
import gemini_instruments


TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
# TESTGMOSLIST = [
#     os.path.join('GMOS', '.fits'),
#     os.path.join('GMOS', '.fits'),
#     os.path.join('GMOS', '.fits'),
#     os.path.join('GMOS', '.fits'),
#     os.path.join('GMOS', '.fits'),
# ]
TESTNIRILIST = [
    os.path.join('NIRI', 'N20130404S0372_aligned.fits'),
    os.path.join('NIRI', 'N20130404S0373_aligned.fits'),
    os.path.join('NIRI', 'N20130404S0374_aligned.fits'),
    os.path.join('NIRI', 'N20130404S0375_aligned.fits'),
    os.path.join('NIRI', 'N20130404S0376_aligned.fits'),

]

TESTDEFAULTPARAMS = {
    "suffix": "_stack",
    "mask": True,
    "nhigh": 1,
    "nlow": 1,
    "operation": "average",
    "reject_method": "avsigclip"
}


class TestGemcombine:
    """
    Suite of tests for the gemcombine eti.
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        # TestGemcombine.gmos_files = []
        # for filename in TESTGMOSLIST:
        #     TestGemcombine.gmos_files.append(filename)
        TestGemcombine.niri_files = []
        for filename in TESTNIRILIST:
            fullpath = os.path.join(TESTDATAPATH, filename)
            TestGemcombine.niri_files.append(fullpath)

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

    @pytest.mark.skip(reason="Test requires IRAF/PyRAF")
    def test_niri_default(self):
        """
        Test the stacking of NIRI data using QAP default parameters.
        """
        from gempy.gemini.eti import gemcombineeti

        # where is the fits diff tool?
        inputs = []
        for filename in TestGemcombine.niri_files:
            ad = astrodata.from_file(filename)
            inputs.append(ad)
        parameters = TESTDEFAULTPARAMS
        gemcombine_task = \
            gemcombineeti.GemcombineETI(inputs, parameters)
        ad_stack = gemcombine_task.run()
        ad_stack.write(overwrite=True)
        del inputs
        del ad_stack
        ##  NEED TO ADD A FITS DIFF.  Then remove overwrite and delete
        ##  the output fits once the diff is completed.
