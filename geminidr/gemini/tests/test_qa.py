# pytest suite
"""
Tests for primitives_qa.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""
import os

import astrodata
import pytest

from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.utils import logutils

# TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_qa.log'

# --- Fixtures ---
@pytest.fixture
def ad(path_to_inputs):
    # File used before: GMOS/N20150624S0106_refcatAdded.fits
    path = os.path.join(path_to_inputs, "GMOS/N20150624S0106.fits")
    return astrodata.open(path)


# --- Tests ----
class TestQA:
    """
    Suite of tests for the functions in the gemini_tools module.
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        if os.path.exists(logfilename):
            os.remove(logfilename)
        log = logutils.get_logger(__name__)
        log.root.handlers = []
        logutils.config(mode='standard', file_name=logfilename)

    @classmethod
    def teardown_class(cls):
        """Run once at the end."""
        os.remove(logfilename)

    # noinspection PyPep8Naming
    @pytest.mark.xfail(reason="Correct values hardcoded")
    def test_measureBG(self, ad):

        p = GMOSImage([ad])
        ad = p.measureBG()[0]

        correct = [726.18213, 724.36047, 727.34491,
                   728.49664, 728.08966, 719.83728]

        for rv, cv in zip(ad.hdr['SKYLEVEL'], correct):
            assert abs(rv - cv) < 0.1, 'Wrong background level'

        assert (ad.phu['SKYLEVEL'] - 727.174) < 0.1

        f = open(logfilename, 'r')

        for line in f.readlines():
            if 'BG band' in line:
                assert line.split()[6] == 'BG80', 'Wrong BG band'

        ad.phu['REQBG'] = '50-percentile'
        ad = p.measureBG()[0]

        assert any('WARNING: BG requirement not met' in line
                   for line in f.readlines()), 'No BG warning'

    # noinspection PyPep8Naming
    @pytest.mark.xfail(reason="Correct values hardcoded")
    def test_measureCC(self, ad):

        p = GMOSImage([ad])
        ad = p.measureCC()[0]
        correct = [28.18, 28.16, 28.14, 28.11, 28.17, 28.12]
        for rv, cv in zip(ad.hdr['MEANZP'], correct):
            assert abs(rv - cv) < 0.02, 'Wrong zeropoint'

        f = open(logfilename, 'r')
        for line in f.readlines():
            if 'CC bands' in line:
                assert 'CC50, CC70' in line, 'Wrong CC bands'
        for ext in ad:
            ext.OBJCAT['MAG_AUTO'] += 0.3
        ad = p.measureCC()[0]
        correct = [c - 0.3 for c in correct]
        for rv, cv in zip(ad.hdr['MEANZP'], correct):
            assert abs(rv - cv) < 0.02, 'Wrong zeropoint after edit'
        ccwarn = False
        for line in f.readlines():
            if 'CC bands' in line:
                assert 'CC70, CC80' in line, 'Wrong CC bands after edit'
            ccwarn |= 'WARNING: CC requirement not met' in line
        assert ccwarn, 'No CC warning'

    # noinspection PyPep8Naming
    @pytest.mark.xfail(reason="Correct values hardcoded")
    def test_measureIQ(self, ad):

        p = GMOSImage([ad])
        ad = p.measureIQ()[0]
        # Try to give a reasonable-sized goal
        assert abs(ad.phu['MEANFWHM'] - 0.42) < 0.02, 'Wrong FWHM'
        assert abs(ad.phu['MEANELLP'] - 0.09) < 0.02, 'Wrong ellipticity'

        f = open(logfilename, 'r')
        for line in f.readlines():
            if 'IQ range' in line:
                assert line.split()[8] == 'IQ20', 'Wrong IQ band'

        ad.phu['REQIQ'] = '70-percentile'
        for ext in ad:
            ext.OBJCAT['PROFILE_FWHM'] *= 2.5
            ext.OBJCAT['PROFILE_EE50'] *= 2.5

        ad = p.measureIQ()[0]
        iqwarn = False

        for line in f.readlines():
            if 'IQ range' in line:
                assert line.split()[8] == 'IQ85', 'Wrong IQ band after edit'
            iqwarn |= 'WARNING: IQ requirement not met' in line

        assert iqwarn, 'NO IQ warning'
