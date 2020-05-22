# pytest suite
"""
Tests for primitives_bookkeeping.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""

# TODO @bquint: clean up these tests

import astrodata
import gemini_instruments
import os
import pytest

# from . import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage
from gempy.utils import logutils


TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_bookkeeping.log'


# --- Fixtures ---
@pytest.fixture(scope="class")
def log():

    if os.path.exists(logfilename):
        os.remove(logfilename)

    log = logutils.get_logger(__name__)
    log.root.handlers = []
    logutils.config(mode='standard', file_name=logfilename)

    yield log

    os.remove(logfilename)


# --- Tests ---
class TestBookkeeping:
    """
    Suite of tests for the functions in the primitives_standardize module.
    """

    @pytest.mark.xfail(reason="Test needs revision", run=False)
    def test_addToList(self):
        filenames = ['N20070819S{:04d}_flatCorrected.fits'.format(i)
                     for i in range(104, 109)]

        adinputs = [astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', f))
                    for f in filenames]

        # Add one image twice, just for laughs; it should appear only once
        adinputs.append(adinputs[0])

        p = NIRIImage(adinputs)
        p.stacks = {}
        p.addToList(purpose='forTest')

        for f in filenames:
            newfilename = f.replace('flatCorrected', 'forTest')
            assert os.path.exists(newfilename)
            os.remove(newfilename)

        # Check there's one stack of length 5
        assert len(p.stacks) == 1
        assert len(p.stacks[p.stacks.keys()[0]]) == 5

    @pytest.mark.xfail(reason="Test needs revision", run=False)
    def test_getList(self):
        pass

    @pytest.mark.xfail(reason="Test needs revision", run=False)
    def test_showInputs(self):
        pass

    @pytest.mark.xfail(reason="Test needs revision", run=False)
    def test_showList(self):
        pass

    @pytest.mark.xfail(reason="Test needs revision", run=False)
    def test_writeOutputs(self):
        filenames = ['N20070819S{:04d}_flatCorrected.fits'.format(i)
                     for i in range(104, 106)]

        adinputs = [astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', f))
                    for f in filenames]

        p = NIRIImage(adinputs)
        p.writeOutputs(prefix='test', suffix='_blah', strip=True)

        # Check renamed files are on disk and the filenames have been
        # changed for the adinputs
        for f, ad in zip(filenames, p.streams['main']):
            newfilename = 'test' + f.replace('flatCorrected', 'blah')
            assert os.path.exists(newfilename)

            os.remove(newfilename)
            assert newfilename == ad.filename
