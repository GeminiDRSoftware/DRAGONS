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
from geminidr.gmos.primitives_gmos_image import GMOSImage
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


@pytest.fixture(scope="function")
def niri_ads(request, astrofaker):
    return [astrofaker.create('NIRI', ['IMAGE'], filename=f"X{i+1}.fits")
            for i in range(request.param)]


# --- Tests ---
@pytest.mark.parametrize('niri_ads', [3], indirect=True)
def test_append_stream(niri_ads):
    """Some manipulation of streams using appendStream()"""
    def filenames(stream):
        return ''.join([ad.filename[1] for ad in stream])

    p = NIRIImage(niri_ads[:1])
    p.streams['test'] = niri_ads[1:2]
    # Add the AD in 'test' to 'main' leaving it in 'test'
    p.appendStream(from_stream='test', copy=True)
    assert len(p.streams['main']) == 2
    assert len(p.streams['test']) == 1
    # Change filename of version in 'test' to confirm that the one in 'main'
    # is not simply a reference
    p.streams['test'][0].filename = 'X4.fits'
    assert filenames(p.streams['main']) == '12'

    # Add the copy in 'test' to 'main', and delete 'test'
    p.appendStream(from_stream='test', copy=False)
    assert len(p.streams['main']) == 3
    assert filenames(p.streams['main']) == '124'

    # Take 'test2', append 'main', and put the result in 'main'
    p.streams['test2'] = niri_ads[2:]
    p.appendStream(instream='test2', from_stream='main')
    assert filenames(p.streams['main']) == '3124'


@pytest.mark.parametrize('niri_ads', [2], indirect=True)
def test_clear_all_streams(niri_ads):
    p = NIRIImage(niri_ads[:1])
    p.streams['test'] = niri_ads[1:]
    p.clearAllStreams()
    assert not p.streams['test']
    assert len(p.streams['main']) == 1


@pytest.mark.parametrize('niri_ads', [2], indirect=True)
def test_clear_stream(niri_ads):
    p = NIRIImage(niri_ads[:1])
    p.streams['test'] = niri_ads[1:]
    p.clearStream(stream='test')
    assert not p.streams['test']
    assert len(p.streams['main']) == 1
    p.clearStream()
    assert not p.streams['main']


def test_slice_into_streams(astrofaker):
    def gmos_ads():
        ad1 = astrofaker.create("GMOS-N")
        ad1.init_default_extensions()
        ad2 = astrofaker.create("GMOS-N")
        ad2.init_default_extensions()
        return [ad1, ad2]

    # Slice, clearing "main"
    p = GMOSImage(gmos_ads())
    p.sliceIntoStreams(copy=False)
    p.clearStream()
    assert len(p.streams) == 13
    for k, v in p.streams.items():
        assert len(v) == 0 if k == 'main' else 2

    # Slice, not clearing "main"
    p = GMOSImage(gmos_ads())
    p.sliceIntoStreams(copy=True)
    assert len(p.streams) == 13
    for k, v in p.streams.items():
        assert len(v) == 2

    # Slice with different lengths of input
    ad1, ad2 = gmos_ads()
    ad2.phu['EXTRA_KW'] = 33
    del ad1[5]
    p = GMOSImage([ad1, ad2])
    p.sliceIntoStreams(copy=True)
    assert len(p.streams) == 13
    for k, v in p.streams.items():
        assert len(v) == 1 if k == 'ext12' else 2
    # The last stream should only have a slice from ad2
    assert 'EXTRA_KW' in p.streams['ext12'][0].phu


class TestBookkeeping:
    """
    Suite of tests for the functions in the primitives_standardize module.
    """

    @pytest.mark.xfail(reason="Test needs revision", run=False)
    def test_addToList(self):
        filenames = ['N20070819S{:04d}_flatCorrected.fits'.format(i)
                     for i in range(104, 109)]

        adinputs = [astrodata.from_file(os.path.join(TESTDATAPATH, 'NIRI', f))
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

        adinputs = [astrodata.from_file(os.path.join(TESTDATAPATH, 'NIRI', f))
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
