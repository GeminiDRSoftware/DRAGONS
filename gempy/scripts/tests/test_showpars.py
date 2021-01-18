import pytest
from astrodata.testing import download_from_archive
from gempy.scripts import showpars


@pytest.fixture
def testfile(scope='module'):
    return download_from_archive('S20190213S0084.fits')


def test_showpars(testfile, capsys):
    showpars.main([testfile, 'fixPixels'])
    captured = capsys.readouterr()

    expected = """\
Settable parameters on 'fixPixels':
========================================
Name                 Current setting      Description

suffix               '_pixelsFixed'       Filename suffix
regions              None                 Regions to fix, e.g. "450,521; 430:437,513:533"
regions_file         None                 Path to a file containing the regions to fix
axis                 None                 Axis over which the interpolation is done (Fortran order)
use_local_median     False                Use a local median filter for single pixels?
"""

    out = captured.out.splitlines()
    for line in expected.splitlines():
        if line:
            assert line in out

    assert "Docstring for 'fixPixels':" not in captured.out


def test_showpars_with_docstring(testfile, capsys):
    showpars.main([testfile, 'fixPixels', '--doc'])
    captured = capsys.readouterr()

    expected = """\
Settable parameters on 'fixPixels':
========================================
Name                 Current setting      Description

suffix               '_pixelsFixed'       Filename suffix
regions              None                 Regions to fix, e.g. "450,521; 430:437,513:533"
regions_file         None                 Path to a file containing the regions to fix
axis                 None                 Axis over which the interpolation is done (Fortran order)
use_local_median     False                Use a local median filter for single pixels?

Docstring for 'fixPixels':
========================================

This primitive replaces bad pixels by linear interpolation along
lines or columns using the nearest good pixels, similar to IRAF's
fixpix.
"""

    out = captured.out.splitlines()
    for line in expected.splitlines():
        if line:
            assert line in out
