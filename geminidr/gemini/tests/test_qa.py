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

import astrodata, gemini_instruments
import pytest
import logging

from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_image import GMOSImage
from geminidr.niri.primitives_niri_image import NIRIImage


# --- Fixtures ---
@pytest.fixture
def ad(path_to_inputs):
    """Has been run through prepare, addDQ, overscanCorrect, detectSources,
       addReferenceCatalog, determineAstrometricSolution"""
    ad = astrodata.from_file(os.path.join(path_to_inputs, "N20150624S0106_astrometryCorrected.fits"))
    return ad


# --- Tests ----
@pytest.mark.preprocessed_data
def test_measureBG(caplog, ad):
    """
    Check measurements of BG in GMOS image and confirm that the QA report
    properly identifies whether it is/isn't consistent with REQBG.

    We use a generous matching tolerance here because we're working with
    a roughly-processed image.
    """
    tol = 10  # matching tolerance
    caplog.set_level(logging.DEBUG)
    p = GMOSImage([ad])
    ad_out = p.measureBG()[0]

    correct = [303, 283, 318, 328, 322, 299]

    for rv, cv in zip(ad_out.hdr['SKYLEVEL'], correct):
        assert abs(rv - cv) < tol, 'Wrong background level'

    assert (ad_out.phu['SKYLEVEL'] - 310) < tol

    assert len(caplog.records) > 0
    for rec in caplog.records:
        if 'BG band' in rec.message:
            assert rec.message.split()[2] == 'BG80', 'Wrong BG band'

    # Now set REQBG to 50 and confirm it fails (it's BG80)
    caplog.clear()
    ad.phu['REQBG'] = '50-percentile'
    p.measureBG()[0]

    assert any('WARNING: BG requirement not met' in rec.message
               for rec in caplog.records), 'No BG warning'


@pytest.mark.preprocessed_data
def test_measureCC(caplog, ad):
    """
    Check measurements of CC in a GMOS image and confirm that the QA report
    properly identifies the bands
    This image returns CC = 0.11 +/- 0.04 by default
    """
    caplog.set_level(logging.DEBUG)
    p = GMOSImage([ad])
    ad = p.measureCC()[0]
    correct = [27.25, None, 27.21, 27.26, 27.24, None]
    for ext, cv in zip(ad, correct):
        rv = ext.hdr.get('MEANZP')
        if cv:
            assert abs(rv - cv) < 0.02, 'Wrong zeropoint'
        else:
            assert rv is None, 'Unexpected zeropoint'

    found = ccwarn = False
    for rec in caplog.records:
        if 'CC bands' in rec.message:
            found = True
            assert 'CC50, CC70' in rec.message, 'Wrong CC bands'
    ccwarn |= 'WARNING: CC requirement not met' in rec.message
    assert found, 'Did not find "CC bands" line in log'
    assert not ccwarn, 'CC warning raised in original data'

    caplog.clear()
    for ext in ad:
        ext.OBJCAT['MAG_AUTO'] += 0.25
    ad = p.measureCC()[0]
    ccwarn = found = False
    for rec in caplog.records:
        if 'CC bands' in rec.message:
            found = True
            assert 'CC70, CC80' in rec.message, 'Wrong CC bands after edit'
        ccwarn |= 'WARNING: CC requirement not met' in rec.message
    assert found, 'Did not find "CC bands" line in log'
    assert ccwarn, 'No CC warning in modified data'


@pytest.mark.preprocessed_data
def test_measureIQ(caplog, ad):
    """
    Check IQ measurements on GMOS image and that a warning is raised if it
    doesn't satisfy the requirement.
    """
    caplog.set_level(logging.DEBUG)
    p = GMOSImage([ad])
    ad = p.measureIQ()[0]
    # Try to give a reasonable-sized goal
    assert abs(ad.phu['MEANFWHM'] - 0.42) < 0.02, 'Wrong FWHM'
    assert abs(ad.phu['MEANELLP'] - 0.09) < 0.02, 'Wrong ellipticity'

    found = False
    for rec in caplog.records:
        if 'IQ range' in rec.message:
            found = True
            assert rec.message.split()[4] == 'IQ20', 'Wrong IQ band'
    assert found, 'Did not find "IQ range" line in log'

    caplog.clear()
    ad.phu['REQIQ'] = '70-percentile'
    for ext in ad:
        ext.OBJCAT['PROFILE_FWHM'] *= 2.5
        ext.OBJCAT['PROFILE_EE50'] *= 2.5

    ad = p.measureIQ()[0]
    found = iqwarn = False
    for rec in caplog.records:
        if 'IQ range' in rec.message:
            found = True
            assert rec.message.split()[4] == 'IQ85', 'Wrong IQ band after edit'
        iqwarn |= 'WARNING: IQ requirement not met' in rec.message
    assert found, 'Did not find "IQ range" line in log'
    assert iqwarn, 'NO IQ warning after edit'


@pytest.mark.dragons_remote_data
def test_measureIQ_no_objcat_AO(caplog):
    """Confirm we get a report with AO seeing if no OBJCAT"""
    caplog.set_level(logging.DEBUG)
    ad = astrodata.from_file(download_from_archive("N20131215S0156.fits"))
    p = NIRIImage([ad])
    p.measureIQ()

    found1 = found2 = False
    for rec in caplog.records:
        if "No OBJCAT" in rec.message:
            found1 = True
        elif "Zenith-corrected" in rec.message:
            found2 = True
            assert float(rec.message.split()[5]) == 0.796
    assert found1, "No warning about missing OBJCAT"
    assert found2, "Did not find IQ value"


@pytest.mark.dragons_remote_data
def test_measure_IQ_GMOS_thru_slit(caplog):
    """Measure on a GMOS thru-slit LS observation"""
    caplog.set_level(logging.DEBUG)
    ad = astrodata.from_file(download_from_archive("N20180521S0099.fits"))
    p = GMOSImage([ad])
    p.prepare(attach_mdf=True)
    p.addDQ()
    p.overscanCorrect()
    p.ADUToElectrons()
    p.measureIQ(display=False)
    found = False
    for rec in caplog.records:
        if 'FWHM measurement' in rec.message:
            found = True
            assert abs(float(rec.message.split()[2]) - 0.892) < 0.05
    assert found, "FWHM measurement not found in log"


@pytest.mark.dragons_remote_data
def test_measureIQ_no_objcat():
    """Confirm the primitive doesn't crash with no OBJCAT"""
    ad = astrodata.from_file(download_from_archive("N20180105S0064.fits"))
    p = GMOSImage([ad])
    p.measureIQ()[0]


@pytest.mark.dragons_remote_data
def test_measureIQ_no_objcat_GSAOI():
    """Confirm the primitive doesn't for GSAOI with no OBJCAT"""
    ad = astrodata.from_file(download_from_archive("S20150528S0112.fits"))
    p = NIRIImage([ad])
    p.measureBG()[0]


@pytest.mark.dragons_remote_data
def test_measureBG_no_zeropoint(caplog):
    """Confirm the primitive doesn't crash with no nominal_photometric_zeropoint"""
    ad = astrodata.from_file(download_from_archive("N20131215S0152.fits"))
    p = NIRIImage([ad])
    p.measureBG()[0]
