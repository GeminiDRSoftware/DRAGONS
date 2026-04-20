import os

import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive

import igrins_instruments
from igrinsdr.igrins.primitives_igrins_bundle import IgrinsBundle


def test_split_bundle():
    # Simple test to split a bundle and confirm that it creates two
    # files with the appropriate names and tags
    ad = astrodata.open(download_from_archive("N20260228S0200.fits"))
    orig_filename = ad.filename
    orig_tags = ad.tags
    p = IgrinsBundle([ad])
    p.splitBundle()
    assert len(p.adinputs) == 2
    assert p.adinputs[0].filename == orig_filename.replace(".", "_H.")
    assert p.adinputs[1].filename == orig_filename.replace(".", "_K.")
    assert p.adinputs[0].tags == (orig_tags - {'RAW', 'BUNDLE'}) | {'H', 'SPECT'}
