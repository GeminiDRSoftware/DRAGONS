# pytest suite
"""
Unit tests for :any:`geminidr.ghost.primitives_ghost_bundle`.

This is a suite of tests to be run with pytest.
"""
import os
import pytest
from pytest_dragons.fixtures import *

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from astrodata.testing import ad_compare

from geminidr.ghost.primitives_ghost_bundle import GHOSTBundle


@pytest.mark.dragons_remote_data
@pytest.mark.ghostbundle
def test_split_bundle(change_working_dir, path_to_refs):
    """
    This test ensures that splitBundle() produces the correct outputs

    S20230214S0025 has 1 blue, 3 red, and 5 slit images
    """
    with change_working_dir():
        ad = astrodata.open(download_from_archive("S20230214S0025.fits"))
    p = GHOSTBundle([ad])
    p.splitBundle()

    assert len(p.streams['main']) == 5

    blue_files = p.selectFromInputs(tags="BLUE", outstream='blue')
    red_files = p.selectFromInputs(tags="RED", outstream='red')
    slit_files = p.selectFromInputs(tags="SLITV")

    assert len(blue_files) == 1
    assert len(red_files) == 3
    assert len(slit_files) == 1

    for ad in blue_files + red_files:
        assert len(ad) == 4
    assert len(slit_files[0]) == 5

    # There should be one entry for each red/blue file
    sciexp = slit_files[0].SCIEXP
    assert len(sciexp) == 4

    for adout in blue_files + red_files + slit_files:
        adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
        assert ad_compare(adref, adout, ignore_kw=['GHOSTDR'])
