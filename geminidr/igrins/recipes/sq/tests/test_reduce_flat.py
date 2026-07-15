import os
import pytest

import numpy as np
from astropy.coordinates import SkyCoord

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from recipe_system.reduction.coreReduce import Reduce


FLAT_INPUTS = [
    [f"N20260228S{i:04d}_K.fits" for i in range(540, 546)],
]


@pytest.fixture()
def input_files(request, path_to_inputs):
    return [os.path.join(path_to_inputs, filename)
            for filename in request.param]


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_files", FLAT_INPUTS, indirect=True)
def test_make_processed_bpm(input_files, change_working_dir, path_to_refs):
    r = Reduce()
    r.files = input_files
    r.recipename = "makeProcessedBPM"
    with change_working_dir():
        r.runr()
        output_filename = r._output_filenames.pop()
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore=["wcs", "tags"], ignore_kw=["PROCBPM", "SDZWCS"])


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_files", FLAT_INPUTS, indirect=True)
def test_make_processed_flat(input_files, change_working_dir, path_to_refs):
    r = Reduce()
    r.files = input_files
    # This avoids issues when running locally since test_make_processed_bpm
    # will add the BPM to the caldb
    r.uparms = {'addDQ:static_bpm': None}
    with change_working_dir():
        r.runr()
        output_filename = r._output_filenames.pop()
        assert r.recipename == "makeProcessedFlat"
        adout = astrodata.open(os.path.join("calibrations", "processed_flat", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        # A large tolerance is needed here because significant numerical
        # differences arise on different architectures with the Savitzky-Golay
        # smoothing in normalizeFlat()
        ad_compare(adout, adref, rtol=2e-4, ignore=["wcs"], ignore_kw=["PROCFLAT", "ADDMDF", "SDZWCS"])


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_files", FLAT_INPUTS, indirect=True)
def test_new_make_processed_flat(input_files, change_working_dir):
    """This test is for the format of the flat"""
    r = Reduce()
    r.files = input_files
    r.recipename = "newMakeProcessedFlat"
    # This avoids issues when running locally since test_make_processed_bpm
    # will add the BPM to the caldb
    r.uparms = {'addDQ:static_bpm': None}
    with change_working_dir():
        r.runr()
        output_filename = r._output_filenames.pop()
        adout = astrodata.open(os.path.join("calibrations", "processed_flat", output_filename))
        assert len(adout) == 24
        np.testing.assert_equal(adout.hdr['SPECORDR'], list(range(70, 94)))

        # WCS should be in (wavelength, RA, DEC), with 0.1" spacing
        for ext in adout:
            ymid = ext.shape[0] // 2
            wcs1 = ext.wcs(1024, ymid)
            assert len(wcs1) == 3
            c1 = SkyCoord(*wcs1[1:], unit="deg")
            c2 = SkyCoord(*ext.wcs(1024, ymid+1)[1:], unit="deg")
            assert c1.separation(c2).arcsec == pytest.approx(0.1, abs=0.001)
