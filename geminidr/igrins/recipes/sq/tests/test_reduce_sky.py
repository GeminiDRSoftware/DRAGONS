import os
import pytest

import numpy as np

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from recipe_system.reduction.coreReduce import Reduce


# tuples with list of input files, dict of calibrations
SKY_INPUTS = [
    (["N20260301S0028_K.fits"], {"processed_flat": "N20260228S0543_K_flat.fits"})
]


@pytest.fixture()
def input_files(request, path_to_inputs):
    return [os.path.join(path_to_inputs, filename)
            for filename in request.param]


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('input_files, caldict', SKY_INPUTS, indirect=['input_files'])
def test_old_make_processed_arc(input_files, caldict, change_working_dir, path_to_inputs, path_to_refs):
    r = Reduce()
    r.files = input_files
    r.recipename = "oldMakeProcessedArc"
    # This avoids issues when running locally since test_make_processed_bpm
    # will add the BPM to the caldb
    r.uparms = {'addDQ:static_bpm': None}
    r.ucals = {k : os.path.join(path_to_inputs, v) for k, v in caldict.items()}
    with change_working_dir():
        r.runr()
        output_filename = r._output_filenames.pop()
        adout = astrodata.open(os.path.join("calibrations", "processed_arc", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore=["wcs"], ignore_kw=["PROCARC", "ADDMDF", "SDZWCS"])


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('input_files, caldict', SKY_INPUTS, indirect=['input_files'])
def test_make_processed_arc(input_files, caldict, change_working_dir, path_to_inputs):
    """A check on the format of the processed_arc"""
    r = Reduce()
    r.files = input_files
    # This avoids issues when running locally since test_make_processed_bpm
    # will add the BPM to the caldb
    r.uparms = {'addDQ:static_bpm': None}
    r.ucals = {k : os.path.join(path_to_inputs, v.replace("_flat", "_flat_dragons"))
               for k, v in caldict.items()}
    with change_working_dir():
        r.runr()
        assert r.recipename == "makeProcessedArc"
        output_filename = r._output_filenames.pop()
        adout = astrodata.open(os.path.join("calibrations", "processed_arc", output_filename))

        assert len(adout) == 24
        np.testing.assert_equal(adout.hdr['SPECORDR'], list(range(70, 94)))

        # WCS should be in (wavelength, slit coords in arcsec)
        for ext in adout:
            ymid = ext.shape[0] // 2
            wcs1 = ext.wcs(1024, ymid)
            assert len(wcs1) == 2
            slitlen_pix = np.diff(ext.SLITEDGE['c0'] + ext.SLITEDGE['c2'])[0]
            assert abs(wcs1[1] - ext.wcs(1024, ymid+1)[1]) == pytest.approx(5. / slitlen_pix, rel=0.05)
