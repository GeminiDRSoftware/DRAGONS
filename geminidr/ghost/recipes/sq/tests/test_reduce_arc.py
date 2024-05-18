# Tests for the reduction of echelle arc images

import os
import pytest
from numpy.testing import assert_allclose

from astrodata.testing import ad_compare, download_from_archive

import astrodata, gemini_instruments
from geminidr.ghost.primitives_ghost_bundle import GHOSTBundle
from geminidr.ghost.primitives_ghost_spect import GHOSTSpect
from geminidr.ghost.recipes.sq.recipes_ARC import makeProcessedArc
from geminidr.ghost.polyfit.ghost import GhostArm


# arc bundle and root names of processed_bias and flat
datasets = [("S20230514S0006.fits", {"bias": "S20230513S0439.fits",
                                     "flat": "S20230513S0463.fits"})]


@pytest.fixture
def input_filename(change_working_dir, request):
    with change_working_dir():
        ad = astrodata.open(download_from_archive(request.param))
        p = GHOSTBundle([ad])
        adoutputs = p.splitBundle()
        return_dict = {}
        for arm in ("blue", "red"):
            return_dict[arm] = [ad for ad in adoutputs if arm.upper() in ad.tags]
    return return_dict


@pytest.mark.slow
@pytest.mark.dragons_remote_data
@pytest.mark.integration_test
@pytest.mark.ghostspect
@pytest.mark.parametrize("input_filename, caldict", datasets,
                         indirect=["input_filename"])
@pytest.mark.parametrize("arm", ("blue", "red"))
def test_reduce_arc(input_filename, caldict, arm, path_to_inputs, path_to_refs):
    """Reduce both arms of an arc bundle"""
    adinputs = input_filename[arm]
    bias, flat = caldict['bias'], caldict['flat']
    processed_bias = os.path.join(
        path_to_inputs, bias.replace(".fits", f"_{arm}001_bias.fits"))
    processed_flat = os.path.join(
        path_to_inputs, flat.replace(".fits", f"_{arm}001_flat.fits"))
    processed_slit = os.path.join(
        path_to_inputs, adinputs[0].phu['ORIGNAME'].split('_')[0] + "_slit_slit.fits")
    processed_slitflat = os.path.join(
        path_to_inputs, flat.replace(".fits", f"_slit_slitflat.fits"))
    processed_bpm = os.path.join(
        path_to_inputs, f"bpm_20220601_ghost_{arm}_11_full_4amp.fits")
    ucals = {"processed_bias": processed_bias,
             "processed_flat": processed_flat,
             "processed_slit": processed_slit,
             "processed_slitflat": processed_slitflat,
             "processed_bpm": processed_bpm}
    p = GHOSTSpect(adinputs, ucals=ucals)
    makeProcessedArc(p)
    assert len(p.streams['main']) == 1
    output_filename = p.streams['main'][0].filename
    adout = astrodata.open(os.path.join("calibrations", "processed_arc", output_filename))
    adref = astrodata.open(os.path.join(path_to_refs, output_filename))
    # Changed timestamp kw from STCKARCS -> STACKARC and don't have time to
    # re-upload reference, so just add these to the "ignore" list
    assert ad_compare(adref, adout, ignore_kw=['PROCARC', 'STACKARC', 'STCKARCS'])

    # Need to evaluate WFIT
    arm = GhostArm(arm=adout.arm(), mode=adout.res_mode())
    wfitout = arm.evaluate_poly(adout[0].WFIT)

    # We can reuse the GhostArm object since we already know that 'arm'
    # 'and res_mode' match
    wfitref = arm.evaluate_poly(adref[0].WFIT)
    assert_allclose(wfitref, wfitout)
