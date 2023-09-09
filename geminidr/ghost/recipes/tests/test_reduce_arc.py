# Tests for the reduction of echelle arc images

import os
import pytest
from numpy.testing import assert_allclose

import pytest_dragons
from pytest_dragons.fixtures import *
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
def input_filename(request):
    ad = astrodata.open(download_from_archive(request.param))
    p = GHOSTBundle([ad])
    adoutputs = p.splitBundle()
    return_dict = {}
    for arm in ("blue", "red"):
        return_dict[arm] = [ad for ad in adoutputs if arm.upper() in ad.tags]
    return return_dict


@pytest.mark.dragons_remote_data
@pytest.mark.integration_test
@pytest.mark.ghost
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
    ucals = {(ad.calibration_key(), "processed_bias"):
                 processed_bias for ad in adinputs}
    # processed_slit and processed_slitflat not recognized as a user_cal
    # while processed_flat fails because of the -STACK (it's needed for
    # extractProfile and fitWavelength)
    uparms = {"flat": processed_flat,
              "extractProfile:slit": processed_slit,
              "extractProfile:slitflat": processed_slitflat}
    p = GHOSTSpect(adinputs, ucals=ucals, uparms=uparms)
    makeProcessedArc(p)
    assert len(p.streams['main']) == 1
    adout = p.streams['main'].pop()
    output_filename = adout.filename
    adref = astrodata.open(os.path.join(path_to_refs, output_filename))
    assert ad_compare(adref, adout)

    # Need to evaluate WFIT
    arm = GhostArm(arm=adout.arm(), mode=adout.res_mode())
    wfitout = arm.evaluate_poly(adout[0].WFIT)

    # We can reuse the GhostArm object since we already know that 'arm'
    # 'and res_mode' match
    wfitref = arm.evaluate_poly(adref[0].WFIT)
    assert_allclose(wfitref, wfitout)
