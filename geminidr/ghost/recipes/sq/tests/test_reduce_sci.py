# Tests for the reduction of echelle science images

import os
import pytest

import pytest_dragons
from pytest_dragons.fixtures import *
from astrodata.testing import ad_compare, download_from_archive

import astrodata, gemini_instruments
from geminidr.ghost.primitives_ghost_bundle import GHOSTBundle
from geminidr.ghost.primitives_ghost_spect import GHOSTSpect
from geminidr.ghost.recipes.sq.recipes_SPECT import reduceScience


# arc bundle and root names of processed_bias and flat
datasets = [("S20230514S0022.fits", {"bias": "S20230513S0013.fits",
                                     "flat": "S20230511S0035.fits",
                                     "arc": "S20230510S0034.fits",
                                     "slit": "S20230514S0022_slit_blue001_red001_slit.fits",
                                     "standard": "S20230513S0229.fits"})]


@pytest.fixture
def input_filename(request):
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
@pytest.mark.parametrize("skysub", (True, False))
def test_reduce_science(input_filename, caldict, arm, skysub, path_to_inputs,
                        path_to_outputs, path_to_refs, change_working_dir):
    """Reduce both arms of a science bundle"""
    adinputs = input_filename[arm]
    bias, flat, arc, slit = caldict['bias'], caldict['flat'], caldict['arc'], caldict.get('slit')
    if isinstance(bias, dict):
        bias = bias[arm]
    else:
        bias = bias.replace(".fits", f"_{arm}001_bias.fits")
    if isinstance(flat, dict):
        flat = flat[arm]
        # will be "SyyyymmddSnnnn_arm001_flat.fits"
        slitflat = flat.split("_")[0] + "_slit_slitflat.fits"
    else:
        slitflat = flat.replace(".fits", "_slit_slitflat.fits")
        flat = flat.replace(".fits", f"_{arm}001_flat.fits")
    if slit is None:
        slit = [ad.filename.replace(".fits", "_slit.fits") for ad in adinputs]
    processed_bias = os.path.join(path_to_inputs, bias)
    processed_flat = os.path.join(path_to_inputs, flat)
    processed_slit = os.path.join(path_to_inputs, slit)
    processed_arc = os.path.join(path_to_inputs, arc.replace(".fits", f"_{arm}001_arc.fits"))
    processed_slitflat = os.path.join(path_to_inputs, slitflat)
    processed_bpm = os.path.join(
        path_to_inputs, f"bpm_20220601_ghost_{arm}_11_full_4amp.fits")
    ucals = {"processed_bias": processed_bias,
             "processed_flat": processed_flat,
             "processed_arc": processed_arc,
             "processed_slit": processed_slit,
             "processed_slitflat": processed_slitflat,
             "processed_bpm": processed_bpm}
    uparms = {"extractSpectra:sky_subtract": skysub}
    standard = caldict.get('standard')
    if standard:
        standard = standard.replace(".fits", f"_{arm}001_standard.fits")
        uparms["responseCorrect:standard"] = os.path.join(path_to_inputs, standard)
    p = GHOSTSpect(adinputs, ucals=ucals, uparms=uparms)
    with change_working_dir():
        reduceScience(p)
        assert len(p.streams['main']) == 1
        adout = p.streams['main'].pop()
        output_filename = adout.filename
        adref = astrodata.open(os.path.join(
            path_to_refs, f"skysub_{skysub}", output_filename))
        assert ad_compare(adref, adout)

        # Now compare the _calibrated.fits files (not order-combined)
        intermediate_filename = output_filename.replace("_dragons", "_calibrated")
        adout = astrodata.open(os.path.join(path_to_outputs, "outputs", intermediate_filename))
        adref = astrodata.open(os.path.join(
            path_to_refs, f"skysub_{skysub}", intermediate_filename))
        assert ad_compare(adref, adout)
