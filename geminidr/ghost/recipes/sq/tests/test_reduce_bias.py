# Tests for the reduction of echelle bias images

import os
import pytest

from astrodata.testing import ad_compare, download_from_archive

import astrodata, gemini_instruments
from geminidr.ghost.primitives_ghost_bundle import GHOSTBundle
from geminidr.ghost.primitives_ghost_spect import GHOSTSpect
from geminidr.ghost.recipes.sq.recipes_BIAS import makeProcessedBias


datasets = ["S20230513S0439.fits"]


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


@pytest.mark.dragons_remote_data
@pytest.mark.integration_test
@pytest.mark.ghostspect
@pytest.mark.parametrize("input_filename", datasets, indirect=True)
@pytest.mark.parametrize("arm", ("blue", "red"))
def test_reduce_bias(change_working_dir, path_to_inputs, input_filename, arm, path_to_refs):
    """Reduce an arm of a bias bundle"""
    with change_working_dir():
        adinputs = input_filename[arm]
        processed_bpm = os.path.join(
            path_to_inputs, f"bpm_20220601_ghost_{arm}_11_full_4amp.fits")
        ucals = {"processed_bpm": processed_bpm}
        p = GHOSTSpect(adinputs, ucals=ucals)
        makeProcessedBias(p)
        assert len(p.streams['main']) == 1
        output_filename = p.streams['main'][0].filename
        adout = astrodata.open(os.path.join("calibrations", "processed_bias", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        assert ad_compare(adref, adout, ignore_kw=['PROCBIAS', 'OVERRDNS'])
