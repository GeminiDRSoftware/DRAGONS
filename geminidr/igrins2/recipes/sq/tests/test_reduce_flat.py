import os
import pytest

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
        adout = astrodata.open(os.path.join("calibrations", "processed_bpm", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore_kw=['PROCBPM'])


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_files", FLAT_INPUTS, indirect=True)
def test_make_processed_flat(input_files, change_working_dir, path_to_refs):
    r = Reduce()
    r.files = input_files
    with change_working_dir():
        r.runr()
        output_filename = r._output_filenames.pop()
        assert r.recipename == "makeProcessedFlat"
        adout = astrodata.open(os.path.join("calibrations", "processed_flat", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
