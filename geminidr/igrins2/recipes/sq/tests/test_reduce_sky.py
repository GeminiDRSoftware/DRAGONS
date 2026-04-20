import os
import pytest

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from recipe_system.reduction.coreReduce import Reduce

import igrins_instruments


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
def test_make_processed_arc(input_files, caldict, change_working_dir, path_to_inputs, path_to_refs):
    r = Reduce()
    r.files = input_files
    r.drpkg = "igrinsdr"
    r.ucals = {k : os.path.join(path_to_inputs, v) for k, v in caldict.items()}
    with change_working_dir():
        r.runr()
        output_filename = r._output_filenames.pop()
        assert r.recipename == "makeProcessedArc"
        adout = astrodata.open(os.path.join("calibrations", "processed_arc", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore_kw=['PROCARC'])
