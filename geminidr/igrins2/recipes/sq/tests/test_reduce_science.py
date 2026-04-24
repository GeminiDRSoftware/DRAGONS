import os
import pytest

from numpy.testing import assert_allclose

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from recipe_system.reduction.coreReduce import Reduce


# tuples with list of input files, dict of calibrations
SCI_INPUTS = [
    ([f"N20260303S{i:04d}_K.fits" for i in range(33, 37)], {"processed_flat": "N20260228S0543_K_flat.fits",
                                                            "processed_arc": "N20260301S0028_K_arc.fits"})
]


@pytest.fixture()
def input_files(request, path_to_inputs):
    return [os.path.join(path_to_inputs, filename)
            for filename in request.param]


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('input_files, caldict', SCI_INPUTS, indirect=['input_files'])
def test_make_processed_std(input_files, caldict, change_working_dir, path_to_inputs,
                            path_to_refs):
    r = Reduce()
    r.files = input_files
    r.ucals = {k : os.path.join(path_to_inputs, v) for k, v in caldict.items()}
    with change_working_dir():
        r.runr()
        assert r.recipename == "makeTgt"
        output_filename = r._output_filenames.pop()
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adref, adout, ignore_kw=[k for k in adref[0].hdr['WAT2*']])
        assert_allclose(adref[0].WAVELENGTHS, adout[0].WAVELENGTHS)

        output_filename = output_filename.replace("1d", "2d")
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adref, adout, ignore_kw=[k for k in adref[0].hdr['WAT2*']])
        assert_allclose(adref[0].WAVELENGTHS, adout[0].WAVELENGTHS)

        output_filename = output_filename.replace("2d", "_debug")
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adref, adout, ignore_kw=[k for k in adref[0].hdr['WAT2*']])

