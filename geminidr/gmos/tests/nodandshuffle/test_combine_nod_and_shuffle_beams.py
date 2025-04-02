import pytest

import os, re

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

# Use the same dataset, but with and without source alignment attempt
# There's an emission line so alignment only works with a specified region
# and the second test should fail to cross-correlate and revert to the offset
datasets = [("N20180908S0020_distortionCorrected.fits", {}, -371.75),
            ("N20180908S0020_distortionCorrected.fits", {"align_sources": True}, -371.75),
            ("N20180908S0020_distortionCorrected.fits", {"align_sources": True, "region": "2855:2865"}, -376.0),
            ]


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,kwargs,result", datasets, indirect=["ad"])
def test_combine_nod_and_shuffle_beams(ad, kwargs, result, caplog):
    p = GMOSLongslit([ad])
    p.combineNodAndShuffleBeams(**kwargs)
    for rec in caplog.records:
        m = re.match(r".*? (-?\d*.\d*) pixels", rec.message)
        if m:
            assert abs(float(m.group(1)) - result) < 1.0
            break


@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """Return AD object in input directory"""
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return astrodata.from_file(path)
    raise FileNotFoundError(path)
