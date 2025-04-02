import os
import pytest

import astrodata, gemini_instruments
from geminidr.ghost.primitives_ghost_spect import GHOSTSpect


@pytest.mark.slow
@pytest.mark.preprocessed_data
@pytest.mark.ghostspect
def test_synthetic_slit_profile(path_to_inputs):
    """Simple test to confirm that spectra can be extracted
    with a synthetic slit profile. The output is not checked"""
    sci_filename = "S20230513S0229_red001_arraysTiled.fits"
    raw_flat_filename = "S20230511S0035.fits"
    ad = astrodata.from_file(os.path.join(path_to_inputs, sci_filename))
    arm = ad.arm()
    processed_flat = os.path.join(
        path_to_inputs,
        raw_flat_filename.replace(".fits", f"_{arm}001_flat.fits"))
    processed_slitflat = os.path.join(
        path_to_inputs,
        raw_flat_filename.replace(".fits", f"_slit_slitflat.fits"))
    ucals = {"processed_flat": processed_flat,
             "processed_slitflat": processed_slitflat}
    p = GHOSTSpect([ad], ucals=ucals)
    p.extractSpectra(seeing=0.7)
