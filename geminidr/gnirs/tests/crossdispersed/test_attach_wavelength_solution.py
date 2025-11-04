import pytest
import os

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_attach_wavelength_solution_missing_distortion_models(path_to_inputs):
    """
    Simple test to check that the attachWavelengthSolution primitive works if
    extensions are missing distortion models.
    """
    ad = astrodata.open(os.path.join(path_to_inputs, "N20200818S0038_flatCorrected.fits"))
    arc = astrodata.open(os.path.join(path_to_inputs, "N20200818S0350_arc.fits"))

    p = GNIRSCrossDispersed([ad])
    p.attachWavelengthSolution(arc=arc)

    num_ext_without_distortion = 0
    for ext, arc_ext in zip(p.adinputs[0], arc):
        assert (("distortion_corrected" in ext.wcs.available_frames) ==
                ("distortion_corrected" in arc_ext.wcs.available_frames))
        if "distortion_corrected" not in ext.wcs.available_frames:
            num_ext_without_distortion += 1

    # Because this test is specifically for the case where some extensions
    # are missing distortion models!
    assert num_ext_without_distortion > 0, "At least one extension should be missing distortion model"
