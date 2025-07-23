import logging
import os
import pytest

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed


@pytest.mark.gnirsxd
@pytest.mark.wavecal
@pytest.mark.preprocessed_data
def test_determine_wavelength_solution_wcs_has_2_world_axes(path_to_inputs, caplog):
    """
    Check that the WCS has two world axes after determining the wavelength
    solution. This dataset fails to find a solution on at least one extension,
    so is a good check that the WCS is correct regardless of whether a solution
    is found or not.
    """
    caplog.set_level(logging.INFO, logger="geminidr")
    ad = astrodata.open(os.path.join(path_to_inputs,
                                     "N20190928S0085_aperturesFound.fits"))
    p = GNIRSCrossDispersed([ad])
    adout = p.determineWavelengthSolution(absorption=True).pop()

    for record in caplog.records:
        if "Matched 0/0" in record.message:
            break
    else:
        pytest.fail("No log message indicating that no solution was found.")

    for ext in adout:
        assert ext.wcs.output_frame.naxes == 2, f"Problem with extension {ext.id}"
