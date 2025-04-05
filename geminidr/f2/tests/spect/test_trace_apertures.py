#!/usr/bin/env python3
"""
Regression tests for F2 LS extraction1D. These tests run on real data to ensure
that the output is always the same. Further investigation is needed to check if
these outputs are scientifically relevant.
"""

import os
import numpy as np
import pytest

import astrodata
import geminidr

from geminidr.f2 import primitives_f2_longslit
from gempy.utils import logutils
from gempy.library import astromodels as am
from recipe_system.testing import ref_ad_factory

# Test parameters -------------------------------------------------------------
test_datasets = [
    "S20200220S0118_aperturesFound.fits",  # 2pix, J, R3K, no illumination in center
]

# Test Definitions ------------------------------------------------------------
@pytest.mark.f2
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", test_datasets, indirect=True)
def test_regression_trace_apertures(ad, change_working_dir, ref_ad_factory):

    with change_working_dir():
        logutils.config(file_name="log_regression_{}.txt".format(ad.data_label()))
        p = primitives_f2_longslit.F2Longslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.traceApertures()
        aperture_traced_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(aperture_traced_ad.filename)

    for ext, ref_ext in zip(aperture_traced_ad, ref_ad):
        input_table = ext.APERTURE
        reference_table = ref_ext.APERTURE

        assert input_table['aper_lower'][0] <= 0
        assert input_table['aper_upper'][0] >= 0

        assert len(input_table) == len(reference_table)

        for input_row, ref_row in zip(input_table, reference_table):
            input_model = am.table_to_model(input_row)
            ref_model = am.table_to_model(ref_row)
            pixels = np.arange(*input_model.domain)
            actual = input_model(pixels)
            desired = ref_model(pixels)

            np.testing.assert_allclose(desired, actual, atol=0.5)


@pytest.mark.interactive
@pytest.mark.parametrize("ad", [test_datasets[0]], indirect=True)
def test_interactive_trace_apertures(ad, change_working_dir):
    """
    Simply tests if we can run traceApertures() in interactive mode easily.

    Parameters
    ----------
    ad : fixture
        Custom fixture that loads the input AstroData object.
    change_working_dir : fixture
        Custom fixture that changes the current working directory.
    """
    with change_working_dir():
        logutils.config(file_name="log_regression_{}.txt".format(ad.data_label()))
        p = primitives_f2_longslit.F2Longslit([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.traceApertures(interactive=True)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `calculateSensitivity`
        primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.from_file(path)
    else:
        raise FileNotFoundError(path)

    return ad
