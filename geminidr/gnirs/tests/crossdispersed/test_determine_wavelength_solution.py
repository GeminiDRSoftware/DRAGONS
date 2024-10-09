#!/usr/bin/env python3
"""
Tests for wavelength solution determination in GNIRS cross-dispsersed data.
Covers both the initial central wavelength/dispersion value for each order and
the actual step of finding the wavelength solution.
"""

import os
import logging

import numpy as np
import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
import geminidr
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from geminidr.gnirs.tests.crossdispersed import CREATED_INPUTS_PATH_FOR_TESTS
from gempy.library import astromodels as am
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce

# -- Test parameters ----------------------------------------------------------
determine_wavelength_solution_parameters = {
    'order': 3,
    'debug_min_lines': '50, 20',
    'center': None,
    'nsum': 10,
    'min_snr': 3,
    'min_sep': 2,
    'weighting': 'global',
    'fwidth': None,
    'central_wavelength': None,
    'dispersion': None,
    'linelist': None,
    'absorption': False,
    'debug_alternative_centers': False,
    'resolution': None,
    'combine_method': 'mean',
    }

datasets = {
    # 10 l/mm Longblue SXD
    "GN-2016A-Q-7": {
        "arc": ["N20170511S0269.fits"],
        "flats": [f"N20170511S{i:04d}.fits" for i in range(271, 282)],
        "user_pars": {}
        },
    # 10 l/mm Longblue LXD
    "GN-2013B-Q-41": {
        "arc": ["N20130821S0301.fits"],
        "flats": [f"N20130821S{i:04d}.fits" for i in range(302, 318)],
        "user_pars": {}
        },
    # 32 l/mm Shortblue SXD
    "GN-2021A-Q-215": {
        "arc": ["N20210129S0324.fits"],
        "flats": [f"N20210129S{i:04d}.fits" for i in range(304, 324)],
        "user_pars": {}
        },
    # 111 l/mm Shortblue SXD (south)
    "GS-2006A-Q-9": {
        "arc": ["S20060311S0321.fits"],
        "flats": [f"S20060311S{i:04d}.fits" for i in (323, 324, 325, 326, 327,
                                                      333, 334, 335, 336, 337)],
        "user_pars": {}
        },
    # 111 l/mm Shortblue SXD (north)
    "GN-2020B-Q-323": {
        "arc": ["N20210131S0104.fits"],
        "flats": [f"N20210131S{i:04d}.fits" for i in range(92, 101)],
        "user_pars": {}
        },
    }

# Format is flat, arc, user_pars
input_pars = [
    ("N20170511S0274_flat.fits", "N20170511S0269_arc.fits", dict()), # 10 l/mm Longblue SXD
    ("N20130821S0308_flat.fits", "N20130821S0301_arc.fits", dict()), # 10 l/mm Longblue LXD
    ("N20210129S0314_flat.fits", "N20210129S0324_arc.fits", dict()), # 32 l/mm Shortblue SXD
    ("S20060311S0333_flat.fits", "S20060311S0321_arc.fits", dict()), # 111 l/mm Shortblue SXD
    ("N20210131S0096_flat.fits", "N20210131S0104_arc.fits", dict()), # 111 l/mm Shortblue SXD
    ]

# -- Test definitions ---------------------------------------------------------
@pytest.mark.wavecal
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("flat, arc, params", input_pars, indirect=['arc', 'flat'])
def test_regression_determine_wavelength_solution(
        flat, arc, params, caplog, change_working_dir, path_to_refs, request):
    """Make sure that the wavelength solution gives same results on different
    runs.
    """
    caplog.set_level(logging.INFO, logger="geminidr")

    with change_working_dir():
        logutils.config(file_name='log_regress_{:s}.txt'.format(arc.data_label()))
        p = GNIRSCrossDispersed([arc])
        p.viewer = geminidr.dormantViewer(p, None)

        p.flatCorrect(flat=flat)
        p.determineWavelengthSolution(**{**determine_wavelength_solution_parameters,
                                         **params})

        wcalibrated_ad = p.streams["main"][0]

        for record in caplog.records:
            if record.levelname == "WARNING":
                assert "No acceptable wavelength solution found" not in record.message

    ref_ad = astrodata.open(os.path.join(path_to_refs, wcalibrated_ad.filename))
    model = am.get_named_submodel(wcalibrated_ad[0].wcs.forward_transform, "WAVE")
    ref_model = am.get_named_submodel(ref_ad[0].wcs.forward_transform, "WAVE")

    x = np.arange(wcalibrated_ad[0].shape[1])
    wavelength = model(x)
    ref_wavelength = ref_model(x)

    pixel_scale = wcalibrated_ad[0].pixel_scale()  # arcsec / px
    slit_size_in_px = wcalibrated_ad[0].slit_width() / pixel_scale
    dispersion = abs(wcalibrated_ad[0].dispersion(asNanometers=True))  # nm / px

    # We don't care about what the wavelength solution is doing at
    # wavelengths outside where we've matched lines
    lines = ref_ad[0].WAVECAL["wavelengths"].data
    indices = np.where(np.logical_and(ref_wavelength > lines.min(),
                                      ref_wavelength < lines.max()))
    tolerance = 0.5 * (slit_size_in_px * dispersion)

    np.testing.assert_allclose(wavelength[indices], ref_wavelength[indices],
                           atol=tolerance)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def arc(path_to_inputs, request):
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
        Input spectrum processed up to right before the
        `determineWavelengthSolution` primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad

@pytest.fixture(scope='function')
def flat(path_to_inputs, request):
    """
    Returns the pre-processed flat file. Same as `ad`.

    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad

# -- Recipe to create pre-processed data --------------------------------------
def create_inputs_recipe():
    """
    Creates input data for tests using flats and arcs.

    """

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for dataset in datasets.values():
        print("Downloading files...")
        # basename_f = flat.split("_")[0] + ".fits"
        # basename_a = arc.split("_")[0] + ".fits"
        flats = [astrodata.open(download_from_archive(filename)) for filename
                 in dataset["flats"]]
        arc = astrodata.open(download_from_archive(dataset["arc"].pop()))

        print("Reducing flat...")
        p = GNIRSCrossDispersed(flats)
        p.prepare(bad_wcs="new")
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=True)
        p.selectFromInputs(tags='GCAL_IR_ON,LAMPON', outstream='IRHigh')
        p.removeFromInputs(tags='GCAL_IR_ON,LAMPON')
        p.stackFlats(stream='main')
        p.stackFlats(stream='IRHigh')
        # Illumination of all orders from QH lamp is sufficient to find edges.
        p.determineSlitEdges(stream='main', search_radius=30)
        p.transferAttribute(stream='IRHigh', source='main', attribute='SLITEDGE')
        p.cutSlits(stream='main')
        p.cutSlits(stream='IRHigh')
        # Bring slit 1 from IRHigh stream to main (1-indexed).
        p.combineSlices(from_stream='IRHigh', ids='1')
        p.clearStream(stream='IRHigh')
        p.maskBeyondSlit()
        p.normalizeFlat()
        p.thresholdFlatfield()
        os.chdir("inputs/")
        processed_flat = p.writeOutputs(suffix="_flat", strip=True).pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_flat.filename))

        print("Reducing arc...")
        p = GNIRSCrossDispersed([arc])
        p.prepare(bad_wcs="new")
        p.addDQ()
        p.addVAR(read_noise=True)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        os.chdir("inputs/")
        processed_arc = p.writeOutputs(suffix="_arc", strip=True).pop()
        os.chdir("../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_arc.filename))

def create_refs_recipe():
    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("refs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for flat, arc, params in input_pars:
        ad_arc = astrodata.open(os.path.join('inputs', arc))
        ad_flat = astrodata.open(os.path.join('inputs', flat))
        p = GNIRSCrossDispersed([ad_arc])
        p.flatCorrect(flat=ad_flat)
        # p.attachPinholeModel() Shouldn't need this for the test
        p.determineWavelengthSolution(**{**determine_wavelength_solution_parameters,
                                         **params})
        os.chdir('refs/')
        p.writeOutputs()
        os.chdir('..')

if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    if "--create-refs" in sys.argv[1:]:
        create_refs_recipe()
    else:
        pytest.main()
