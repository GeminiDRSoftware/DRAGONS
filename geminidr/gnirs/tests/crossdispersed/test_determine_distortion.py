#!/usr/bin/env python3
"""
Test related to GNIRS Cross-dispersed spectroscopy arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import numpy as np
import os
import pytest

import astrodata, gemini_instruments
import geminidr
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from recipe_system.testing import ref_ad_factory

# Test parameters -------------------------------------------------------------
fixed_parameters_for_determine_distortion = {
    "fwidth": None,
    "id_only": False,
    "max_missed": 1,
    "max_shift": 0.05,
    "min_snr": 5.,
    "nsum": 10,
    "step": 5,
    "spatial_order": 2,
    "spectral_order": 2,
    "min_line_length": 0.8,
    "debug_reject_bad": False
}

input_pars = [
    # Process Arcs: GNIRS
    # (Input File, params)
    # 10 l/mm Longblue SXD
    ('N20170511S0269_wavelengthSolutionDetermined.fits', dict()),
    # 10 l/mm Longblue LXD
    ('N20130821S0301_wavelengthSolutionDetermined.fits', dict()),
    # 32 l/mm Shortblue SXD
    ('N20210129S0324_wavelengthSolutionDetermined.fits', dict()),
    # 111 l/mm Shortblue SXD
    ('N20231030S0034_wavelengthSolutionDetermined.fits', dict()),
    # 32 l/mm Longblue LXD
    ('N20201223S0216_wavelengthSolutionDetermined.fits', dict()),
    # 32 l/mm Shortblue SXD
    ('S20060507S0070_wavelengthSolutionDetermined.fits', dict()),
    # 111 l/mm Shortblue SXD
    ('S20060311S0321_wavelengthSolutionDetermined.fits', dict()),
]

# Tests -----------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params", input_pars, indirect=['ad'])
@pytest.mark.skip("MUST WORK")
def test_regression_for_determine_distortion_using_wcs(
        ad, params, change_working_dir, ref_ad_factory):

    with change_working_dir():
        p = GNIRSCrossDispersed([ad])
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(distortion_determined_ad.filename)

    errstr = ""
    for ext, ref_ext in zip(distortion_determined_ad, ref_ad):
        # Confirm that the distortion model is placed after the rectification model
        assert (ext.wcs.available_frames.index("distortion_corrected") >
                ext.wcs.available_frames.index("rectified"))
        assert (ref_ext.wcs.available_frames.index("distortion_corrected") >
                ref_ext.wcs.available_frames.index("rectified"))

        model = ext.wcs.get_transform("pixels", "distortion_corrected")
        ref_model = ref_ext.wcs.get_transform("pixels", "distortion_corrected")

        # Otherwise we're doing something wrong!
        assert model[-1].__class__.__name__ == ref_model[-1].__class__.__name__ == "Chebyshev2D"

        Y, X = np.mgrid[:ext.shape[0], :ext.shape[1]]

        # We only care about pixels in the illuminated region
        xx, yy = X[ext.mask == 0], Y[ext.mask == 0]
        diffs = model(xx, yy)[1] - ref_model(xx, yy)[1]  # 1 is y-axis in astropy
        try:
            np.testing.assert_allclose(diffs, 0, atol=1)
        except AssertionError as e:
            errstr += f"Extension {ext.id}\n{str(e)}"

    if errstr:
        raise AssertionError(errstr)


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
        Input spectrum processed up to right before the `distortionDetermine`
        primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad


# -- Recipe to create pre-processed data ---------------------------------------
def create_inputs_and_refs_recipe():
    from astrodata.testing import download_from_archive
    from recipe_system.reduction.coreReduce import Reduce
    from geminidr.gnirs.tests.crossdispersed import CREATED_INPUTS_PATH_FOR_TESTS

    associated_calibrations = {
        'N20170511S0269': {'flats': [f'N20170511S{i:04d}.fits' for i in range(271, 282)]},
        'N20130821S0301': {'flats': [f'N20130821S{i:04d}.fits' for i in range(302, 318)]},
        'N20210129S0324': {'flats': [f'N20210129S{i:04d}.fits' for i in range(304, 324)],
                           'pinholes': ['N20231029S0343.fits']},
        'N20231030S0034': {'flats': [f'N20231030S{i:04d}.fits' for i in range(22, 34)],
                           'pinholes': ['N20231029S0343.fits']},
        'N20201223S0216': {'flats': [f'N20201223S{i:04d}.fits' for i in range(208, 216)],
                           'pinholes': ['N20201223S0105.fits']},
        'S20060507S0070': {'flats': [f'S20060507S{i:04d}.fits' for i in range(128, 147)],
                           'pinholes': ['S20060507S0125.fits']},
        'S20060311S0321': {'flats': [f'S20060311S{i:04d}.fits' for i in range(323, 343)]},
    }

    os.makedirs(CREATED_INPUTS_PATH_FOR_TESTS, exist_ok=True)
    os.chdir(CREATED_INPUTS_PATH_FOR_TESTS)
    os.makedirs("inputs/", exist_ok=True)
    os.makedirs("refs/", exist_ok=True)
    print(f'Current working directory:\n    {os.getcwd()}')

    for filename, params in input_pars[5:]:
        # Ensure that params in input_pars only refer to determineDistortion
        user_params = {f'determineDistortion:{k}': v for k, v in params.items()}
        user_params['determineWavelengthSolution:write_outputs'] = True
        user_params['determineDistortion:write_outputs'] = True
        root_filename = filename.split("_")[0]
        cals = associated_calibrations[root_filename]

        print("Reducing flats")
        flats = [download_from_archive(f) for f in cals['flats']]
        flat_reduce = Reduce()
        flat_reduce.files.extend(flats)
        flat_reduce.runr()
        processed_flat = flat_reduce.output_filenames.pop()
        del flat_reduce
        print(f"Reduced flat is {processed_flat}")
        user_params['flat'] = processed_flat

        try:
            pinholes = [download_from_archive(f) for f in cals['pinholes']]
        except KeyError:
            print("No pinholes to reduce")
            user_params['attachPinholeRectification:do_cal'] = 'skip'
        else:
            print("Reducing pinholes")
            pinhole_reduce = Reduce()
            pinhole_reduce.uparms = {'flat': processed_flat}
            pinhole_reduce.files.extend(pinholes)
            pinhole_reduce.runr()
            processed_pinhole = pinhole_reduce.output_filenames.pop()
            del pinhole_reduce
            user_params['pinhole'] = processed_pinhole

        print("Reducing arc")
        arc = download_from_archive(f"{root_filename}.fits")
        arc_reduce = Reduce()
        arc_reduce.uparms = user_params
        arc_reduce.files.append(arc)
        arc_reduce.runr()

        processed_arc = arc_reduce.output_filenames.pop()
        input_file = processed_arc.replace("arc", "wavelengthSolutionDetermined")
        ref_file = processed_arc.replace("arc", "distortionDetermined")
        os.rename(input_file, f"inputs/{input_file}")
        os.rename(ref_file, f"refs/{ref_file}")
        print(f'Wrote pre-processed file to:\n    {processed_arc}')

if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_and_refs_recipe()
    else:
        pytest.main()
