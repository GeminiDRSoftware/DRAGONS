# pytest suite
"""
Tests for the NIRI.nonlinearityCorrect() primitive

This is a suite of tests to be run with pytest.
"""
import os
import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from astrodata.testing import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage
from recipe_system.testing import ref_ad_factory

datasets = ["N20070819S0104_varAdded.fits"]


@pytest.mark.niri
@pytest.mark.parametrize("ad", datasets, indirect=True)
@pytest.mark.preprocessed_data
@pytest.mark.regression
def test_regression_nonlinearity_correct(ad, ref_ad_factory):
    p = NIRIImage([ad])
    ad_out = p.nonlinearityCorrect().pop()
    ad_ref = ref_ad_factory(ad_out.filename)

    assert ad_compare(ad_out, ad_ref, rtol=3.5e-7)

# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)
    return ad

# -- Recipe to create pre-processed data -------------------------------------
def create_inputs():
    """
    Creates input file(s) for the nonlinearityCorrect() test
    """
    root_path = os.path.join("./dragons_test_inputs/")
    module_path = f"geminidr/gmos/recipes/ql/{__file__.split('.')[0]}/"
    path = os.path.join(root_path, module_path, "inputs/")

    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename in datasets:
        raw_filename = filename.replace("_varAdded", "")
        print('Downloading files...')
        sci_path = download_from_archive(raw_filename)
        ad = astrodata.open(sci_path)

        print(f'Reducing {raw_filename}')
        p = NIRIImage([ad])
        p.prepare()
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.writeOutputs()


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs()
    else:
        pytest.main()
