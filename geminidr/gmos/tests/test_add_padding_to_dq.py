import os
import pytest

import astrodata
import gemini_instruments
from geminidr.gmos import primitives_gmos
from gempy.utils import logutils


test_cases = [
    "N20180508S0021.fits",  # GMOS-N Ham
]


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", test_cases, indirect=True)
def test_add_padding_to_data(ad):
    """
    Test that the input did not have padded columns before calling
    `p.addPaddingToDQ()` and that the padding was properly added after
    calling the primitive in data that was not mosaicked nor tiled.
    """
    logutils.config(file_name='log_{}.txt'.format(ad.filename.strip('.fits')))
    p = primitives_gmos.GMOS([ad])
    adp = p.addPaddingToDQ().pop()


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", test_cases, indirect=True)
def test_add_padding_to_tiled_data(ad_tiled):
    """
    Test that the input did not have padded columns before calling
    `p.addPaddingToDQ()` and that the padding was properly added after
    calling the primitive in data that was not mosaicked nor tiled.
    """
    logutils.config(file_name='log_{}.txt'.format(ad.filename.strip('.fits')))
    p = primitives_gmos.GMOS([ad])
    adp = p.addPaddingToDQ().pop()


@pytest.mark.preprocessed_data
@pytest.mark.integration
def test_add_padding_to_mosaicked_data():
    """
    ToDO: failed to create input data.
    """
    pass


@pytest.fixture
def ad(request, path_to_inputs):
    filename = request.param.replace(".fits", "_dqAdded.fits")
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad


@pytest.fixture
def ad_tiled(request, path_to_inputs):
    filename = request.param.replace(".fits", "_tiled.fits")
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        _ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return _ad


def create_inputs_recipe():
    """
    Creates static input data for the `addPaddingToDQ` tests.
    """
    from astrodata.testing import download_from_archive
    from geminidr.gmos.tests import CREATED_INPUTS_PATH_FOR_TESTS

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    os.chdir("inputs/")
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename in test_cases:
        print('Downloading files...')
        path = download_from_archive(filename)
        ad = astrodata.open(path)
        logutils.config(file_name='log_{}.txt'.format(filename.strip('.fits')))

        p = primitives_gmos.GMOS([ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.writeOutputs()

        p.tileArrays()
        p.writeOutputs()

        p.mosaicDetectors()
        p.writeOutputs()


if __name__ == '__main__':
    from sys import argv
    if '--create-inputs' in argv[1:]:
        create_inputs_recipe()
