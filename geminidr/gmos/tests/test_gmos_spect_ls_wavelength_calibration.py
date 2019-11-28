#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.
"""
import os

import numpy as np
# noinspection PyPackageRequirements
import pytest

import astrodata
# noinspection PyUnresolvedReferences
import gemini_instruments
import geminidr
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils

# noinspection PyPackageRequirements

input_files = [
    "process_arcs/GMOS/N20100115S0346_mosaic.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/N20130112S0390_mosaic.fits",  # B600:0.500 E2V
    "process_arcs/GMOS/N20170609S0173_mosaic.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/N20170403S0452_mosaic.fits",  # B600:0.590 HAM Full Frame 1x1
]


@pytest.fixture(scope="session", autouse=True)
def setup_log(tmp_path_factory):
    """
    Fixture that setups DRAGONS' logging system to avoid duplicated outputs.

    Parameters
    ----------
    tmp_path_factory : fixture
        PyTest's built-in session-scoped fixture that creates a temporary
        directory. It can be set using `--basetemp=mydir` command line argument.
    """
    output_dir = tmp_path_factory.getbasetemp()

    log_file = os.path.join(
        output_dir,
        "{}.log".format(os.path.splitext(os.path.basename(__file__))[0]),
    )

    logutils.config(mode="standard", file_name=log_file)


@pytest.fixture(scope="module")
def ad_in(request, path_to_inputs):
    force_preprocess = request.config.getoption("--force-preprocess-data")

    fname = os.path.join(path_to_inputs, request.param)

    if os.path.exists(fname):
        print("\n Loading existing input file: {:s}\n".format(fname))
        _ad = astrodata.open(fname)

    elif force_preprocess:

        print("\n Pre-processing input file: {:s}\n".format(fname))
        subpath, basename = os.path.split(request.param)
        basename, extension = os.path.splitext(basename)
        basename = basename.split('_')[0] + extension

        raw_fname = testing.download_from_archive(basename, path=subpath)

        _ad = astrodata.open(raw_fname)

        _p = primitives_gmos_spect.GMOSSpect([_ad])
        _p.viewer = geminidr.dormantViewer(_p, None)

        # todo: find a better way to write ad to other directories
        cwd = os.getcwd()
        os.chdir(os.path.join(path_to_inputs, subpath))

        _p.prepare()
        _p.addDQ(static_bpm=None)
        _p.addVAR(read_noise=True)
        _p.overscanCorrect()
        _p.ADUToElectrons()
        _p.addVAR(poisson_noise=True)
        _p.mosaicDetectors()
        _p.makeIRAFCompatible()
        _ad = _p.writeOutputs()[0]

        print(fname)

        os.chdir(cwd)

    else:
        raise IOError("Cannot find input file:\n {:s}".format(fname))

    return _ad


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", input_files, indirect=True)
def test_reduced_arcs_contain_wavelength_solution_model_with_expected_rms(ad_in):
    """
    Make sure that the WAVECAL model was fitted with an RMS smaller
    than 0.5.
    """
    _p = primitives_gmos_spect.GMOSSpect([])

    ad_out = _p.determineWavelengthSolution([ad_in])

    for ext in ad_out:
        if not hasattr(ext, "WAVECAL"):
            continue

        table = ext.WAVECAL
        coefficients = table["coefficients"]
        rms = coefficients[table["name"] == "rms"]

        np.testing.assert_array_less(rms, 0.5)
