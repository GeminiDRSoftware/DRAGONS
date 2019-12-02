#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import os
from warnings import warn

import numpy as np
import pytest

import astrodata
import geminidr
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect
from gempy.library import astromodels
from gempy.utils import logutils
from .plots_gmos_spect_longslit_arcs import PlotGmosSpectLongslitArcs

input_files = [
    # Process Arcs: GMOS-N ---
    "process_arcs/GMOS/N20100115S0346_mosaic.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/N20130112S0390_mosaic.fits",  # B600:0.500 E2V
    "process_arcs/GMOS/N20170609S0173_mosaic.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/N20170403S0452_mosaic.fits",  # B600:0.590 HAM Full Frame 1x1
    "process_arcs/GMOS/N20170415S0255_mosaic.fits",  # B600:0.590 HAM Central Spectrum 1x1
    "process_arcs/GMOS/N20171016S0010_mosaic.fits",  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    "process_arcs/GMOS/N20171016S0127_mosaic.fits",  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    "process_arcs/GMOS/N20100307S0236_mosaic.fits",  # B1200:0.445 EEV
    "process_arcs/GMOS/N20130628S0290_mosaic.fits",  # B1200:0.420 E2V
    "process_arcs/GMOS/N20170904S0078_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/N20170627S0116_mosaic.fits",  # B1200:0.520 HAM
    # "process_arcs/GMOS/N20100830S0594_mosaic.fits",  # R150:0.500 EEV - todo: wavelength solution not stable
    "process_arcs/GMOS/N20100702S0321_mosaic.fits",  # R150:0.700 EEV
    "process_arcs/GMOS/N20130606S0291_mosaic.fits",  # R150:0.550 E2V
    "process_arcs/GMOS/N20130112S0574_mosaic.fits",  # R150:0.700 E2V
    # "process_arcs/GMOS/N20130809S0337_mosaic.fits",  # R150:0.700 E2V - todo: RMS > 0.5 (RMS = 0.59)
    # "process_arcs/GMOS/N20140408S0218_mosaic.fits",  # R150:0.700 E2V - todo: RMS > 0.5 (RMS = 0.51)
    # "process_arcs/GMOS/N20180119S0232_mosaic.fits",  # R150:0.520 HAM - todo: RMS > 0.5 (RMS = 0.73)
    # "process_arcs/GMOS/N20180516S0214_mosaic.fits",  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2 - todo: fails test_distortion_model_is_the_same
    "process_arcs/GMOS/N20171007S0439_mosaic.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20171007S0441_mosaic.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20101212S0213_mosaic.fits",  # R400:0.550 EEV
    "process_arcs/GMOS/N20100202S0214_mosaic.fits",  # R400:0.700 EEV
    "process_arcs/GMOS/N20130106S0194_mosaic.fits",  # R400:0.500 E2V
    "process_arcs/GMOS/N20130422S0217_mosaic.fits",  # R400:0.700 E2V
    "process_arcs/GMOS/N20170108S0210_mosaic.fits",  # R400:0.660 HAM
    "process_arcs/GMOS/N20171113S0135_mosaic.fits",  # R400:0.750 HAM
    "process_arcs/GMOS/N20100427S1276_mosaic.fits",  # R600:0.675 EEV
    # "process_arcs/GMOS/N20180120S0417_mosaic.fits",  # R600:0.860 HAM - todo: RMS > 0.5 (RMS = 0.58)
    "process_arcs/GMOS/N20100212S0143_mosaic.fits",  # R831:0.450 EEV
    "process_arcs/GMOS/N20100720S0247_mosaic.fits",  # R831:0.850 EEV
    "process_arcs/GMOS/N20130808S0490_mosaic.fits",  # R831:0.571 E2V
    "process_arcs/GMOS/N20130830S0291_mosaic.fits",  # R831:0.845 E2V
    "process_arcs/GMOS/N20170910S0009_mosaic.fits",  # R831:0.653 HAM
    "process_arcs/GMOS/N20170509S0682_mosaic.fits",  # R831:0.750 HAM
    # "process_arcs/GMOS/N20181114S0512_mosaic.fits",  # R831:0.865 HAM - todo: RMS > 0.5 (RMS = 0.52) | `gswavelength` cannot find solution either.
    "process_arcs/GMOS/N20170416S0058_mosaic.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20170416S0081_mosaic.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20180120S0315_mosaic.fits",  # R831:0.865 HAM
    # # Process Arcs: GMOS-S ---
    # "process_arcs/GMOS/S20130218S0126_mosaic.fits",  # B600:0.500 EEV - todo: breaks p.determineWavelengthSolution() | `gswavelength` cannot find solution either.
    "process_arcs/GMOS/S20130111S0278_mosaic.fits",  # B600:0.520 EEV
    "process_arcs/GMOS/S20130114S0120_mosaic.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20130216S0243_mosaic.fits",  # B600:0.480 EEV
    "process_arcs/GMOS/S20130608S0182_mosaic.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20131105S0105_mosaic.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20140504S0008_mosaic.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20170103S0152_mosaic.fits",  # B600:0.600 HAM
    "process_arcs/GMOS/S20170108S0085_mosaic.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/S20130510S0103_mosaic.fits",  # B1200:0.450 EEV
    "process_arcs/GMOS/S20130629S0002_mosaic.fits",  # B1200:0.525 EEV
    "process_arcs/GMOS/S20131123S0044_mosaic.fits",  # B1200:0.595 EEV
    "process_arcs/GMOS/S20170116S0189_mosaic.fits",  # B1200:0.440 HAM - todo: very weird non-linear plot | non-linear plot using `gswavelength` seems fine.
    "process_arcs/GMOS/S20170103S0149_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170730S0155_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20171219S0117_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170908S0189_mosaic.fits",  # B1200:0.550 HAM
    # "process_arcs/GMOS/S20131230S0153_mosaic.fits",  # R150:0.550 EEV - todo: wavelength solution not stable
    # "process_arcs/GMOS/S20130801S0140_mosaic.fits",  # R150:0.700 EEV - todo: RMS > 0.5 (RMS = 0.69)
    # "process_arcs/GMOS/S20170430S0060_mosaic.fits",  # R150:0.717 HAM - todo: RMS > 0.5 (RMS = 0.78)
    # "process_arcs/GMOS/S20170430S0063_mosaic.fits",  # R150:0.727 HAM - todo: RMS > 0.5 (RMS = 1.26)
    "process_arcs/GMOS/S20171102S0051_mosaic.fits",  # R150:0.950 HAM
    "process_arcs/GMOS/S20130114S0100_mosaic.fits",  # R400:0.620 EEV
    "process_arcs/GMOS/S20130217S0073_mosaic.fits",  # R400:0.800 EEV
    # "process_arcs/GMOS/S20170108S0046_mosaic.fits",  # R400:0.550 HAM - todo: RMS > 0.5 (RMS = 0.60)
    "process_arcs/GMOS/S20170129S0125_mosaic.fits",  # R400:0.685 HAM
    "process_arcs/GMOS/S20170703S0199_mosaic.fits",  # R400:0.800 HAM
    "process_arcs/GMOS/S20170718S0420_mosaic.fits",  # R400:0.910 HAM
    # "process_arcs/GMOS/S20100306S0460_mosaic.fits",  # R600:0.675 EEV - todo: breaks p.determineWavelengthSolution
    # "process_arcs/GMOS/S20101218S0139_mosaic.fits",  # R600:0.675 EEV - todo: breaks p.determineWavelengthSolution
    "process_arcs/GMOS/S20110306S0294_mosaic.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/S20110720S0236_mosaic.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/S20101221S0090_mosaic.fits",  # R600:0.690 EEV
    "process_arcs/GMOS/S20120322S0122_mosaic.fits",  # R600:0.900 EEV
    "process_arcs/GMOS/S20130803S0011_mosaic.fits",  # R831:0.576 EEV
    "process_arcs/GMOS/S20130414S0040_mosaic.fits",  # R831:0.845 EEV
    "process_arcs/GMOS/S20170214S0059_mosaic.fits",  # R831:0.440 HAM
    "process_arcs/GMOS/S20170703S0204_mosaic.fits",  # R831:0.600 HAM
    "process_arcs/GMOS/S20171018S0048_mosaic.fits",  # R831:0.865 HAM
]

reference_files = [
    "_".join(f.split("_")[:-1]) + "_wavelengthSolutionDetermined.fits"
    for f in input_files
]


@pytest.fixture(scope="module")
def ad(request, path_to_inputs, path_to_outputs):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `determineWavelengthSolution` on it and return the output object with a
    `.WAVECAL` table. This makes tests more efficient because the primitive is
    run only once, instead of x n_tests.

    If the input file does not exist, this fixture raises a IOError.

    If the input file does not exist and PyTest is called with the
    `--force-preprocess-data`, this fixture looks for cached raw data and
    process it. If the raw data does not exist, it is then cached via download
    from the Gemini Archive.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    path_to_inputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached input files.
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        output folder.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.

    Raises
    ------
    IOError
        If the input file does not exist and if --force-preprocess-data is False.
    """
    force_preprocess = request.config.getoption("--force-preprocess-data")

    fname = os.path.join(path_to_inputs, request.param)

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

    print('\n\n Running test inside folder:\n  {}'.format(path_to_outputs))

    if os.path.exists(fname):
        print("\n Loading existing input file:\n  {:s}\n".format(fname))
        _ad = astrodata.open(fname)

    elif force_preprocess:

        print("\n\n Pre-processing input file:\n  {:s}\n".format(fname))
        subpath, basename = os.path.split(request.param)
        basename, extension = os.path.splitext(basename)
        basename = basename.split('_')[0] + extension

        raw_fname = testing.download_from_archive(basename, path=subpath)

        _ad = astrodata.open(raw_fname)
        _ad = preprocess_data(_ad, os.path.join(path_to_inputs, subpath))

    else:
        raise IOError("Cannot find input file:\n {:s}".format(fname))

    ad_out = p.determineWavelengthSolution([_ad])[0]

    tests_failed_before_module = request.session.testsfailed

    yield ad_out

    _dir = os.path.join(path_to_outputs, os.path.dirname(request.param))
    os.makedirs(_dir, exist_ok=True)

    do_plots(ad_out, _dir)

    if request.session.testsfailed > tests_failed_before_module:
        fname_out = os.path.join(_dir, ad_out.filename)
        ad_out.write(filename=fname_out, overwrite=True)
        print('\n Saved file to:\n  {}\n'.format(fname_out))

    del ad_out


@pytest.fixture(scope="module")
def ad_ref(request, path_to_refs):
    """
    Loads existing reference FITS files as AstroData objects.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    path_to_refs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached reference files.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.

    Raises
    ------
    IOError
        If the reference file does not exist. It should be created and verified
        manually.
    """
    fname = os.path.join(path_to_refs, request.param)

    if not os.path.exists(fname):
        IOError(" Cannot find reference file:\n {:s}".format(fname))

    return astrodata.open(fname)


def do_plots(ad, output_dir):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : astrodata
    output_dir : str
    """
    try:
        p = PlotGmosSpectLongslitArcs(ad, output_dir)
        p.wavelength_calibration_plots()
        p.close_all()

    except ImportError:
        warn("Could not generate plots")


def preprocess_data(ad, path):
    """
    Recipe used to generate input data for Wavelength Calibration tests. It is
    called only if the input data do not exist and if `--force-preprocess-data`
    is used in the command line.

    Parameters
    ----------
    ad : AstroData
        Input raw arc data loaded as AstroData.
    path : str
        Path that points to where the input data is cached.

    Returns
    -------
    AstroData
        Pre-processed arc data.
    """
    _p = primitives_gmos_spect.GMOSSpect([ad])

    _p.prepare()
    _p.addDQ(static_bpm=None)
    _p.addVAR(read_noise=True)
    _p.overscanCorrect()
    _p.ADUToElectrons()
    _p.addVAR(poisson_noise=True)
    _p.mosaicDetectors()
    ad = _p.makeIRAFCompatible()[0]

    _p.writeOutputs(outfilename=os.path.join(path, ad.filename))

    return ad


@pytest.fixture(scope="session", autouse=True)
def setup_log(path_to_outputs):
    """
    Fixture that setups DRAGONS' logging system to avoid duplicated outputs.

    Parameters
    ----------
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        output folder.
    """
    log_file = "{}.log".format(os.path.splitext(os.path.basename(__file__))[0])
    log_file = os.path.join(path_to_outputs, log_file)

    logutils.config(mode="standard", file_name=log_file)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", input_files, indirect=True)
def test_reduced_arcs_contain_wavelength_solution_model_with_expected_rms(ad):
    """
    Make sure that the WAVECAL model was fitted with an RMS smaller than 0.5.
    """
    table = ad[0].WAVECAL
    coefficients = table["coefficients"]
    rms = coefficients[table["name"] == "rms"]

    np.testing.assert_array_less(rms, 0.5)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(input_files, reference_files), indirect=True)
def test_reduced_arcs_contains_stable_wavelength_solution(ad, ad_ref):
    """
    Make sure that the wavelength solution gives same results on different
    runs.
    """
    table = ad[0].WAVECAL
    table_ref = ad_ref[0].WAVECAL

    model = astromodels.dict_to_chebyshev(
        dict(zip(table["name"], table["coefficients"])))

    ref_model = astromodels.dict_to_chebyshev(
        dict(zip(table_ref["name"], table_ref["coefficients"])))

    x = np.arange(ad[0].shape[1])
    y = model(x)
    ref_y = ref_model(x)

    np.testing.assert_allclose(y, ref_y, rtol=1)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(input_files, reference_files), indirect=True)
def test_reduced_arcs_are_similar(ad, ad_ref):
    """
    Compares the output and the reference arcs to make sure their data are
    similar before running further tests.
    """
    for ext_out, ext_ref in zip(ad, ad_ref):
        np.testing.assert_allclose(ext_out.data, ext_ref.data, rtol=1)
