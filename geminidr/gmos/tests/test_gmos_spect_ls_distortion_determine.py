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


input_files = [
    # Process Arcs: GMOS-N ---
    # (Input File, fwidth, order, min_snr)
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
    "process_arcs/GMOS/N20100830S0594_mosaic.fits",  # R150:0.500 EEV
    "process_arcs/GMOS/N20100702S0321_mosaic.fits",  # R150:0.700 EEV
    "process_arcs/GMOS/N20130606S0291_mosaic.fits",  # R150:0.550 E2V
    "process_arcs/GMOS/N20130112S0574_mosaic.fits",  # R150:0.700 E2V
    "process_arcs/GMOS/N20130809S0337_mosaic.fits",  # R150:0.700 E2V
    "process_arcs/GMOS/N20140408S0218_mosaic.fits",  # R150:0.700 E2V
    "process_arcs/GMOS/N20180119S0232_mosaic.fits",  # R150:0.520 HAM
    "process_arcs/GMOS/N20180516S0214_mosaic.fits",  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2
    "process_arcs/GMOS/N20171007S0439_mosaic.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20171007S0441_mosaic.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20101212S0213_mosaic.fits",  # R400:0.550 EEV
    "process_arcs/GMOS/N20100202S0214_mosaic.fits",  # R400:0.700 EEV
    "process_arcs/GMOS/N20130106S0194_mosaic.fits",  # R400:0.500 E2V
    "process_arcs/GMOS/N20130422S0217_mosaic.fits",  # R400:0.700 E2V
    "process_arcs/GMOS/N20170108S0210_mosaic.fits",  # R400:0.660 HAM
    "process_arcs/GMOS/N20171113S0135_mosaic.fits",  # R400:0.750 HAM
    "process_arcs/GMOS/N20100427S1276_mosaic.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/N20180120S0417_mosaic.fits",  # R600:0.860 HAM
    "process_arcs/GMOS/N20100212S0143_mosaic.fits",  # R831:0.450 EEV
    "process_arcs/GMOS/N20100720S0247_mosaic.fits",  # R831:0.850 EEV
    "process_arcs/GMOS/N20130808S0490_mosaic.fits",  # R831:0.571 E2V
    "process_arcs/GMOS/N20130830S0291_mosaic.fits",  # R831:0.845 E2V
    "process_arcs/GMOS/N20170910S0009_mosaic.fits",  # R831:0.653 HAM
    "process_arcs/GMOS/N20170509S0682_mosaic.fits",  # R831:0.750 HAM
    "process_arcs/GMOS/N20181114S0512_mosaic.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20170416S0058_mosaic.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20170416S0081_mosaic.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20180120S0315_mosaic.fits",  # R831:0.865 HAM
    # Process Arcs: GMOS-S ---
    # "process_arcs/GMOS/S20130218S0126_mosaic.fits",  # B600:0.500 EEV - todo: won't pass
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
    "process_arcs/GMOS/S20170116S0189_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170103S0149_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170730S0155_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20171219S0117_mosaic.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170908S0189_mosaic.fits",  # B1200:0.550 HAM
    "process_arcs/GMOS/S20131230S0153_mosaic.fits",  # R150:0.550 EEV
    "process_arcs/GMOS/S20130801S0140_mosaic.fits",  # R150:0.700 EEV
    "process_arcs/GMOS/S20170430S0060_mosaic.fits",  # R150:0.717 HAM
    # "process_arcs/GMOS/S20170430S0063_mosaic.fits",  # R150:0.727 HAM - todo: won't pass
    "process_arcs/GMOS/S20171102S0051_mosaic.fits",  # R150:0.950 HAM
    "process_arcs/GMOS/S20130114S0100_mosaic.fits",  # R400:0.620 EEV
    "process_arcs/GMOS/S20130217S0073_mosaic.fits",  # R400:0.800 EEV
    "process_arcs/GMOS/S20170108S0046_mosaic.fits",  # R400:0.550 HAM
    "process_arcs/GMOS/S20170129S0125_mosaic.fits",  # R400:0.685 HAM
    "process_arcs/GMOS/S20170703S0199_mosaic.fits",  # R400:0.800 HAM
    "process_arcs/GMOS/S20170718S0420_mosaic.fits",  # R400:0.910 HAM
    # "process_arcs/GMOS/S20100306S0460_mosaic.fits",  # R600:0.675 EEV - todo: won't pass
    # "process_arcs/GMOS/S20101218S0139_mosaic.fits",  # R600:0.675 EEV - todo: won't pass
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
    "_".join(f.split("_")[:-1]) + "_distortionDetermined.fits"
    for f in input_files
]


@pytest.fixture(scope="module")
def ad(request, path_to_inputs, path_to_outputs, path_to_refs):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `determineDistortion` primitive on it, and return the output object with a
    `.FITCOORD` table. This makes tests more efficient because the primitive is
    run only once, instead of N x Numbes of tests.

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

    ad_out = p.determineDistortion(
        [_ad],
        spatial_order=3,
        spectral_order=4,
        id_only=False,
        min_snr=5.,
        fwidth=None,
        nsum=10,
        max_shift=0.05,
        max_missed=5)[0]

    tests_failed_before_module = request.session.testsfailed

    yield ad_out

    _dir = os.path.join(path_to_outputs, os.path.dirname(request.param))
    _ref_dir = os.path.join(path_to_refs, os.path.dirname(request.param))

    os.makedirs(_dir, exist_ok=True)

    if request.config.getoption("--do-plots"):
        do_plots(ad_out, _dir, _ref_dir)

    if request.session.testsfailed > tests_failed_before_module:
        fname_out = os.path.join(_dir, ad_out.filename)
        ad_out.write(filename=fname_out, overwrite=True)
        print('\n Saved file to:\n  {}\n'.format(fname_out))

    del ad_out


def do_plots(ad, output_path, reference_path):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : AstroData
    output_path : str or Path
    reference_path : str or Path
    """
    try:
        from .plots_gmos_spect_longslit_arcs import PlotGmosSpectLongslitArcs
    except ImportError:
        warn("Could not generate plots")
        return

    ad_ref = astrodata.open(os.path.join(reference_path, ad.filename))

    p = PlotGmosSpectLongslitArcs(ad, output_folder=output_path, ref_folder=reference_path)
    p.show_distortion_map(ad)
    p.show_distortion_model_difference(ad, ad_ref)
    p.close_all()


def preprocess_data(ad, path):
    """
    Recipe used to generate input data for Determine Distorion tests. It is
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
@pytest.mark.parametrize("ad, ad_ref", zip(input_files, reference_files), indirect=True)
def test_determine_distortion_comparing_models_coefficients(ad, ad_ref):
    """
    Runs the `determineDistorion` primitive on a preprocessed data and compare
    its model with the one in the reference file.
    """
    assert ad.filename == ad_ref.filename

    c = np.ma.masked_invalid(ad[0].FITCOORD["coefficients"])
    c_ref = np.ma.masked_invalid(ad_ref[0].FITCOORD["coefficients"])

    np.testing.assert_allclose(c, c_ref, atol=2)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(input_files, reference_files), indirect=True)
def test_determine_distortion_comparing_modeled_arrays(ad, ad_ref):
    """
    Runs the `determineDistorion` primitive on a preprocessed data and compare
    its model with the one in the reference file. The distortion model needs to
    be reconstructed because different coefficients might return same results.
    """
    assert ad.filename == ad_ref.filename

    table = ad[0].FITCOORD
    model_dict = dict(zip(table['name'], table['coefficients']))
    model = astromodels.dict_to_chebyshev(model_dict)

    ref_table = ad_ref[0].FITCOORD
    ref_model_dict = dict(zip(ref_table['name'], ref_table['coefficients']))
    ref_model = astromodels.dict_to_chebyshev(ref_model_dict)

    X, Y = np.mgrid[:ad[0].shape[0], :ad[0].shape[1]]

    np.testing.assert_allclose(model(X, Y), ref_model(X, Y), atol=1)
