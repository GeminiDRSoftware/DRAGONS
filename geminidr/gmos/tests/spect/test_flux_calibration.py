#!/usr/bin/env python
"""
Tests for the `calculateSensitivity` primitive using GMOS-S and GMOS-N data.
"""
import numpy as np
from copy import deepcopy
import itertools
import os
import pytest

from scipy.interpolate import BSpline
from gwcs import coordinate_frames as cf, WCS as gWCS

import astrodata, gemini_instruments
from astrodata.testing import assert_most_close

from astropy.io import fits
from astropy import units as u
from astropy.modeling import models

from gempy.library import astromodels
from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory


test_datasets = [
    "N20180109S0287_sensitivityCalculated.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
]


# --- Tests -------------------------------------------------------------------
@pytest.mark.gmosls
def test_flux_calibration_with_fake_data():
    """
    Test the :func:`~geminidr.gmos.GMOSSpect.calculateSensitivity` primitive
    by creating a fake spectrum using spectrophotometric data from the look-up
    tables in :mod:`geminidr.gemini.lookup`, then calculating the sensitivity
    using this data and applying the primitive. The input fake data and the
    flux calibrated data should be the same.
    """

    def _get_spectrophotometric_file_path(specphot_name):

        from geminidr.gemini.lookups import spectrophotometric_standards

        path = list(spectrophotometric_standards.__path__).pop()
        file_path = os.path.join(
            path, specphot_name.lower().replace(' ', '') + ".dat")

        return file_path

    def _get_spectrophotometric_data(object_name):

        file_path = _get_spectrophotometric_file_path(object_name)
        table = primitives_gmos_spect.Spect([])._get_spectrophotometry(file_path)

        std_wavelength = table['WAVELENGTH_AIR'].data
        std_flux = table['FLUX'].data

        if std_wavelength[0] // 1000 > 1:  # Converts from \AA to nm
            std_wavelength = std_wavelength / 10

        std_flux = std_flux[(350 <= std_wavelength) * (std_wavelength <= 850)]
        std_wavelength = std_wavelength[(350 <= std_wavelength) & (std_wavelength <= 850)]

        spline = BSpline(std_wavelength, std_flux, 3)
        wavelength = np.linspace(std_wavelength.min(), std_wavelength.max(), 1000)
        flux = spline(wavelength).astype(np.float32)
        return wavelength, flux

    def _create_fake_data(object_name):
        astrofaker = pytest.importorskip('astrofaker')

        wavelength, flux = _get_spectrophotometric_data(object_name)

        wavecal = {
            'degree': 1,
            'domain': [0., wavelength.size - 1],
            'c0': wavelength.mean(),
            'c1': wavelength.mean() / 2,
        }
        wave_model = models.Chebyshev1D(**wavecal)
        wave_model.inverse = astromodels.make_inverse_chebyshev1d(wave_model, rms=0.01,
                                                                  max_deviation=0.03)

        hdu = fits.ImageHDU()
        hdu.header['CCDSUM'] = "1 1"
        hdu.data = flux

        _ad = astrofaker.create('GMOS-S', 'LS')
        _ad.add_extension(hdu, pixel_scale=1.0)

        _ad[0].mask = np.zeros(_ad[0].data.size, dtype=np.uint16)  # ToDo Requires mask
        _ad[0].variance = np.ones_like(_ad[0].data)  # ToDo Requires Variance
        in_frame = cf.CoordinateFrame(naxes=1, axes_type=['SPATIAL'],
                                      axes_order=(0,), unit=u.pix,
                                      axes_names=('x',), name='pixels')
        out_frame = cf.SpectralFrame(unit=u.nm, name='world', axes_names=('AWAV',))
        _ad[0].wcs = gWCS([(in_frame, wave_model),
                           (out_frame, None)])

        _ad[0].hdr.set('NAXIS', 1)
        _ad[0].phu.set('OBJECT', object_name)
        _ad[0].phu.set('EXPTIME', 1.)
        _ad[0].hdr.set('BUNIT', "electron")

        assert _ad.object() == object_name
        assert _ad.exposure_time() == 1

        return _ad

    ad = _create_fake_data("Feige 34")
    p = primitives_gmos_spect.GMOSSpect([ad])
    std_ad = p.calculateSensitivity()[0]
    flux_corrected_ad = p.fluxCalibrate(standard=std_ad)[0]

    for ext, flux_corrected_ext in zip(ad, flux_corrected_ad):
        np.testing.assert_allclose(flux_corrected_ext.data, ext.data, atol=1e-4)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", test_datasets, indirect=True)
def test_regression_on_flux_calibration(ad, ref_ad_factory, change_working_dir):
    """
    Regression test for the :func:`~geminidr.gmos.GMOSSpect.fluxCalibrate`
    primitive.

    Parameters
    ----------
    ad : pytest.fixture (AstroData)
        Fixture that reads the filename and loads as an AstroData object.
    change_working_dir : pytest.fixture
        Fixture that changes the working directory
        (see :mod:`astrodata.testing`).
    reference_ad : pytest.fixture
        Fixture that contains a function used to load the reference AstroData
        object (see :mod:`recipe_system.testing`).
    """
    with change_working_dir():
        logutils.config(file_name='log_regression_{:s}.txt'.format(ad.data_label()))
        p = primitives_gmos_spect.GMOSSpect([ad])
        p.fluxCalibrate(standard=ad)
        flux_calibrated_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(flux_calibrated_ad.filename)

    for flux_cal_ext, ref_ext in zip(flux_calibrated_ad, ref_ad):
        np.testing.assert_allclose(
            flux_cal_ext.data, ref_ext.data, atol=1e-4)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", test_datasets, indirect=True)
def test_flux_calibration_with_resampling(ad, path_to_inputs):
    """
    Test the :func:`~geminidr.gmos.GMOSSpect.fluxCalibrate` primitive with
    combinations of linear and log-linear resampling for both the standard
    and the science data. This is a combination test for both the
    resampleToCommonFrame and fluxCalibrate primitives.

    Parameters
    ----------
    ad : pytest.fixture (AstroData)
        Fixture that reads the filename and loads as an AstroData object.
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    """
    output_fluxes = []
    wall = ad[0].wcs(np.arange(ad[0].data.size)[ad[0].mask==0])
    calcsens_params = {"filename": os.path.join(path_to_inputs, "hz44_stis_006.fits")}

    for std_sampling, sci_sampling in itertools.product(["linear", "loglinear"],
                                                        repeat=2):
        p = primitives_gmos_longslit.GMOSLongslit([deepcopy(ad)])
        p.resampleToCommonFrame(output_wave_scale=std_sampling)
        adstd = p.calculateSensitivity(**calcsens_params)[0]
        p = primitives_gmos_longslit.GMOSLongslit([deepcopy(ad)])
        p.resampleToCommonFrame(output_wave_scale=sci_sampling)
        adout = p.fluxCalibrate(standard=adstd).pop()
        wout = adout[0].wcs(np.arange(adout[0].data.size))
        fout = np.interp(wall, wout[adout[0].mask == 0],
                         adout[0].data[adout[0].mask == 0])
        output_fluxes.append(fout)

    for fout1, fout2 in list(itertools.combinations(output_fluxes, 2)):
        assert_most_close(fout1, fout2, max_miss=wall.size // 100, rtol=0.01)



# --- Helper functions and fixtures -------------------------------------------
@pytest.fixture
def ad(request, path_to_inputs):
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
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    import os
    from astrodata.testing import download_from_archive
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS
    from recipe_system.utils.reduce_utils import normalize_ucals
    from recipe_system.reduction.coreReduce import Reduce

    associated_calibrations = {
        "N20180109S0287.fits": {
            "arcs": ["N20180109S0315.fits"],
            "bias": ["N20180103S0563.fits",
                     "N20180103S0564.fits",
                     "N20180103S0565.fits",
                     "N20180103S0566.fits",
                     "N20180103S0567.fits",],
            "flat": ["N20180109S0288.fits"],
        }
    }

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        bias_path = [download_from_archive(f) for f in cals['bias']]
        flat_path = [download_from_archive(f) for f in cals['flat']]
        arc_path = [download_from_archive(f) for f in cals['arcs']]

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing BIAS for {:s}'.format(data_label))
        logutils.config(file_name='log_bias_{}.txt'.format(data_label))
        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_path)
        bias_reduce.runr()
        bias_master = bias_reduce.output_filenames.pop()
        calibration_files = ['processed_bias:{}'.format(bias_master)]
        del bias_reduce

        print('Reducing FLAT for {:s}'.format(data_label))
        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_path)
        flat_reduce.ucals = normalize_ucals(calibration_files)
        flat_reduce.runr()
        flat_master = flat_reduce.output_filenames.pop()
        calibration_files.append('processed_flat:{}'.format(flat_master))
        del flat_reduce

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend(arc_path)
        arc_reduce.ucals = normalize_ucals(calibration_files)
        arc_reduce.runr()
        arc_master = arc_reduce.output_filenames.pop()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect(bias=bias_master)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.attachWavelengthSolution(arc=arc_master)
        p.flatCorrect(flat=flat_master)
        p.QECorrect()
        p.distortionCorrect()
        p.findApertures(max_apertures=1)
        p.skyCorrectFromSlit()
        p.traceApertures()
        p.extractSpectra()
        p.calculateSensitivity()

        os.chdir("inputs/")
        _ = p.writeOutputs().pop()
        os.chdir("../../")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
