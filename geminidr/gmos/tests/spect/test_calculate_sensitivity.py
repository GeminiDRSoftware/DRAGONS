#!/usr/bin/env python
"""
Tests for the `calculateSensitivity` primitive using GMOS-S and GMOS-N data.

Changelog
---------
2020-06-02:
    - Recreated input files using:
        - astropy 4.1rc1
        - gwcs 0.13.1.dev19+gc064a02 - This should be cloned and the
            `transform-1.1.0` string should be replaced by `transform-1.2.0`
            in the `gwcs/schemas/stsci.edu/gwcs/step-1.0.0.yaml` file.

"""
import numpy as np
import os
import pytest

from scipy.interpolate import BSpline

import astrodata
import gemini_instruments

from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory


datasets = [
    "N20180109S0287_extracted.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
]


# --- Tests -------------------------------------------------------------------
@pytest.mark.gmosls
def test_calculate_sensitivity_from_science_equals_one_and_table_equals_one(
        path_to_outputs):

    def _create_fake_data():
        astrofaker = pytest.importorskip('astrofaker')

        hdu = fits.ImageHDU()
        hdu.data = np.ones((1000,), dtype=np.float32)

        _ad = astrofaker.create('GMOS-S', mode='LS')
        _ad.add_extension(hdu, pixel_scale=1.0)

        _ad[0].mask = np.zeros_like(_ad[0].data, dtype=np.uint16)  # ToDo Requires mask
        _ad[0].variance = np.ones_like(_ad[0].data)  # ToDo Requires Variance

        _ad[0].phu.set('OBJECT', "DUMMY")
        _ad[0].phu.set('EXPTIME', 1.)
        _ad[0].hdr.set('BUNIT', "electron")
        _ad[0].hdr.set('CTYPE1', "Wavelength")
        _ad[0].hdr.set('CUNIT1', "nm")
        _ad[0].hdr.set('CRPIX1', 1)
        _ad[0].hdr.set('CRVAL1', 350.)
        _ad[0].hdr.set('CDELT1', 0.1)
        _ad[0].hdr.set('CD1_1', 0.1)

        assert _ad.object() == 'DUMMY'
        assert _ad.exposure_time() == 1
        _ad.create_gwcs()

        return _ad

    def _create_fake_table():

        wavelengths = np.arange(200., 900., 5) * u.Unit("nm")
        flux = np.ones(wavelengths.size) * u.Unit("erg cm-2 s-1 AA-1")
        bandpass = np.ones(wavelengths.size) * u.Unit("nm")

        _table = QTable([wavelengths, flux, bandpass],
                        names=['WAVELENGTH', 'FLUX', 'FWHM'])

        _table.name = os.path.join(path_to_outputs, 'std_data.dat')
        _table.write(_table.name, format='ascii.ecsv')

        return _table.name

    def _get_wavelength_calibration(hdr):

        from astropy.modeling.models import Linear1D, Const1D

        _wcal_model = (
            Const1D(hdr.get('CRVAL1')) +
            Linear1D(slope=hdr.get('CD1_1'), intercept=hdr.get('CRPIX1')-1))

        assert _wcal_model(0) == hdr.get('CRVAL1')

        return _wcal_model

    table_name = _create_fake_table()
    ad = _create_fake_data()

    p = primitives_gmos_spect.GMOSSpect([ad])

    s_ad = p.calculateSensitivity(bandpass=5, filename=table_name, order=6).pop()
    assert hasattr(s_ad[0], 'SENSFUNC')

    for s_ext in s_ad:

        sens_table = s_ext.SENSFUNC
        sens_model = BSpline(
            sens_table['knots'].data, sens_table['coefficients'].data, 3)

        wcal_model = _get_wavelength_calibration(s_ext.hdr)

        wlengths = (wcal_model(np.arange(s_ext.data.size + 1)) * u.nm).to(
            sens_table['knots'].unit)

        sens_factor = sens_model(wlengths) * sens_table['coefficients'].unit

        np.testing.assert_allclose(sens_factor.physical.value, 1, atol=1e-4)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_regression_on_calculate_sensitivity(ad, change_working_dir, ref_ad_factory):

    with change_working_dir():
        logutils.config(file_name='log_regression_{:s}.txt'.format(ad.data_label()))
        p = primitives_gmos_spect.GMOSSpect([ad])
        p.calculateSensitivity(bandpass=5, order=6)
        calc_sens_ad = p.writeOutputs().pop()

    assert hasattr(calc_sens_ad[0], 'SENSFUNC')

    ref_ad = ref_ad_factory(calc_sens_ad.filename)

    for calc_sens_ext, ref_ext in zip(calc_sens_ad, ref_ad):
        np.testing.assert_allclose(
            calc_sens_ext.data, ref_ext.data, atol=1e-4)


# --- Helper functions and fixtures -------------------------------------------
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
    from gempy.utils import logutils
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals

    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS

    associated_calibrations = {
        "N20180109S0287.fits": {
            'bias': ["N20180103S0563.fits",
                     "N20180103S0564.fits",
                     "N20180103S0565.fits",
                     "N20180103S0566.fits",
                     "N20180103S0567.fits"],
            'flat': ["N20180109S0288.fits"],
            'arcs': ["N20180109S0315.fits"]
        }
    }

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))
    os.makedirs("inputs/", exist_ok=True)

    for filename, cals in associated_calibrations.items():

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        bias_path = [download_from_archive(f) for f in cals['bias']]
        flat_path = [download_from_archive(f) for f in cals['flat']]
        arc_path = [download_from_archive(f) for f in cals['arcs']]

        sci_ad = astrodata.from_file(sci_path)
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
        del arc_reduce

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None, user_bpm=None, add_illum_mask=False)
        p.addVAR(read_noise=True, poisson_noise=False)
        p.overscanCorrect(function="spline3", lsigma=3., hsigma=3.,
                          nbiascontam=0, niter=2, order=None)
        p.biasCorrect(bias=bias_master, do_cal='procmode')
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=False)
        p.attachWavelengthSolution(arc=arc_master)
        p.flatCorrect(flat=flat_master)
        p.QECorrect()
        p.distortionCorrect(order=3, subsample=1)
        p.findApertures(max_apertures=1, threshold=0.01, min_sky_region=20)
        p.skyCorrectFromSlit(order=5, grow=0)
        p.traceApertures(order=2, nsum=10, step=10, max_missed=5,
                         max_shift=0.05)
        p.extractSpectra(grow=10, method="standard", width=None)

        os.chdir("inputs/")
        processed_ad = p.writeOutputs().pop()
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_ad.filename))
        os.chdir("../")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
