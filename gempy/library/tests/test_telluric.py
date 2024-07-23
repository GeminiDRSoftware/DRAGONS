import pytest

from gempy.library.telluric import TelluricSpectrum
from geminidr.core.primitives_telluric import GaussianLineSpreadFunction

import numpy as np

from astropy.modeling import fitting, models
from astropy import units as u
from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

import astrodata, gemini_instruments


@pytest.fixture(scope="module")
def tspek():
    data = np.ones((10001,), dtype=np.float32)
    input_frame = astrodata.wcs.pixel_frame(1)
    output_frame = cf.SpectralFrame(unit=u.nm, axes_order=(0,),
                                    axes_names=("WAVE",))
    wcs = gWCS([(input_frame, models.Scale(0.01) | models.Shift(400)),
                (output_frame, None)])
    ndd = astrodata.NDAstroData(data=data, wcs=wcs)
    # We can pass the NDAstroData instead of an "ext"
    tspek = TelluricSpectrum(ndd, name="test",
                             line_spread_function=GaussianLineSpreadFunction(ndd))
    return tspek


def test_telluric_spectrum_deepcopies_data():
    """Confirm that we can modify the TelluricSpectrum's nddata without
    affecting the original NDAstroData"""
    data = np.ones((1001,), dtype=np.float32)
    input_frame = astrodata.wcs.pixel_frame(1)
    output_frame = cf.SpectralFrame(unit=u.nm, axes_order=(0,),
                                    axes_names=("WAVE",))
    wcs = gWCS([(input_frame, models.Scale(0.1) | models.Shift(500)),
                (output_frame, None)])
    ndd = astrodata.NDAstroData(data=data, variance=data, wcs=wcs)
    tspek = TelluricSpectrum(ndd, name="test")
    # Simple modifications
    tspek.nddata.data *= 3
    tspek.nddata.variance += 1
    assert np.allclose(tspek.data, 3)
    assert np.allclose(ndd.data, 1)
    assert np.allclose(tspek.variance, 2)
    assert np.allclose(ndd.variance, 1)


def test_telluric_spectrum_attributes(tspek):
    assert np.allclose(tspek.dwaves, 0.01)
    assert np.allclose(tspek.waves, np.arange(400, 500.01, 0.01))
    assert tspek.absolute_dispersion == pytest.approx(0.01)
    assert tspek.name == "test"
    assert tspek.in_vacuo


@pytest.mark.parametrize("resolution", [4000, 5000, 10000, 15000])
def test_set_pca_without_lsf_params(tspek, resolution):
    """Confirm that the convolved FWHM of a fairly isolated absorption
    line at 458.8nm is close to the correct resolution"""
    tspek.lsf.resolution = resolution
    tspek.set_pca()
    m_init = models.Const1D(1) + models.Gaussian1D(
        amplitude=-0.05, mean=458.8, stddev=0.1)
    indices = np.logical_and(tspek.waves>=458.7, tspek.waves<458.9)
    fit_it = fitting.TRFLSQFitter()
    m_final = fit_it(m_init, tspek.waves[indices],
                     tspek.pca.components[0][indices])
    fitted_resolution = 0.424661 * m_final.mean_1 / m_final.stddev_1
    assert fitted_resolution == pytest.approx(resolution, rel=0.1)


@pytest.mark.parametrize("resolution", [5000, 7000, 10000, 15000])
def test_set_pca_with_lsf_params(tspek, resolution):
    """Confirm that the convolved FWHM of a fairly isolated absorption
    line at 458.8nm is close to the correct resolution"""
    tspek.set_pca(lsf_params={"resolution": np.logspace(3.5,4.5,5)})
    tspek.pca.set_interpolation_parameters([resolution])
    m_init = models.Const1D(1) + models.Gaussian1D(
        amplitude=-0.05, mean=458.8, stddev=0.1)
    indices = np.logical_and(tspek.waves>=458.7, tspek.waves<458.9)
    fit_it = fitting.TRFLSQFitter()
    m_final = fit_it(m_init, tspek.waves[indices],
                     tspek.pca.components[0][indices])
    fitted_resolution = 0.424661 * m_final.mean_1 / m_final.stddev_1
    assert fitted_resolution == pytest.approx(resolution, rel=0.1)
