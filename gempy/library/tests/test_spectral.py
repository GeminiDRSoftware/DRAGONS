#!/usr/bin/env python
import numpy as np
import pytest

from astropy import units as u
from astropy.io import fits

from gempy.library import spectral


def test_create_spek1d_from_astrodata(fake_astrodata):
    spectrum = spectral.Spek1D(fake_astrodata[0])
    assert spectrum.spectral_axis[0] == 350 * u.nm
    np.testing.assert_allclose(np.diff(spectrum.spectral_axis), 0.1 * u.nm)


def test_get_region_from_fake_spectrum(fake_astrodata):
    np.testing.assert_allclose(fake_astrodata[0].data, 1)

    spectrum = spectral.Spek1D(fake_astrodata[0])
    region = spectral.SpectralRegion(400. * u.nm, 401. * u.nm)
    data, mask, variance = spectrum.signal(region)

    assert spectrum.unit == u.electron
    assert isinstance(data, u.Quantity)
    assert isinstance(mask, np.uint16)
    assert isinstance(variance, u.Quantity)
    assert data == abs((400. - 401.) / fake_astrodata[0].hdr['CD1_1'] * u.electron)


def test_addition(fake_spectra):
    spek1, spek2 = fake_spectra
    result = spek2 + spek1
    np.testing.assert_allclose(result.flux, 5 * u.electron)
    np.testing.assert_allclose(result.variance, 2)

    result = spek1 + 2. * u.electron
    np.testing.assert_allclose(result.flux, 4 * u.electron)

    with pytest.raises(u.core.UnitConversionError):
        result = spek1 + 2

    spek2.add(2 * u.electron)
    np.testing.assert_allclose(spek2.flux, 5 * u.electron)


def test_subtraction(fake_spectra):
    spek1, spek2 = fake_spectra
    result = spek2 - spek1
    np.testing.assert_allclose(result.flux, 1 * u.electron)
    np.testing.assert_allclose(result.variance, 2)

    result = spek1 - 3. * u.electron
    np.testing.assert_allclose(result.flux, -1 * u.electron)

    with pytest.raises(u.core.UnitConversionError):
        result = spek1 - 2

    spek2.subtract(2 * u.electron)
    np.testing.assert_allclose(spek2.flux, 1 * u.electron)


def test_multiplication(fake_spectra):
    spek1, spek2 = fake_spectra
    result = spek1 * spek2
    np.testing.assert_allclose(result.flux, 6 * u.electron**2)
    np.testing.assert_allclose(result.variance, 13)

    result = spek2 * 2.5
    np.testing.assert_allclose(result.flux, 7.5 * u.electron)

    result = spek1 * (-2 * u.m)
    np.testing.assert_allclose(result.flux, -4 * u.m * u.electron)

    spek1.multiply(-2 * u.m)
    np.testing.assert_allclose(spek1.flux, -4 * u.m * u.electron)


def test_division(fake_spectra):
    spek1, spek2 = fake_spectra
    result = spek2 / spek1
    np.testing.assert_allclose(result.flux, 1.5)
    np.testing.assert_allclose(result.variance, 13/16)

    result = spek1 / (10 * u.s)
    np.testing.assert_allclose(result.flux, 0.2 * u.electron / u.s)

    spek1.divide(10 * u.s)
    np.testing.assert_allclose(spek1.flux, 0.2 * u.electron / u.s)


def test_as_specrtum1d(fake_spectra):
    spek1, spek2 = fake_spectra
    spectrum1d = spek1.asSpectrum1D()
    np.testing.assert_allclose(spectrum1d.flux, 2 * u.electron)
    np.testing.assert_allclose(spectrum1d.uncertainty.array, 1)
    np.testing.assert_allclose(spectrum1d.spectral_axis, np.arange(350, 450, 0.1) * u.nm)


@pytest.fixture(scope="module")
def fake_astrodata():
    astrofaker = pytest.importorskip('astrofaker')

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = np.zeros((1000,))

    _ad = astrofaker.create('GMOS-S', extra_keywords={'GRATING': 'R831'})
    _ad.add_extension(hdu, pixel_scale=1.0)
    _ad[0].wcs = None  # or else imaging WCS will be added

    _ad[0].data = _ad[0].data.ravel() + 1.
    _ad[0].mask = np.zeros(_ad[0].data.size, dtype=np.uint16)  # ToDo Requires mask
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

    return _ad


@pytest.fixture(scope="function")
def fake_spectra(fake_astrodata):
    spek1 = spectral.Spek1D(fake_astrodata[0], copy=True)
    spek2 = spectral.Spek1D(fake_astrodata[0], copy=True)
    spek1._data = np.full_like(spek1.data, 2.)
    spek2._data = np.full_like(spek2.data, 3.)
    return spek1, spek2

if __name__ == '__main__':
    pytest.main()
