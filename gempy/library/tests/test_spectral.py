#!/usr/bin/env python
import numpy as np
import pytest

import astrodata
import gemini_instruments

from astropy import units
from astropy.io import fits
from gempy.library import spectral


def test_create_object_from_fake_ad():
    ad = _create_fake_data()
    s = spectral.Spek1D(ad[0])


def test_spectral_region_creation_with_two_arguments():
    _ = spectral.SpectralRegion(10 * units.pixel, 11 * units.pixel)
    _ = spectral.SpectralRegion(0.549e-6 * units.m, 0.551e-6 * units.m)
    _ = spectral.SpectralRegion(0.549 * units.um, 0.551 * units.um)
    _ = spectral.SpectralRegion(549. * units.nm, 551. * units.nm)
    _ = spectral.SpectralRegion(5490. * units.AA, 5510. * units.AA)


def test_spectral_region_creation_with_single_tuple():
    _ = spectral.SpectralRegion((5490. * units.AA, 5510. * units.AA))


def test_spectral_region_creation_with_single_list():
    _ = spectral.SpectralRegion([5490. * units.AA, 5510. * units.AA])


def test_spectral_region_creation_fails_when_input_is_not_quantity():
    with pytest.raises(TypeError) as err:
        _ = spectral.SpectralRegion(549., 551.)
    assert "Expected astropy.units.Quantity instances as inputs" in str(err.value)


def test_get_region_from_fake_spectrum():
    ad = _create_fake_data()
    np.testing.assert_allclose(ad[0].data, 1)

    spectrum = spectral.Spek1D(ad[0])
    region = spectral.SpectralRegion(549. * units.nm, 551. * units.nm)
    data, mask, variance = spectrum.signal(region)

    assert data == (551 - 549) / ad[0].hdr['CD1_1']


def _create_fake_data():
    astrofaker = pytest.importorskip('astrofaker')

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = np.zeros((1000, 1))

    _ad = astrofaker.create('GMOS-S')
    _ad.add_extension(hdu, pixel_scale=1.0)

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


if __name__ == '__main__':
    pytest.main()
