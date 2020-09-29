#!/usr/bin/env python
import numpy as np
import pytest

from astropy import units
from astropy.io import fits
from gempy.library import spectral


def test_get_region_from_fake_spectrum():
    ad = _create_fake_data()
    np.testing.assert_allclose(ad[0].data, 1)

    spectrum = spectral.Spek1D(ad[0])
    region = spectral.SpectralRegion(400. * units.nm, 401. * units.nm)
    data, mask, variance = spectrum.signal(region)

    assert spectrum.unit == units.electron
    assert isinstance(data, units.Quantity)
    assert isinstance(mask, np.uint16)
    assert isinstance(variance, units.Quantity)
    assert data == abs((400. - 401.) / ad[0].hdr['CD1_1'] * units.electron)


def _create_fake_data():
    astrofaker = pytest.importorskip('astrofaker')

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = np.zeros((1000,))

    _ad = astrofaker.create('GMOS-S', extra_keywords={'GRATING': 'R831'})
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
