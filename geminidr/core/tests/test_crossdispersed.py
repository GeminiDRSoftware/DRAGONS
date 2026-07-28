import pytest

from copy import deepcopy
from glob import glob

import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy import units as u

from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS

import astrodata
from geminidr.core.primitives_crossdispersed import CrossDispersed


@pytest.fixture(scope='module')
def create_ad(request, napertures=2, norders=6):
    """
    Create a test AstroData object with 12 extensions, representing
    2 apertures and 6 spectral orders.

    Parameters
    ----------
    napertures : int
        Number of apertures to simulate
    norders : int
        Number of spectral orders to simulate
    contiguous_apertures:  bool
        If True, extensions with the same aperture number will be contiguous
        interleaved (1, 2, 1, 2, ...)
    """
    contiguous_apertures = request.param
    ad = astrodata.create(fits.PrimaryHDU(
        header=fits.Header({'TELESCOP': 'Gemini-North',
                            'DATALAB': 'GN-2001-Q-001'})))
    ad.filename = "N20010101S0001.fits"
    ad.orig_filename = "N20010101S0001.fits"
    t = models.Scale(0.1) | models.Shift(1000)
    output_frame = cf.SpectralFrame(unit=(u.nm,), axes_names=('AWAV',))

    for i in range(napertures * norders):
        ad.append(np.ones((10,), dtype=np.float32))
        ad[-1].hdr['APERTURE'] = (i // norders if contiguous_apertures else
                                  i % napertures) + 1
        ad[-1].hdr['SPECORDR'] = (i % norders if contiguous_apertures else
                                  i // napertures) + 1
        ad[-1].hdr['BUNIT'] = "adu"
        ad[-1].wcs = gWCS([(astrodata.wcs.pixel_frame(1), t),
                           (deepcopy(output_frame), None)])
    return ad


@pytest.mark.parametrize('create_ad', (True, False), indirect=True)
def test_separate_by_spectral_order(create_ad):
    p = CrossDispersed([])
    adoutputs = p._separate_by_spectral_order(create_ad)

    # These checks should suffice for any combination of napertures and norders
    assert sum([len(adout) for adout in adoutputs]) == len(create_ad)

    for adout in adoutputs:
        assert len(set(adout.hdr['SPECORDR'])) == 1
        assert len(set(adout.hdr['APERTURE'])) == len(adout)


@pytest.mark.parametrize('create_ad', (True,), indirect=True)
def test_write_1d_spectra(create_ad, change_working_dir):
    p = CrossDispersed([create_ad])

    with change_working_dir():
        p.write1DSpectra(overwrite=True)
        output_files = sorted(glob("*.dat"))

        assert len(output_files) == len(create_ad)
        for f in output_files:
            data = np.loadtxt(f, skiprows=2).T
            assert data.shape == (2, 10)

