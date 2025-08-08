import pytest

import numpy as np
from astropy.io import fits

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
    ad = astrodata.create(fits.PrimaryHDU())
    for i in range(napertures * norders):
        ad.append(np.empty((10,10)))
        ad[-1].hdr['APERTURE'] = (i // norders if contiguous_apertures else
                                  i % napertures) + 1
        ad[-1].hdr['SPECORDR'] = (i % norders if contiguous_apertures else
                                  i // napertures) + 1
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
