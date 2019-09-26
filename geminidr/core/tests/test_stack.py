# pytest suite
"""
Tests for primitives_stack.

This is a suite of tests to be run with pytest.

"""
import astrodata, gemini_instruments

from astropy.io import fits
from astropy.table import Table
import numpy as np

from geminidr.niri.primitives_niri_image import NIRIImage

def test_stackframes_refcat_propagation():
    phu = fits.PrimaryHDU()
    phu.header.update(OBSERVAT='Gemini-North', INSTRUME='NIRI',
                      ORIGNAME='N20010101S0001.fits')
    data = np.zeros((2,2))

    ad1 = astrodata.create(phu)
    ad1.append(data)
    ad2 = astrodata.create(phu)
    ad2.append(data)

    p = NIRIImage([ad1, ad2])
    p.prepare()

    refcat = Table([[1,2], ['a','b']], names=('Id', 'Cat_Id'))
    ad1.REFCAT = refcat
    refcat['Cat_Id'] = ['b','c']
    ad2.REFCAT = refcat

    adout = p.stackFrames()[0]

    # The merged REFCAT should contain 'a', 'b', 'c'
    assert len(adout.REFCAT) == 3
    np.testing.assert_equal(adout.REFCAT['Id'], np.arange(1,4))
    assert all(adout.REFCAT['Cat_Id'] == ['a','b','c'])