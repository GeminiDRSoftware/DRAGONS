"""
Tests for primitives_stack.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from numpy.testing import assert_array_equal

import astrodata
import gemini_instruments
from geminidr.niri.primitives_niri_image import NIRIImage


@pytest.fixture
def adinputs():
    phu = fits.PrimaryHDU()
    phu.header.update(OBSERVAT='Gemini-North', INSTRUME='NIRI',
                      ORIGNAME='N20010101S0001.fits')

    data = np.ones((2, 2))

    adinputs = []
    for i in range(2):
        ad = astrodata.create(phu)
        ad.append(data + i)
        adinputs.append(ad)
    return adinputs


def test_not_prepared(adinputs):
    p = NIRIImage(adinputs)
    with pytest.raises(IOError):
        p.stackFrames()


def test_error_only_one_file(adinputs, caplog):
    # With only one file
    p = NIRIImage(adinputs[1:])
    p.stackFrames()
    assert caplog.records[2].message == (
        'No stacking will be performed, since at least two input AstroData '
        'objects are required for stackFrames')


def test_error_extension_number(adinputs, caplog):
    p = NIRIImage(adinputs)
    p.prepare()
    adinputs[1].append(np.zeros((2, 2)))
    match = "Not all inputs have the same number of extensions"
    with pytest.raises(IOError, match=match):
        p.stackFrames()


def test_error_extension_shape(adinputs, caplog):
    adinputs[1][0].data = np.zeros((3, 3))
    p = NIRIImage(adinputs)
    p.prepare()
    match = "Not all inputs images have the same shape"
    with pytest.raises(IOError, match=match):
        p.stackFrames()


def test_stackframes_refcat_propagation(adinputs):
    refcat = Table([[1, 2], ['a', 'b']], names=('Id', 'Cat_Id'))
    for i, ad in enumerate(adinputs):
        if i > 0:
            refcat['Cat_Id'] = ['b', 'c']
        ad.REFCAT = refcat

    p = NIRIImage(adinputs)
    p.prepare()
    adout = p.stackFrames()[0]

    # The merged REFCAT should contain 'a', 'b', 'c'
    assert len(adout.REFCAT) == 3
    np.testing.assert_equal(adout.REFCAT['Id'], np.arange(1, 4))
    assert all(adout.REFCAT['Cat_Id'] == ['a', 'b', 'c'])


def test_rejmap(adinputs):
    for i in (2, 3, 4):
        adinputs.append(adinputs[0] + i)

    p = NIRIImage(adinputs)
    p.prepare()
    adout = p.stackFrames(reject_method='minmax', nlow=1, nhigh=1,
                          save_rejection_map=True)[0]
    assert_array_equal(adout[0].REJMAP, 2)  # rejected 2 values for each pixel
