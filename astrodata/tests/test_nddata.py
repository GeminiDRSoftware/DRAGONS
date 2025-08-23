import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from astrodata.fits import windowedOp
from astrodata import wcs as adwcs
from astrodata.nddata import ADVarianceUncertainty, NDAstroData
from astropy.io import fits
from astropy.nddata import NDData, VarianceUncertainty
from astropy.modeling import models
from astropy.table import Table
from gwcs.wcs import WCS as gWCS
from gwcs.coordinate_frames import Frame2D


@pytest.fixture
def testnd():
    shape = (5, 5)
    hdr = fits.Header({'CRPIX1': 1, 'CRPIX2': 2})
    nd = NDAstroData(data=np.arange(np.prod(shape)).reshape(shape),
                     variance=np.ones(shape) + 0.5,
                     mask=np.zeros(shape, dtype=bool),
                     wcs=gWCS(models.Shift(1) & models.Shift(2),
                              input_frame=adwcs.pixel_frame(2),
                              output_frame=adwcs.pixel_frame(2, name='world')),
                     unit='ct')
    nd.meta['other'] = {'OBJMASK': np.arange(np.prod(shape)).reshape(shape),
                        'OBJCAT': Table([[1,2,3]], names=[['number']])}
    nd.mask[3, 4] = True
    return nd


def _stack(arrays):
    arrays = [x for x in arrays]
    data = np.array([arr.data for arr in arrays]).sum(axis=0)
    unc = np.array([arr.uncertainty.array for arr in arrays]).sum(axis=0)
    mask = np.array([arr.mask for arr in arrays]).sum(axis=0)
    return NDAstroData(data=data, variance=unc, mask=mask)


def test_var(testnd):
    data = np.zeros(5)
    var = np.array([1.2, 2, 1.5, 1, 1.3])
    nd1 = NDAstroData(data=data, uncertainty=ADVarianceUncertainty(var))
    nd2 = NDAstroData(data=data, variance=var)
    assert_array_equal(nd1.variance, nd2.variance)


def test_window(testnd):
    win = testnd.window[2:4, 3:5]
    assert win.unit == 'ct'
    #assert_array_equal(win.wcs.wcs.crpix, [1, 2])
    assert_array_equal(win.data, [[13, 14], [18, 19]])
    assert_array_equal(win.mask, [[False, False], [False, True]])
    assert_array_almost_equal(win.uncertainty.array, 1.5)
    assert_array_almost_equal(win.variance, 1.5)


def test_windowedOp(testnd):
    result = windowedOp(_stack, [testnd, testnd],
                        kernel=(3, 3),
                        with_uncertainty=True,
                        with_mask=True)
    assert_array_equal(result.data, testnd.data * 2)
    assert_array_equal(result.uncertainty.array, testnd.uncertainty.array * 2)
    assert result.mask[3, 4] == 2

    nd2 = NDAstroData(data=np.zeros((4, 4)))
    with pytest.raises(ValueError, match=r"Can't calculate final shape.*"):
        result = windowedOp(_stack, [testnd, nd2], kernel=(3, 3))

    with pytest.raises(AssertionError, match=r"Incompatible shape.*"):
        result = windowedOp(_stack, [testnd, testnd], kernel=[3], shape=(5, 5))


def test_windowedOp_with_result(testnd):
    result = NDAstroData(data=np.empty_like(testnd.data),
                         variance=np.empty_like(testnd.data),
                         mask=np.empty_like(testnd.data, dtype=np.uint16))
    windowedOp(_stack, [testnd, testnd],
               kernel=(3, 3),
               result=result,
               with_uncertainty=True,
               with_mask=True)

    assert_array_equal(result.data, testnd.data * 2)
    assert_array_equal(result.uncertainty.array, testnd.uncertainty.array * 2)
    assert result.mask[3, 4] == 2

    nd2 = NDAstroData(data=np.zeros((4, 4)))
    with pytest.raises(ValueError, match=r"Can't calculate final shape.*"):
        result = windowedOp(_stack, [testnd, nd2], kernel=(3, 3))

    with pytest.raises(AssertionError, match=r"Incompatible shape.*"):
        result = windowedOp(_stack, [testnd, testnd], kernel=[3], shape=(5, 5))


def test_transpose(testnd):
    testnd.variance[0, -1] = 10
    ndt = testnd.T
    assert_array_equal(ndt.data[0], [0, 5, 10, 15, 20])
    assert ndt.variance[-1, 0] == 10
    assert ndt.wcs(1, 2) == testnd.wcs(2, 1)


def test_set_section(testnd):
    sec = NDData(np.zeros((2, 2)),
                 uncertainty=VarianceUncertainty(np.ones((2, 2))))
    testnd.set_section((slice(0, 2), slice(1, 3)), sec)
    assert_array_equal(testnd[:2, 1:3].data, 0)
    assert_array_equal(testnd[:2, 1:3].variance, 1)


def test_uncertainty_negative_numbers():
    arr = np.zeros(5)

    # No warning if all 0
    with warnings.catch_warnings(record=True) as w:
        ADVarianceUncertainty(arr)
    assert len(w) == 0

    arr[2] = -0.001

    with pytest.warns(RuntimeWarning, match='Negative variance values found.'):
        result = ADVarianceUncertainty(arr)

    assert not np.all(arr >= 0)
    assert isinstance(result, ADVarianceUncertainty)
    assert result.array[2] == 0

    # check that it always works with a VarianceUncertainty instance
    result.array[2] = -0.001

    with pytest.warns(RuntimeWarning, match='Negative variance values found.'):
        result2 = ADVarianceUncertainty(result)

    assert not np.all(arr >= 0)
    assert not np.all(result.array >= 0)
    assert isinstance(result2, ADVarianceUncertainty)
    assert result2.array[2] == 0


def test_wcs_slicing():
    nd = NDAstroData(np.zeros((50, 50)))
    in_frame = Frame2D(name="in_frame")
    out_frame = Frame2D(name="out_frame")
    nd.wcs = gWCS([(in_frame, models.Identity(2)),
                   (out_frame, None)])
    assert nd.wcs(10, 10) == (10, 10)
    assert nd[10:].wcs(10, 10) == (10, 20)
    assert nd[..., 10:].wcs(10, 10) == (20, 10)
    assert nd[:, 5].wcs(10) == (5, 10)
    assert nd[20, -10:].wcs(0) == (40, 20)
    # and with flips
    assert nd[::-1].wcs(10, 10) == (10, 39)
    assert nd[:, ::-1].wcs(10, 10) == (39, 10)


def test_access_to_other_planes(testnd):
    assert hasattr(testnd, 'OBJMASK')
    assert testnd.OBJMASK.shape == testnd.data.shape
    assert hasattr(testnd, 'OBJCAT')
    assert isinstance(testnd.OBJCAT, Table)
    assert len(testnd.OBJCAT) == 3


def test_access_to_other_planes_when_windowed(testnd):
    ndwindow = testnd.window[1:, 1:]
    assert ndwindow.data.shape == (4, 4)
    assert ndwindow.data[0, 0] == testnd.shape[1] + 1
    assert ndwindow.OBJMASK.shape == (4, 4)
    assert ndwindow.OBJMASK[0, 0] == testnd.shape[1] + 1
    assert isinstance(ndwindow.OBJCAT, Table)
    assert len(ndwindow.OBJCAT) == 3


# Basically the same test as above but using slicing.
def test_access_to_other_planes_when_sliced(testnd):
    ndwindow = testnd[1:, 1:]
    assert ndwindow.data.shape == (4, 4)
    assert ndwindow.data[0, 0] == testnd.shape[1] + 1
    assert ndwindow.OBJMASK.shape == (4, 4)
    assert ndwindow.OBJMASK[0, 0] == testnd.shape[1] + 1
    assert isinstance(ndwindow.OBJCAT, Table)
    assert len(ndwindow.OBJCAT) == 3


if __name__ == '__main__':
    pytest.main()
