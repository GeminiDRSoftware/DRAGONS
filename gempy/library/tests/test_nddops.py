import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)

from astrodata import NDAstroData
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.nddops import NDStacker, combine1d


@pytest.fixture
def testdata():
    return np.repeat(np.arange(5, dtype=float), 2, axis=0).reshape(5, 2)


@pytest.fixture
def testvar():
    data = np.repeat(np.arange(5, dtype=float), 2, axis=0).reshape(5, 2)
    return (data + 1) / 2


@pytest.fixture
def testmask():
    mask = np.zeros((5, 2), dtype=np.uint16)
    mask[:2, 1] = 1
    return mask


def test_process_mask():
    # List of input DQ pixels, with correct output DQ
    # and the indices of pixels that should be used in the combination
    tests_and_results = (
        ([5, 9, 8, 8], 8, [2, 3]),
        ([0, 1, 2, 3], 2, [0, 2]),
        ([2, 4, 0, 1], 6, [0, 1, 2]),
        ([2, 4, 8, 8], 6, [0, 1]),
        ([8, 1, 9, 1], 8, [0]),
    )

    for mask_pixels, correct_output, good_pixels in tests_and_results:
        pixel_usage = np.full_like(mask_pixels, DQ.max, dtype=DQ.datatype)
        pixel_usage[good_pixels] = 0

        # Test of _process_mask()
        in_mask = np.array(mask_pixels, dtype=DQ.datatype).reshape(-1, 1)
        mask, out_mask = NDStacker._process_mask(in_mask)
        assert out_mask[0] == correct_output
        assert np.array_equal(mask.T[0], pixel_usage)

        # Second test to confirm that additional iterations (required to
        # process the other output pixel) do not change output
        in_mask = np.array([[x, DQ.no_data] for x in mask_pixels],
                           dtype=DQ.datatype)
        mask, out_mask = NDStacker._process_mask(in_mask)
        assert out_mask[0] == correct_output
        assert np.array_equal(mask.T[0], pixel_usage)


def test_no_rejection(testdata):
    out_data, out_mask, out_var = NDStacker.none(testdata)
    assert_allclose(out_data, testdata)
    assert out_var is None
    assert out_mask is None


def test_unpack_nddata(testdata, testvar, testmask):
    nd = NDAstroData(testdata, mask=testmask, variance=testvar)
    out_data, out_mask, out_var = NDStacker.none(nd)
    assert_allclose(out_data, testdata)
    assert_allclose(out_var, testvar)
    assert_allclose(out_mask, testmask)


def test_ndstacker(capsys):
    stacker = NDStacker(combine="foo")
    assert capsys.readouterr().out == \
        'No such combiner as foo. Using mean instead.\n'
    assert stacker._combiner is NDStacker.mean

    stacker = NDStacker(reject="foo")
    assert capsys.readouterr().out == \
        'No such rejector as foo. Using none instead.\n'
    assert stacker._rejector is NDStacker.none


def test_varclip():
    # Confirm rejection of high pixel and correct output DQ
    data = np.array([1., 1., 2., 2., 2., 100.]).reshape(6, 1)
    ndd = NDAstroData(data,
                      mask=np.zeros_like(data, dtype=DQ.datatype),
                      variance=np.ones_like(data))
    ndd.mask[5, 0] = DQ.saturated
    stackit = NDStacker(combine="mean", reject="varclip")
    result = stackit(ndd)
    assert_allclose(result.data, 1.6)  # 100 is rejected
    assert_allclose(result.mask, 0)

    data = np.array([1., 1., 2., 2., 2., 100.]).reshape(6, 1)
    ndd = NDAstroData(data, variance=np.ones_like(data))
    ndd.variance[5] = 400
    stackit = NDStacker(combine="mean", reject="varclip", lsigma=3, hsigma=3)
    result = stackit(ndd)
    assert_allclose(result.data, 1.6)  # 100 is rejected

    stackit = NDStacker(combine="mean", reject="varclip", lsigma=5, hsigma=5)
    result = stackit(ndd)
    assert_allclose(result.data, 18)  # 100 is not rejected


def test_sigclip(capsys):
    # Confirm rejection of high pixel and correct output DQ
    data = np.array([1., 1., 1., 2., 2., 2., 2., 100.]).reshape(8, 1)
    ndd = NDAstroData(data)
    stackit = NDStacker(combine="mean",
                        reject="sigclip",
                        lsigma=3,
                        hsigma=3,
                        debug_pixel=0)
    result = stackit(ndd, save_rejection_map=True)
    assert_allclose(result.data, 1.5714285714285714)  # 100 is rejected
    assert result.meta['other']['REJMAP'].data[0] == 1

    out = capsys.readouterr().out
    expected = """\
Rejection: sigclip {'lsigma': 3, 'hsigma': 3}
img     data        mask    variance       immediately after rejection
  0          1.0000     0               -
  1          1.0000     0               -
  2          1.0000     0               -
  3          2.0000     0               -
  4          2.0000     0               -
  5          2.0000     0               -
  6          2.0000     0               -
  7        100.0000 32768               -
"""
    assert expected.splitlines() == out.splitlines()[13:23]

    stackit = NDStacker(combine="mean", reject="sigclip", lsigma=5, hsigma=5)
    result = stackit(ndd)
    assert_allclose(result.data, 13.875)  # 100 is not rejected


def test_combine():
    data = np.array([1., 1., 1., 2., 2., 2., 2., 100.]).reshape(8, 1)

    out_data, out_mask, out_var = NDStacker.combine(data,
                                                    combiner='mean',
                                                    rejector='sigclip')
    assert_allclose(out_data, 1.5714285)

    out_data, out_mask, out_var = NDStacker.combine(data, combiner='median')
    assert_allclose(out_data, 2)


def test_minmax(testvar, testmask):
    testdata = np.array([
        [24., 12.],
        [22., 14.],
        [23., 11.],
        [20., 13.],
        [21., 10.],
    ])

    out_data, out_mask, out_var = NDStacker.minmax(testdata)
    assert_array_equal(out_data, testdata)
    assert_array_equal(out_mask, False)

    out_data, out_mask, out_var = NDStacker.minmax(testdata, nlow=1, nhigh=1)
    assert_array_equal(out_data, testdata)
    assert_array_equal(out_mask, [
        [True, False],
        [False, True],
        [False, False],
        [True, False],
        [False, True],
    ])

    testmask = np.array(
        [[0, DQ.saturated], [0, DQ.saturated], [0, 0], [DQ.max, 0], [0, 0]],
        dtype=DQ.datatype)
    out_data, out_mask, out_var = NDStacker.minmax(testdata,
                                                   nlow=1,
                                                   nhigh=1,
                                                   mask=testmask)
    assert_array_equal(out_data, testdata)
    assert_array_equal(out_mask[:, 1], [4, DQ.max, 0, 0, DQ.max])
    assert_array_equal(out_mask, [
        [0, DQ.saturated],
        [0, DQ.max],
        [0, 0],
        [DQ.max, 0],
        [0, DQ.max],
    ])

    with pytest.raises(ValueError):
        NDStacker.minmax(testdata, nlow=3, nhigh=3)


def test_mean(testdata, testvar, testmask):
    out_data, out_mask, out_var = NDStacker.mean(testdata)
    assert_allclose(out_data, 2)
    assert_allclose(out_var, 0.5)
    assert out_mask is None

    out_data, out_mask, out_var = NDStacker.mean(testdata, variance=testvar)
    assert_allclose(out_data, 2)
    assert_allclose(out_var, 0.3)

    out_data, out_mask, out_var = NDStacker.mean(testdata, mask=testmask)
    assert_allclose(out_data, [2., 3.])
    assert_array_almost_equal(out_var, [0.5, 0.33], decimal=2)
    assert_allclose(out_mask, 0)

    out_data, out_mask, out_var = NDStacker.mean(testdata,
                                                 mask=testmask,
                                                 variance=testvar)
    assert_allclose(out_data, [2., 3.])
    assert_array_almost_equal(out_var, [0.3, 0.66], decimal=2)
    assert_allclose(out_mask, 0)


def test_wtmean(testdata, testvar, testmask):
    out_data, out_mask, out_var = NDStacker.wtmean(testdata)
    assert_allclose(out_data, 2)
    assert_allclose(out_var, 0.5)
    assert out_mask is None

    testvar = np.ones((5, 2))
    testvar[0, 0] = np.inf
    testvar[4, 1] = np.inf

    out_data, out_mask, out_var = NDStacker.wtmean(testdata, variance=testvar)
    assert_allclose(out_data, [2.5, 1.5])
    assert_allclose(out_var, 0.25)

    out_data, out_mask, out_var = NDStacker.wtmean(testdata, mask=testmask)
    assert_allclose(out_data, [2., 3.])

    out_data, out_mask, out_var = NDStacker.wtmean(testdata,
                                                   mask=testmask,
                                                   variance=testvar)
    assert_allclose(out_data, [2.5, 2.5])
    assert_allclose(out_var, [0.25, 0.5])
    assert_allclose(out_mask, 0)


@pytest.mark.parametrize('func', [NDStacker.median, NDStacker.lmedian])
def test_median_odd(func, testdata, testvar, testmask):
    out_data, out_mask, out_var = func(testdata)
    assert_allclose(out_data, 2)
    assert_array_almost_equal(out_var, 0.79, decimal=2)
    assert out_mask is None

    out_data, out_mask, out_var = func(testdata, variance=testvar)
    assert_allclose(out_data, 2)
    # FIXME: median and lmedian do not give the same result:
    # assert_allclose(out_var, 1.5)

    out_data, out_mask, out_var = func(testdata, mask=testmask)
    assert_allclose(out_data, [2., 3.])
    assert_array_almost_equal(out_var, [0.79, 0.52], decimal=2)
    assert_allclose(out_mask, 0)

    out_data, out_mask, out_var = func(testdata,
                                       mask=testmask,
                                       variance=testvar)
    assert_allclose(out_data, [2., 3.])
    assert_array_almost_equal(out_var, [0.47, 1.05], decimal=2)
    assert_allclose(out_mask, 0)


@pytest.mark.parametrize('func,expected_median,expected_var',
                         [(NDStacker.median, 2.5, 0.65),
                          (NDStacker.lmedian, 2, 0.79)])
def test_median_even(func, expected_median, expected_var, testdata, testvar,
                     testmask):
    testdata = testdata[1:]

    out_data, out_mask, out_var = func(testdata)
    assert_allclose(out_data, expected_median)
    assert_array_almost_equal(out_var, expected_var, decimal=2)
    assert out_mask is None

    out_data, out_mask, out_var = func(testdata, variance=testvar)
    assert_allclose(out_data, expected_median)
    # FIXME: median and lmedian do not give the same result:
    # assert_allclose(out_var, 1.5)


def test_combine1d_sum():
    big_value = 10
    data = np.ones((10, ))
    ndd = NDAstroData(data, mask=np.zeros_like(data, dtype=DQ.datatype),
                      variance=np.ones_like(data))
    ndd.mask[[1, 2]] = [1, 2]
    ndd.data[4] = big_value

    x1 = -0.5
    for x2 in np.arange(0., 4.5, 0.5):
        result = combine1d(ndd, x1, x2, proportional_variance=True)
        if x2 > 3.5:
            np.testing.assert_almost_equal(4 + big_value * (x2 - 3.5),
                                           result.data)
        else:
            np.testing.assert_almost_equal(x2 - x1, result.data)
        np.testing.assert_almost_equal(x2 - x1, result.variance)
        if x2 > 1.5:
            assert result.mask == 3
        elif x2 > 0.5:
            assert result.mask == 1
        else:
            assert result.mask == 0


def test_combine1d_average_with_mask():
    """Confirm that the average is always 1 even if there's a masked bad pixel"""
    big_value = 10
    data = np.ones((10, ))
    ndd = NDAstroData(data, mask=np.zeros_like(data, dtype=DQ.datatype),
                      variance=np.ones_like(data))
    ndd.data[4] = big_value
    ndd.mask[4] = 1

    x1 = -0.5
    for x2 in np.arange(0., 8.5, 0.5):
        result = combine1d(ndd, x1, x2, average=True)
        assert result.data == pytest.approx(1.0)


def test_combine_1d_average_and_sum():
    """Confirm that average and sum give consistent results"""
    data = np.arange(10)
    ndd = NDAstroData(data, mask=np.zeros_like(data, dtype=DQ.datatype),
                      variance=np.ones_like(data))
    for x1 in np.arange(-0.5, 7.5, 0.9):
        for x2 in np.arange(x1+0.7, x1+4, 0.8):
            dx = np.minimum(x2, 9.5) - np.maximum(x1, -0.5)
            sum1d_result = combine1d(ndd, x1, x2, average=False)
            avg1d_result = combine1d(ndd, x1, x2, average=True)
            assert sum1d_result.data / dx == pytest.approx(avg1d_result.data)
            assert sum1d_result.variance / (dx*dx) == pytest.approx(avg1d_result.variance)
