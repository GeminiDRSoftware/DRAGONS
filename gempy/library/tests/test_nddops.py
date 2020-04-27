import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from astrodata import NDAstroData
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.nddops import NDStacker, sum1d


@pytest.fixture
def testdata():
    return np.repeat(np.arange(5, dtype=float), 2, axis=0).reshape(5, 2)


@pytest.fixture
def testvar():
    data = np.repeat(np.arange(5, dtype=float), 2, axis=0).reshape(5, 2)
    return (data+1) / 2


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
        pixel_usage = np.full_like(mask_pixels, 32768).astype(DQ.datatype)
        pixel_usage[good_pixels] = 0

        # Test of _process_mask()
        in_mask = np.array([[x] for x in mask_pixels]).astype(DQ.datatype)
        mask, out_mask = NDStacker._process_mask(in_mask)
        assert out_mask == correct_output
        assert np.array_equal(mask.T[0], pixel_usage)

        # Second test to confirm that additional iterations (required to
        # process the other output pixel) do not change output
        in_mask = np.array([[x, DQ.no_data]
                            for x in mask_pixels]).astype(DQ.datatype)
        mask, out_mask = NDStacker._process_mask(in_mask)
        assert out_mask[0] == correct_output
        assert np.array_equal(mask.T[0], pixel_usage)


def test_no_rejection(testdata):
    out_data, out_mask, out_var = NDStacker.none(testdata)
    assert_allclose(out_data, testdata)
    assert out_var is None
    assert out_mask is None


def test_varclip():
    # Confirm rejection of high pixel and correct output DQ
    data = np.array([1., 1., 2., 2., 2., 100.]).reshape(6, 1)
    ndd = NDAstroData(data)
    ndd.mask = np.zeros_like(data, dtype=DQ.datatype)
    ndd.mask[5, 0] = DQ.saturated
    ndd.variance = np.ones_like(data)
    stackit = NDStacker(combine="mean", reject="varclip")
    result = stackit(ndd)
    assert_allclose(result.data, 1.6)  # 100 is rejected
    assert_allclose(result.mask, 0)

    data = np.array([1., 1., 2., 2., 2., 100.]).reshape(6, 1)
    ndd = NDAstroData(data)
    ndd.variance = np.ones_like(data)
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
    result = stackit(ndd)
    assert_allclose(result.data, 1.5714285714285714)  # 100 is rejected
    out = capsys.readouterr().out
    assert """\
Rejection: sigclip {'lsigma': 3, 'hsigma': 3}
img     data        mask    variance       after rejection
  0          1.0000     0               -
  1          1.0000     0               -
  2          1.0000     0               -
  3          2.0000     0               -
  4          2.0000     0               -
  5          2.0000     0               -
  6          2.0000     0               -
  7        100.0000     1               -
""" in out

    stackit = NDStacker(combine="mean", reject="sigclip", lsigma=5, hsigma=5)
    result = stackit(ndd)
    assert_allclose(result.data, 13.875)  # 100 is not rejected


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


@pytest.mark.xfail(reason='review this')
def test_sum1d():
    big_value = 10
    data = np.ones((10, ))
    ndd = NDAstroData(data)
    ndd.mask = np.zeros_like(data, dtype=DQ.datatype)
    ndd.mask[1] = 1
    ndd.mask[2] = 2
    ndd.data[4] = big_value
    ndd.variance = np.ones_like(data)
    x1 = -0.5
    for x2 in np.arange(0., 4.5, 0.5):
        result = sum1d(ndd, x1, x2, proportional_variance=True)
        if x2 > 3.5:
            np.testing.assert_almost_equal(4 + big_value * (x2-3.5),
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
