import numpy as np
from gempy.library.nddops import NDStacker, sum1d
from geminidr.gemini.lookups import DQ_definitions as DQ
from astrodata import NDAstroData

def test_process_mask():
    # List of input DQ pixels, with correct output DQ
    # and the indices of pixels that should be used in the combination
    tests_and_results = (([5,9,8,8], 8, [2,3]),
                         ([0,1,2,3], 2, [0,2]),
                         ([2,4,0,1], 6, [0,1,2]),
                         ([2,4,8,8], 6, [0,1]),
                         ([8,1,9,1], 8, [0]))

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
        in_mask = np.array([[x, DQ.no_data] for x in mask_pixels]).astype(DQ.datatype)
        mask, out_mask = NDStacker._process_mask(in_mask)
        assert out_mask[0] == correct_output
        assert np.array_equal(mask.T[0], pixel_usage)

def test_varclip():
    # Confirm rejection of high pixel and correct output DQ
    data = np.array([1.,1.,2.,2.,2.,100.]).reshape(6,1)
    ndd = NDAstroData(data)
    ndd.mask = np.zeros_like(data, dtype=DQ.datatype)
    ndd.mask[5,0] = DQ.saturated
    ndd.variance = np.ones_like(data)
    stackit = NDStacker(combine="mean", reject="varclip")
    result = stackit(ndd)
    assert result == [1.6]
    assert result.mask == [0]

def test_sum1d():
    big_value = 10
    data = np.ones((10,))
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
            np.testing.assert_almost_equal(4 + big_value*(x2-3.5), result.data)
        else:
            np.testing.assert_almost_equal(x2-x1, result.data)
        np.testing.assert_almost_equal(x2-x1, result.variance)
        if x2 > 1.5:
            assert result.mask == 3
        elif x2 > 0.5:
            assert result.mask == 1
        else:
            assert result.mask == 0