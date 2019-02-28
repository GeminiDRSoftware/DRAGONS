import numpy as np
from gempy.library.nddops import NDStacker
from geminidr.gemini.lookups import DQ_definitions as DQ

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
