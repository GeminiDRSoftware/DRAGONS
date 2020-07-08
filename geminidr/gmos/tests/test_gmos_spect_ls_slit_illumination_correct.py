#!/usr/bin/env python
"""
Tests for the `p.slitIlluminationCorrect` primitive.
"""

import numpy as np
import pytest

from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

astrofaker = pytest.importorskip("astrofaker")


def test_dont_do_slit_illumination():

    in_ad = astrofaker.create("GMOS-S", mode="SPECT")

    p = GMOSLongslit([in_ad])
    out_ad = p.slitIlluminationCorrect(do_illum=False)[0]

    for in_ext, out_ext in zip(in_ad, out_ad):
        assert np.testing.assert_equal(in_ext.data, out_ext.data)




if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        pass
    else:
        pytest.main()
