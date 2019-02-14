#!/usr/bin/env python

import pytest
import os

import astrodata
import gemini_instruments
import geminidr


def test_f2_data_reduction(test_path):

    test_data_set = ['S20131126S1111.fits',]

    test_data_set_with_full_path = [
        os.path.join(test_path, 'F2', f) for f in test_data_set]

    test_ad_data = [astrodata.open(f) for f in test_data_set_with_full_path]

    p = geminidr.f2.primitives_f2.F2(test_ad_data)

    p.prepare()
    p.addDQ(add_illum_mask=False)
    p.addVAR(read_noise=True)
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackDarks()
    p.storeProcessedDark()




if __name__ == '__main__':
    pytest.main()

