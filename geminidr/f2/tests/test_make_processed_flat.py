#!/usr/bin/env python

import astrodata
import gemini_instruments
import os
import pytest

from copy import deepcopy
from glob import glob

from geminidr.f2.primitives_f2_image import F2Image


@pytest.fixture
def test_path():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    return path


def test_make_processed_flat(test_path):

    flat_files = ['S20131126S1111.fits',
                  'S20131126S1112.fits',
                  'S20131126S1113.fits',
                  ]

    flat_files_full_path = [os.path.join(test_path, 'F2', f)
                            for f in flat_files]

    ad_flat_inputs = [astrodata.open(f) for f in flat_files_full_path]

    p = F2Image(ad_flat_inputs)
    p.prepare()
    p.addDQ()


if __name__ == '__main__':

    pytest.main()
