import pytest
import tempfile
import os

import numpy as np

import astrodata
import gemini_instruments

from common_astrodata_test import from_test_data, from_chara

# Object construction
def test_for_length():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert len(ad) == 3

def test_append_array_to_root_no_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    ad.append(np.ones((100, 100)))

def test_append_array_to_root():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ad.append(np.ones((100, 100)), reset_ver=True)
    assert len(ad) == (lbefore + 1)
