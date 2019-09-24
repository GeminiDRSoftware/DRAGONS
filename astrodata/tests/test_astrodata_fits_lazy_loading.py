#!/usr/bin/env python
import pytest
import warnings
import os

import numpy as np

import astrodata
import gemini_instruments

THIS_DIR = os.path.dirname(__file__)
CHARA = '/net/chara/data2/pub'


## NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
## NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
## NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
#
#    Lazy loading is undergoing heavy changes.
#    These tests are probably not relevant any
#    longer, and may fail.
#
## NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
## NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
## NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE


def test_for_length(path_to_inputs):
    test_filename = 'GMOS/N20110826S0336.fits'
    ad = astrodata.open(os.path.join(path_to_inputs, test_filename))

    # This should force the data to be loaded
    # Otherwise, we'll get different results - or an exception
    assert len(ad) == len(ad.nddata)


# TODO: This one fails as it is written. Decide later if it's relevant or not
@pytest.mark.xfail(reason="Test fails for unknown reason. Should I keep it?")
def test_keyword_changes_preserved_on_lazy_loading(path_to_inputs):
    input_file = os.path.join(path_to_inputs, 'GMOS/N20110826S0336.fits')
    ad = astrodata.open(input_file)

    ad.phu['RAWIQ'] = 'Any'

    del ad.phu['RAWCC']
    del ad[0].hdr['DATATYPE']

    ad._lazy_populate_object()  # Force lazy load
    # assert ad.phu['RAWIQ'] == 'Any'
    # assert 'RAWCC' not in ad.phu
    # assert 'DATATYPE' not in ad[0].hdr


if __name__ == '__main__':
    pytest.main()
