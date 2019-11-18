#!/usr/bin/env python

import astrodata
import gemini_instruments
import pytest

from astrodata.testing import download_from_archive
from geminidr.f2.primitives_f2_image import F2Image


@pytest.mark.remote_data
def test_make_processed_flat():
    flat_files = [
        'S20131126S1111.fits',
        'S20131126S1112.fits',
        'S20131126S1113.fits',
    ]
    flat_files_full_path = [download_from_archive(f, path='F2')
                            for f in flat_files]

    ad_flat_inputs = [astrodata.open(f) for f in flat_files_full_path]

    p = F2Image(ad_flat_inputs)
    p.prepare()
    p.addDQ()


if __name__ == '__main__':

    pytest.main()
