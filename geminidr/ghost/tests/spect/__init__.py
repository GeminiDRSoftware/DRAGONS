import pytest

import astrodata, gemini_instruments
from astropy.io import fits
import numpy as np


@pytest.fixture
def ad_min():
    rawfilename = 'test_data.fits'

    # Create the AstroData object
    phu = fits.PrimaryHDU()
    phu.header.set('INSTRUME', 'GHOST')
    phu.header.set('DATALAB', 'test')
    phu.header.set('CAMERA', 'RED')
    phu.header.set('CCDNAME', 'E2V-CCD-231-C6')
    phu.header.set('CCDSUM', '1 1')

    # Create a simple data HDU with a zero BPM
    sci = fits.ImageHDU(data=np.ones((1024, 1024,), dtype=np.float32),
                        name='SCI')
    sci.header.set('CCDSUM', '1 1')
    sci.header.set('DATASEC', '[1:1024,1:1024]')
    sci.header.set('DETSEC', '[1:1024,1:1024]')

    ad = astrodata.create(phu, [sci, ])
    ad.filename = rawfilename
    return ad

