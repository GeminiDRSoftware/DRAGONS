import os
import pytest
import numpy as np

import astrodata, gemini_instruments

from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit


# These are all proprietary data and are only available in our test server
datasets = [
    ['N20200308S0047_aperturesTraced.fits',
     'N20200308S0048_aperturesTraced.fits',
     'N20200308S0049_aperturesTraced.fits',
     'N20200308S0050_aperturesTraced.fits'],
]


# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("files", datasets)
def test_adjust_wcs_with_correlation(files, path_to_inputs, caplog):
    """
    We take preprocessed files with a single aperture and run them through
    adjustWCSToReference. After this, the celestial coordinates of the aperture
    should be similar in all images.

    In order to make this a real test, we edit the gWCS objects of the inputs
    so they're wrong (except for the first one, which is the reference)

    NB. WCS adjustment from the header offsets is handled by resampling tests
    in test_resample_2d.py
    """
    caplog.set_level(20)
    adinputs = [astrodata.from_file(os.path.join(path_to_inputs, f)) for f in files]
    pixel_scale = adinputs[0].pixel_scale()
    centers = [ad[0].APERTURE['c0'][0] for ad in adinputs]

    # Hack the WCS of all but the first input so they're wrong
    for ad in adinputs[1:]:
        ad[0].wcs.pipeline[0].transform['crpix2'].offset = 600
    p = GMOSLongslit(adinputs)
    p.adjustWCSToReference(fallback=None)

    # Check that the correlation offsets determined agree with the differences
    # between the aperture centers
    offsets = [float(rec.message.split()[-2]) for rec in caplog.records if 'pixels' in rec.message]
    for off, center in zip(offsets, centers[1:]):
        assert abs(off - (centers[0] - center)) < 0.3  # pixels

    # Return the (RA, dec) as a SkyCoord at the location of each aperture
    # Then confirm that the sky coordinates are all similar
    skycoords = [ad[0].wcs(0, ad[0].APERTURE['c0'][0], with_units=True)[1]
                 for ad in adinputs]
    c0 = skycoords[0]
    for c in skycoords[1:]:
        assert c0.separation(c).arcsecond < 0.5 * pixel_scale


# Todo: Implement recipe to create input files