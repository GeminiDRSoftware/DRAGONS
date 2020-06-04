import os
import pytest

import astrodata, gemini_instruments

from geminidr.gmos import primitives_gmos_longslit

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
def test_adjust_wcs(files, path_to_inputs):
    """
    We take preprocessed files with a single aperture and run them through
    adjustWCSToReference. After this, the celestial coordinates of the aperture
    should be similar in all images.

    In order to make this a real test, we edit the gWCS objects of the inputs
    so they're wrong (except for the first one, which is the reference)
    """
    adinputs = [astrodata.open(os.path.join(path_to_inputs, f)) for f in files]
    # Hack the WCS of all but the first input so they're wrong
    for ad in adinputs[1:]:
        ad[0].wcs.pipeline[0][1]['crpix2'].offset = 600
    p = primitives_gmos_longslit.GMOSLongslit(adinputs)
    p.adjustWCSToReference()
    # Return the (RA, dec) as a SkyCoord at the location of each aperture
    # Then confirm that the sky coordinates are all similar
    skycoords = [ad[0].wcs(0, ad[0].APERTURE['c0'][0], with_units=True)[1]
                 for ad in adinputs]
    c0 = skycoords[0]
    for c in skycoords[1:]:
        assert c0.separation(c).arcsecond < 0.05
