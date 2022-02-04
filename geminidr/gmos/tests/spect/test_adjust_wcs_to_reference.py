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
def test_adjust_wcs(files, path_to_inputs):
    """
    We take preprocessed files with a single aperture and run them through
    adjustWCSToReference. After this, the celestial coordinates of the aperture
    should be similar in all images.

    In order to make this a real test, we edit the gWCS objects of the inputs
    so they're wrong (except for the first one, which is the reference)
    """
    adinputs = [astrodata.open(os.path.join(path_to_inputs, f)) for f in files]
    pixel_scale = adinputs[0].pixel_scale()

    # Hack the WCS of all but the first input so they're wrong
    for ad in adinputs[1:]:
        ad[0].wcs.pipeline[0].transform['crpix2'].offset = 600
    p = GMOSLongslit(adinputs)
    p.adjustWCSToReference()
    # Return the (RA, dec) as a SkyCoord at the location of each aperture
    # Then confirm that the sky coordinates are all similar
    skycoords = [ad[0].wcs(0, ad[0].APERTURE['c0'][0], with_units=True)[1]
                 for ad in adinputs]
    c0 = skycoords[0]
    for c in skycoords[1:]:
        assert c0.separation(c).arcsecond < pixel_scale


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_header_offset(adinputs2, caplog):
    """Test that the offset is correctly read from the headers."""
    p = GMOSLongslit(adinputs2)
    adout = p.adjustWCSToReference(method='offsets')

    for rec in caplog.records:
        assert not rec.message.startswith('WARNING')

    assert np.isclose(adout[0].phu['SLITOFF'], 0)
    assert np.isclose(adout[1].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[2].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[3].phu['SLITOFF'], 0)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.skip("Improved primitive doesn't fail any more")
def test_header_offset_fallback(adinputs2, caplog):
    """For this dataset the correlation method fails, and give an offset very
    different from the header one. So we check that the fallback to the header
    offset works.
    """
    p = GMOSLongslit(adinputs2)
    adout = p.adjustWCSToReference()

    # WARNING when offset is too large
    assert caplog.records[3].message.startswith('WARNING - No cross')

    assert np.isclose(adout[0].phu['SLITOFF'], 0)
    assert np.isclose(adout[1].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[2].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[3].phu['SLITOFF'], 0, atol=0.2, rtol=0)


# Todo: Implement recipe to create input files