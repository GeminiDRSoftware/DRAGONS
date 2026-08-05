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


# Datasets and expected offsets
flip_datasets = [('N20240904S0010_skyCorrected.fits', None),
                 ('N20240904S0011_skyCorrected.fits', -124.5),
                 ('N20240907S0029_skyCorrected.fits', 156.5),
                 ('N20240907S0030_skyCorrected.fits', 33.2),
                 ('N20240908S0021_skyCorrected.fits', 30.0),
                 ('N20240908S0022_skyCorrected.fits', 153.5),
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
    adinputs = [astrodata.open(os.path.join(path_to_inputs, f)) for f in files]
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
    skycoords = [ad[0].wcs.pixel_to_world(0, ad[0].APERTURE['c0'][0])[1]
                 for ad in adinputs]
    c0 = skycoords[0]
    for c in skycoords[1:]:
        assert c0.separation(c).arcsecond < 0.5 * pixel_scale


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_adjust_and_resample_with_flip(path_to_inputs, caplog):
    """
    Check that adjustWCSToReference() copes when some data are taken at an
    antiparallel slit PA.
    """
    caplog.set_level(20)
    adinputs = [astrodata.open(os.path.join(path_to_inputs, f[0]))
                for f in flip_datasets]
    p = GMOSLongslit(adinputs)
    p.adjustWCSToReference(tolerance=3)
    offsets = [float(rec.message.split()[-2]) for rec in caplog.records
               if 'pixels' in rec.message]

    # Check offsets are what's expected
    for offset, expected in zip(offsets, flip_datasets[1:]):
        assert offset == pytest.approx(expected[1], abs=0.5)

    # Check that the APERTURE table has been flipped and shifted by
    # checking the brightest source
    p.resampleToCommonFrame()
    for ad in p.streams['main']:
        assert ad[0].APERTURE['c0'][0] == pytest.approx(1773.6, abs=1)


# Todo: Implement recipe to create input files
