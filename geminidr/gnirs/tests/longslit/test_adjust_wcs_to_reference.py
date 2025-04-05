#!/usr/bin/env python3
import os
import pytest
import numpy as np

import astrodata, gemini_instruments

from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from gempy.library import peak_finding


datasets = [
    ['N20220706S0306_distortionCorrected.fits',
     'N20220706S0307_distortionCorrected.fits',
     'N20220706S0308_distortionCorrected.fits',
     'N20220706S0309_distortionCorrected.fits']
]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
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

    params = {'max_apertures': 1, 'percentile': 80, 'min_sky_region': 50,
              'min_snr': 5.0, 'use_snr': True, 'threshold': 0.1, 'section': ""}

    adinputs = [astrodata.from_file(os.path.join(path_to_inputs, f)) for f in files]
    pixel_scale = adinputs[0].pixel_scale()
    centers = []
    # GMOS version can use pre-found apertures; GNIRS doesn't have findApertures()
    # at the same point in the recipe, so we have to find them manually here.
    for ad in adinputs:
        for ext in ad:
            locations, _, _, _ = peak_finding.find_apertures(
                ext.__class__(nddata=ext.nddata.T, phu=ad.phu, is_single=True),
                **params)
            centers.append(locations[0])

    # Hack the WCS of all but the first input so they're wrong
    for ad in adinputs[1:]:
        ad[0].wcs.pipeline[0].transform['crpix1'].offset = 600
    p = GNIRSLongslit(adinputs)
    p.adjustWCSToReference(fallback=None)

    # Check that the correlation offsets determined agree with the differences
    # between the aperture centers
    offsets = [float(rec.message.split()[-2]) for rec in caplog.records if 'pixels' in rec.message]
    for off, center in zip(offsets, centers[1:]):
        assert abs(off - (centers[0] - center)) < 0.3  # pixels

    # Return the (RA, dec) as a SkyCoord at the location of each aperture
    # Then confirm that the sky coordinates are all similar
    skycoords = [ad[0].wcs(center, 0, with_units=True)[1]
                  for ad, center in zip(adinputs, centers)]
    c0 = skycoords[0]
    for c in skycoords[1:]:
        assert c0.separation(c).arcsecond < 0.5 * pixel_scale
