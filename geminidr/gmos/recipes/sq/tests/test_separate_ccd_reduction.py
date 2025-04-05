import pytest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import vstack

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_image import GMOSImage

from astrodata.testing import download_from_archive
from recipe_system.reduction.coreReduce import Reduce

FILES = [f"N20200823S{i:04d}.fits" for i in (206, 207, 224)]

@pytest.mark.dragons_remote_data
def test_separate_ccd_reduction_astrometry(change_working_dir):
    """
    This test reduces a set of 3 GMOS images (4x4 binned for speed) via each
    of the 3 reduction recipes -- the default one, and the two that reduce
    each CCD separately. It then runs detectSources on each of the outputs
    and matches the world coordinates the objects to confirm they are very
    close. Some offsets will be large because the same objects aren't being
    matched in each of the images, so it compares the 25th percentile to
    check it's small.

    Obviously if there is, for example, a pixel shift, then the minimum
    separation will be a pixel and these comparisons will fail.
    """
    with change_working_dir():
        adoutputs = []
        file_list = [download_from_archive(f) for f in FILES]
        for recipe_name in ("reduce", "reduceSeparateCCDs",
                            "reduceSeparateCCDsCentral"):
            r = Reduce()
            r.files = file_list
            r.uparms = [("do_cal", "skip")]
            r.recipename = recipe_name
            r.suffix = f"_{recipe_name}"
            r.runr()
            adoutputs.append(astrodata.from_file(r._output_filenames[0]))

    p = GMOSImage(adoutputs)
    p.detectSources()
    cats = []
    for ad in adoutputs:
        t = vstack([ext.OBJCAT for ext in ad], metadata_conflicts='silent')
        cats.append(SkyCoord(t['X_WORLD'], t['Y_WORLD'], unit='deg'))
    c1, c2, c3 = cats
    pixscale = adoutputs[0].pixel_scale()
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    assert np.percentile(d2d.arcsec, 25) < 0.2 * pixscale
    idx, d2d, _ = c1.match_to_catalog_sky(c3)
    assert np.percentile(d2d.arcsec, 25) < 0.2 * pixscale
