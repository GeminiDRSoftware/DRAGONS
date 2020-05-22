import pytest

import numpy as np
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.library import transform
from geminidr.gmos.lookups import geometry_conf as geotable

# Star locations: Use unique y values to enable sorting
GMOS_STAR_LOCATIONS = ((200, 50), (204, 450), (4000, 50), (4004, 450))

astrofaker = pytest.importorskip("astrofaker")


@pytest.mark.parametrize('binning', (1, 2, 4))
def test_inverse_transform_gmos(binning):
    # Creates GMOS images with stars at predefined points
    ad = astrofaker.create('GMOS-N')
    ad.init_default_extensions(binning=binning, overscan=False)
    for ext in ad:
        ext.add(np.random.randn(*ext.shape))
        for ystar, xstar in GMOS_STAR_LOCATIONS:
            ext.add_star(amplitude=10000, x=xstar / binning, y=ystar / binning)

    adg = transform.create_mosaic_transform(ad, geotable)
    admos = adg.transform(attributes=None, order=1)
    adout = adg.inverse_transform(admos, order=3)
    p = GMOSImage([adout])
    p.detectSources()
    adout = p.streams['main'][0]
    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    for ext in adout:
        objcat = ext.OBJCAT
        objcat.sort(['Y_IMAGE'])
        for row, location in zip(objcat, GMOS_STAR_LOCATIONS):
            # OBJCAT is 1-indexed
            assert abs(row['Y_IMAGE'] - location[0] / ybin - 1) < 0.1
            assert abs(row['X_IMAGE'] - location[1] / xbin - 1) < 0.1
