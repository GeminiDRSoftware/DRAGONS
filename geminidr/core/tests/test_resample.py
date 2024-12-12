"""
Tests for primitives_resample
"""
import pytest

import os
from copy import deepcopy
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord

from geminidr.niri.primitives_niri_image import NIRIImage


@pytest.mark.parametrize('trim_data', (False, True))
@pytest.mark.parametrize('file_write', (False, True))
@pytest.mark.parametrize('separator', (' ', '   ', '\t'))
def test_shift_images(astrofaker, separator, trim_data, file_write, path_to_outputs):
    """
    Creates several fake AD objects with a single source in different
    locations and then shifts them. Checks that the output image sizes are
    correct and shifts are correct from the location of the brightest pixel

    This tests trimming and not trimming images, and passing the shifts
    via a string and a file on disk.
    """
    offsets = ((-10, -10), (-10, 10), (5, 10), (10, -20), (0, 0))
    min_xoff = min(off[0] for off in offsets)
    min_yoff = min(off[1] for off in offsets)
    max_xoff = max(off[0] for off in offsets)
    max_yoff = max(off[1] for off in offsets)

    orig_adinputs = []
    coords = []
    for i, (xoff, yoff) in enumerate(offsets, start=1):
        ad = astrofaker.create('NIRI', filename=f'test{i}.fits')
        ad.init_default_extensions()
        x, y = 512 - xoff, 512 - yoff
        ad[0].add_star(amplitude=1000, x=x, y=y)
        orig_adinputs.append(ad)
        coords.append(SkyCoord(*ad[0].wcs(x, y), unit=u.deg))

    if file_write:
        shifts_par = os.path.join(path_to_outputs, 'shifts.lis')
        f = open(shifts_par, 'w')
        for xoff, yoff in offsets:
            f.write(f'{xoff}{separator}{yoff}\n')
        f.close()
    else:
        shifts_par = ':'.join([f'{xoff},{yoff}' for xoff, yoff in offsets])

    adinputs = [deepcopy(ad) for ad in orig_adinputs]
    p = NIRIImage(adinputs)
    p.shiftImages(shifts=shifts_par, trim_data=trim_data)

    for ad, coord in zip(p.streams['main'], coords):
        shape = (1024, 1024) if trim_data else (1024 + max_yoff - min_yoff,
                                                1024 + max_xoff - min_xoff)
        assert ad[0].shape == shape
        peak_loc = (512, 512) if trim_data else (512 - min_yoff, 512 - min_xoff)
        y, x = np.unravel_index(ad[0].data.argmax(), ad[0].shape)
        assert (y, x) == peak_loc
        new_coord = SkyCoord(*ad[0].wcs(x, y), unit=u.deg)
        assert coord.separation(new_coord) < 1e-6 * u.arcsec  # should be identical

    if file_write:
        os.remove(shifts_par)
