import pytest

import datetime
import numpy as np

from astropy.modeling import models
from astropy import units as u
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS

from astrodata import wcs as adwcs

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect, make_wavelength_table

from . import ad_min  # minimum AD fixture


@pytest.mark.ghostspect
@pytest.mark.parametrize('ra,dec,dt,known_corr', [
    (90., -30., '2018-01-03 15:23:32', 0.999986388827),
    (180., -60., '2018-11-12 18:35:15', 1.00001645007),
    (270., -90., '2018-07-09 13:48:35', 0.999988565947),
    (0., -45., '2018-12-31 18:59:48', 0.99993510834),
    (101.1, 0., '2018-02-23 17:18:55', 0.999928361662),
])
def test_barycentricCorrect(ad_min, ra, dec, dt, known_corr):
    """
    Checks to make:

    - Make random checks that a non-1.0 correction factor works properly
    - Check before & after data shape

    Testing of the helper _compute_barycentric_correction is done
    separately.
    """
    # Add a wavl extension - no need to be realistic
    orig_wavl = np.random.rand(*ad_min[0].data.shape)
    input_frame = adwcs.pixel_frame(2)
    output_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                    axes_names=("AWAV",),
                                    name="Wavelength in air")
    # Needs to be transposed because of astropy x-first
    ad_min[0].wcs = gWCS([(input_frame, models.Tabular2D(lookup_table=orig_wavl.copy().T, name="WAVE")),
                          (output_frame, None)])

    ad_min.phu.set('RA', ra)
    ad_min.phu.set('DEC', dec)
    # Assume a 10 min exposure
    exp_time_min = 10.
    dt_obs = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    dt_start = dt_obs - datetime.timedelta(minutes=exp_time_min)
    ad_min.phu.set('DATE-OBS', dt_start.date().strftime('%Y-%m-%d'))
    ad_min.phu.set('UTSTART', dt_start.time().strftime('%H:%M:%S.00'))
    ad_min.phu.set('EXPTIME', exp_time_min * 60.)

    gs = GHOSTSpect([])
    ad_out = gs.barycentricCorrect([ad_min]).pop()
    new_wavl = make_wavelength_table(ad_out[0])
    corr_fact = (new_wavl / orig_wavl).mean()
    assert np.allclose(new_wavl / known_corr,
                       orig_wavl), "barycentricCorrect appears not to " \
                                   "have made a valid correction " \
                                   f"(should be {known_corr}, " \
                                   f"apparent correction {corr_fact})"

    assert ad_out.phu.get(
        gs.timestamp_keys['barycentricCorrect']), "barycentricCorrect did not " \
                                                  "timestamp-mark the " \
                                                  "output file"
