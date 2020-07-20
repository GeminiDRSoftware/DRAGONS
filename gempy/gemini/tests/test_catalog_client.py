import pytest
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from gempy.gemini.gemini_catalog_client import get_fits_table


TESTS = [
    # (catalog, ra, dec, radius, nbresults)
    ('2mass', 180., 0., 0.02, 5),
    ('sdss9', 160., 30., 0.02, 9),
    ('gmos', 200., -60., 0.02, 8),
]


# These just check that all sources are within the search radius
# and that the number is the same as when it was written
@pytest.mark.skip(reason='requires internal network access')
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize('catalog, ra, dec, radius, nres', TESTS)
def test_get_fits_table(catalog, ra, dec, radius, nres):
    ret = get_fits_table(catalog, ra, dec, radius)
    assert len(ret) == nres

    center = SkyCoord(ra, dec, unit='deg')
    coord = SkyCoord(ret['RAJ2000'], ret['DEJ2000'], unit='deg')
    assert all(center.separation(coord) < Angle(radius * u.deg))
