import urllib

import pytest
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from gempy.gemini.gemini_catalog_client import get_fits_table


# These just check that all sources are within the search radius
# and that the number is the same as when it was written
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize('catalog, ra, dec, radius, nres', [
    ('2mass', 180., 0., 0.02, 5),
    ('sdss9', 160., 30., 0.02, 9),
    ('gmos', 200., -60., 0.02, 8),
])
def test_get_fits_table(catalog, ra, dec, radius, nres):
    """Test some requests with get_fits_table, which by default will use the
    internal server. So we skip the test if this server is no reachable.
    """
    # Test if internal server is available
    try:
        resp = urllib.request.urlopen("http://mkocatalog2")
        if resp.status != 200:
            raise Exception
    except Exception:
        pytest.skip('Internal server cannot be reached')

    ret = get_fits_table(catalog, ra, dec, radius)
    assert len(ret) == nres

    center = SkyCoord(ra, dec, unit='deg')
    coord = SkyCoord(ret['RAJ2000'], ret['DEJ2000'], unit='deg')
    assert all(center.separation(coord) < Angle(radius * u.deg))


# FIXME: vizier returns more sources than mkocatalog2 for sdss9
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize('catalog, ra, dec, radius, nres', [
    ('2mass', 180., 0., 0.02, 5),
    ('sdss9', 160., 30., 0.02, 28),
])
def test_get_fits_table_vizier(catalog, ra, dec, radius, nres):
    """Tests the same requests but on Vizier this time, specifying explicitly
    the server.
    """
    ret = get_fits_table(catalog, ra, dec, radius, server=f'{catalog}_vizier')
    assert len(ret) == nres

    center = SkyCoord(ra, dec, unit='deg')
    coord = SkyCoord(ret['RAJ2000'], ret['DEJ2000'], unit='deg')
    assert all(center.separation(coord) < Angle(radius * u.deg))
