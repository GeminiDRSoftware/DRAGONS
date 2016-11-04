# pytest suite
"""
Tests for gemini_catalog_client.

This is a suite of tests to be run with pytest.

To run:
    1) py.test -v --capture=no
"""
from gempy.gemini.gemini_catalog_client import get_fits_table
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

class TestCatalogClient:
    """
    Suite of tests for the functions in the gemini_tools module.
    """
    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        pass

    @classmethod
    def teardown_class(cls):
        """Run once at the end."""
        pass

    def setup_method(self, method):
        """Run once before every test."""
        pass

    def teardown_method(self, method):
        """Run once after every test."""
        pass

    # These just check that all sources are within the search radius
    # and that the number is the same as when it was written
    def test_2mass(self):
        ra = 180.0
        dec = 0.0
        radius = 0.02
        center = SkyCoord(ra, dec, unit='deg')
        ret = get_fits_table('2mass', ra, dec, radius)
        assert len(ret) == 5
        assert all(center.separation(SkyCoord(ret['RAJ2000'],
                        ret['DEJ2000'], unit='deg')) < Angle(radius * u.deg))

    def test_sdss9(self):
        ra = 160.0
        dec = 30.0
        radius = 0.02
        center = SkyCoord(ra, dec, unit='deg')
        ret = get_fits_table('sdss9', ra, dec, radius)
        assert len(ret) == 9
        assert all(center.separation(SkyCoord(ret['RAJ2000'],
                        ret['DEJ2000'], unit='deg')) < Angle(radius * u.deg))

    def test_gmos(self):
        ra = 200.0
        dec = -60.0
        radius = 0.02
        center = SkyCoord(ra, dec, unit='deg')
        ret = get_fits_table('gmos', ra, dec, radius)
        assert len(ret) == 8
        assert all(center.separation(SkyCoord(ret['RAJ2000'],
                        ret['DEJ2000'], unit='deg')) < Angle(radius * u.deg))

