# pytest suite

"""
Tests for the matching module.

This is a suite of tests to be run with pytest.

To run:
   1) py.test -v   (must in gemini_python or have it in PYTHONPATH)
"""

import numpy as np
from gempy.library import matching

class TestMatching:
    """
    Suite of tests for the functions in the astrotools module.
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

    def make_catalog(self, nsources, range):
        """Create a catalog of nsources fake positions in range 0-range
        for each coordinate, keeping away from the edges"""
        x = np.random.uniform(low=0.05*range, high=0.95*range, size=nsources)
        y = np.random.uniform(low=0.05*range, high=0.95*range, size=nsources)
        return (x, y)

    def transform_coords(self, coords, model, scatter):
        """Transform input coordinates to output coordinates using the
        given model, and add Gaussian scatter of given rms"""
        xin, yin = coords
        xout, yout = model(xin, yin)
        xout += np.random.normal(loc=0.0, scale=scatter, size=len(xout))
        yout += np.random.normal(loc=0.0, scale=scatter, size=len(yout))
        return (xout, yout)

    def test_fit_brute_then_simplex(self):
        nsources = 100
        sig = 1.0
        xshift, yshift = 5.0, 10.0
        incoords = self.make_catalog(nsources, 1024)
        real_model = matching.Shift2D(xshift, yshift)
        refcoords = self.transform_coords(incoords, real_model, sig)
        in_model = real_model.copy()
        for p in in_model.param_names:
            setattr(in_model, p, 0.0)
        model = matching.fit_brute_then_simplex(in_model, incoords, refcoords)
        for p in model.param_names:
            assert (abs(getattr(model, p) - getattr(real_model, p)) <
                    3.0*sig/np.sqrt(nsources))

    def test_find_offsets(self):
        nsources = 100
        sig = 1.0
        xshift, yshift = 5.0, 10.0
        xin, yin = self.make_catalog(nsources, 1024)
        real_model = matching.Shift2D(xshift, yshift)
        xref, yref = self.transform_coords((xin, yin), real_model, sig)
        dx, dy = matching.find_offsets(xin, yin, xref, yref)
        # Quantized pixel-based cross-correlation so allow a 1-pixel scatter
        assert (abs(dx - xshift) <= max(3.0*sig/np.sqrt(nsources), 1))
        assert (abs(dy - yshift) <= max(3.0*sig/np.sqrt(nsources), 1))

    def test_align_catalogs(self):
        nsources = 100
        sig = 1.0
        tol = 0.01
        xshift, yshift = 5.0, 10.0
        xin, yin = self.make_catalog(nsources, 1024)
        real_model = matching.Shift2D(xshift, yshift)
        xref, yref = self.transform_coords((xin, yin), real_model, sig)
        model = matching.align_catalogs(xin, yin, xref, yref,
                                        translation_range=50, tolerance=tol)
        for p in model.param_names:
            assert (abs(getattr(model, p) - getattr(real_model, p)) <
                    max(3.0*sig/np.sqrt(nsources), tol))

    def test_match_sources(self):
        yin, xin = np.mgrid[0:5, 0:5]
        xref = np.array([2.1, 3.1])
        yref = np.array([3.9, 0.1])
        matched = matching.match_sources((xin.ravel(), yin.ravel()),
                                         (xref, yref))
        assert matched[22] == 0
        assert matched[3] == 1
        # Now add priority so input element #4 beats closer #3
        matched = matching.match_sources((xin.ravel(), yin.ravel()),
                                         (xref, yref), priority=[4])
        assert matched[22] == 0
        assert matched[4] == 1