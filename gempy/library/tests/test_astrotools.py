# pytest suite

"""
Tests for the astrotools module.

This is a suite of tests to be run with pytest.

To run:
   1) Set the environment variable GEMPYTHON_TESTDATA to the path that contains
      the file N20130510S0178_forStack.fits.
      Eg. /net/chara/data2/pub/ad_testdata/GMOS
   2) Then run: py.test -v   (must in gemini_python or have it in PYTHONPATH)
"""

#import os
#import os.path
import numpy as np
from gempy.library import astrotools

#TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
#TESTFITS = 'N20130510S0178_forStack.fits'

class TestAstrotools:
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

    def test_rasextodec(self):
        rastring = '20:30:40.506'
        ra = astrotools.rasextodec(rastring)
        assert abs(ra - 307.668775) < 0.0001

    def test_degsextodec(self):
        decstringneg = '-60:50:40.302'
        decstringpos = '60:50:40.302'
        decneg = astrotools.degsextodec(decstringneg)
        decpos = astrotools.degsextodec(decstringpos)
        assert abs(decneg + decpos - 0.) < 0.0001

    def test_get_corners_2d(self):
        corners = astrotools.get_corners((300, 500))
        assert corners == [(0, 0), (299, 0), (0, 499), (299, 499)]

    def test_get_corners_3d(self):
        corners = astrotools.get_corners((300, 500, 400))
        expected_corners = [(0, 0, 0), (299, 0, 0), (0, 499, 0),
                            (299, 499, 0), (0, 0, 399), (299, 0, 399),
                            (0, 499, 399), (299, 499, 399)]
        assert corners == expected_corners

    def test_rotate_2d(self):
        rotation_matrix = astrotools.rotate_2d(30.)
        expected_matrix = np.array([[0.8660254, -0.5],
                                    [ 0.5,0.8660254]])
        assert np.allclose(rotation_matrix, expected_matrix)

    def test_match_cxy(self):
        xx = np.arange(1, 11) * 1.
        yy = np.arange(1, 11) * 1.
        sx = np.arange(2, 12) * 1.
        sy = np.arange(2, 12) * 1.
        (indxy, indr) = astrotools.match_cxy(xx, yy, sx, sy)
        expected_indxy = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_indr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert (indxy == expected_indxy).all and (indr == expected_indr).all()

    def test_clipped_mean(self):
        dist = np.array([ 5,  5,  4,  7,  7,  4,  3,  5,  2,  6,  5, 12,  0,
                       9, 10, 13,  2, 14,  6,  3, 50])
        results = astrotools.clipped_mean(dist)
        expected_values = (6.1, 3.7)
        assert np.allclose(results, expected_values)

# TODO: Unit tests are incomplete
#     def test_get_records(self):
#         pass
#
#     def test_database_string(self):
#         pass
#
#     def test_get_function_with_penalities(self):
#         pass
#
#     def test_get_fitted_function(self):
#         pass
#
#
# class TestGaussFit:
#     """
#     Suite of tests for the methods in the GaussFit class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_method(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_model_gauss_1d(self):
#         pass
#
#     def test_model_gauss_2d(self):
#         pass
#
#     def test_call_diff(self):
#         pass
#
# class TestMoffatFit:
#     """
#     Suite of tests for the methods in the MoffatFit class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_method(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_model_moffat_2d(self):
#         pass
#
#     def test_calc_diff(self):
#         pass
#
#
# class TestWCSTweak:
#     """
#     Suite of tests for the methods in the WCSTweak class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_transform_ref(self):
#         pass
#
#     def test_calc_diff(self):
#         pass
#
#
# class TestRecord:
#     """
#     Suite of tests for the methods in the Record class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_aslist(self):
#         pass
#
#     def test_get_fields(self):
#         pass
#
#     def test_get_task_name(self):
#         pass
#
#     def test_read_array_field(self):
#         pass
#
#
# class TestIdentifyRecord:
#     """
#     Suite of tests for the methods in the IdentifyRecord class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_get_model_name(self):
#         pass
#
#     def test_get_nterms(self):
#         pass
#
#     def test_get_range(self):
#         pass
#
#     def test_get_coeff(self):
#         pass
#
#     def test_get_ydata(self):
#         pass
#
#
# class TestFitscoordsRecord:
#     """
#     Suite of tests for the methods in the FitscoordsRecord class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_get_coeff(self):
#         pass
#
#
# class TestIDB:
#     """
#     Suite of tests for the methods in the IDB class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_aslist(self):
#         pass
#
#
# class TestReidentifyRecord:
#     """
#     Suite of tests for the methods in the ReidentifyRecord class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_get_ydata(self):
#         pass
#
#
# class TestSpectralDatabase:
#     """
#     Suite of tests for the methods in the SpectralDatabase class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_identify_db_from_table(self):
#         pass
#
#     def test_fitcoords_db_from_table(self):
#         pass
#
#     def test_write_to_disk(self):
#         pass
#
#     def test_as_binary_table(self):
#         pass
#
#
# class TestFittedFunction:
#     """
#     Suite of tests for the methods in the FittedFunction class in astrotools
#     """
#
#     @classmethod
#     def setup_class(cls):
#         """Run once at the beginning."""
#         pass
#
#     @classmethod
#     def teardown_class(cls):
#         """Run once at the end."""
#         pass
#
#     def setup_method(self, method):
#         """Run once before every test."""
#         pass
#
#     def teardown_methode(self, method):
#         """Run once after every test."""
#         pass
#
#     def test_init(self):
#         pass
#
#     def test_get_params(self):
#         pass
#
#     def test_get_model_function(self):
#         pass
#
#     def test_get_stamp_data(self):
#         pass
#
#     def test_get_rsquared(self):
#         pass
#
#     def test_get_success(self):
#         pass
#
#     def test_get_name(self):
#         pass
#
#     def test_get_background(self):
#         pass
#
#     def test_get_peak(self):
#         pass
#
#     def test_get_center(self):
#         pass
#
#     def test_get_fwhm(self):
#         pass
#
#     def test_get_width(self):
#         pass
#
#     def test_get_theta(self):
#         pass
#
#     def test_get_beta(self):
#         pass
#
#     def test_get_fwhm_ellipticity(self):
#         pass
