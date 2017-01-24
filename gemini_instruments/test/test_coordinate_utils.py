# pytest suite
"""
Tests for coordinate_itils.

This is a suite of tests to be run with pytest.

To run:
    1) py.test -v --capture=no
"""
from datetime import datetime
from gemini_instruments import gmu

class TestCoordinateUtils:
    """
    Suite of tests for the functions in the coordinate utils module.
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

    def test_icrs(self):
        ret = gmu.toicrs('APPT', 23.0, 2.0, ut_datetime=datetime(2016, 11, 3))
        correct_value = (22.779814880901004, 1.913607699746111)
        for rv, cv in zip(ret, correct_value):
            assert abs(rv - cv) < 0.00001