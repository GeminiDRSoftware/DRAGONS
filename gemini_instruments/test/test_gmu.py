# pytest suite
"""
Tests for coordinate_itils.

This is a suite of tests to be run with pytest.

To run:
    1) py.test -v --capture=no
"""
from datetime import datetime

from gemini_instruments import gmu


class TestGMU:
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

    def test_removeComponentID1(self):
        value = 'H_G0999'
        assert ('H', gmu.removeComponentID(value))

    def test_removeComponentID2(self):
        value = 'H'
        assert ('H', gmu.removeComponentID(value))

    def test_sectionStrToIntList(self):
        section = '[1:2,3:4]'
        expected = (0, 2, 2, 4)
        assert expected == gmu.sectionStrToIntList(section)

    def test_parse_percentile1(self):
        string = 'Any'
        assert 100 == gmu.parse_percentile(string)

    def test_parse_percentile2(self):
        string = '80-percentile'
        assert 80 == gmu.parse_percentile(string)

    def test_parse_percentile3(self):
        string = '80'
        assert gmu.parse_percentile(string) is None
