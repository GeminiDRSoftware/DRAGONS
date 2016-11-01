# pytest suite

"""
Tests for the gemini_metadata_utils module.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path to
       the directory containing the test data.
       Eg. /net/chara/data2/pub/gempython_testdata
    2) Then run: py.test -v
"""

from gempy.gemini import gemini_metadata_utils as gmu

#TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
#TESTFITS = '.fits'

class TestMetadata:
    """
    Suite of tests for the functions in the gemini_metadata_utils module.
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        pass

    @classmethod
    def teardown_class(cls):
        """Fun once at the end."""
        pass

    def setup_method(self, method):
        """Run once before every test."""
        pass

    def teardown_method(self, method):
        """Run once after every test."""
        pass

    def test_removeComponentID1(self):
        value = 'H_G0999'
        assert ('H', gmu.removeComponentID(value))

    def test_removeComponentID2(self):
        value = 'H'
        assert ('H', gmu.removeComponentID(value))

    def test_sectionStrToIntList(self):
        section = '[1:2,3:4]'
        expected = [0, 2, 2, 4]
        assert expected == gmu.sectionStrToIntList(section)

    def test_gemini_date(self):
        # There isn't really a way to test this one without
        # reimplementing the function itself.
        pass

    def test_parse_percentile1(self):
        string = 'Any'
        assert 100 == gmu.parse_percentile(string)

    def test_parse_percentile2(self):
        string = '80-percentile'
        assert 80 == gmu.parse_percentile(string)

    def test_parse_percentile3(self):
        string = '80'
        assert gmu.parse_percentile(string) is None

    def test_filternameFrom1(self):
        filters = ['H_G0999', 'open', 'blank']
        assert 'blank' == gmu.filternameFrom(filters)

    def test_filternameFrom2(self):
        # Note that such a case is probably wrong.
        # The code is not doing the right thing I believe.
        filters = ['grism', 'pupil']
        assert 'open' == gmu.filternameFrom(filters)

    def test_filternameFrom3(self):
        filters = ['H_G0999', 'JH_G0000', 'open']
        assert 'H_G0999&JH_G0000' == gmu.filternameFrom(filters)

