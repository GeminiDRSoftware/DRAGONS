
import os
import numpy as np
import pytest
import tempfile

from astrodata import nddata
import astrodata
import gemini_instruments

test_path = os.getenv('DRAGONS_TESTDATA', '.')
# naming all fits files for easier legibility in code
GRACES = "N20190116G0054i.fits"
GNIRS = 'N20190206S0279.fits'
GMOSN = "N20110826S0336.fits"
GMOSS = "S20180223S0229.fits"
NIFS = "N20160727S0077.fits"
NIRI = 'N20190120S0287.fits'
F2 = 'S20190213S0084.fits'


# Fixtures for module and class
@pytest.fixture(scope='module')
def setup_module(request):
    print('setup test_astrodata_fits module')

    def fin():
        print('\nteardown test_astrodata_fits module')
    request.addfinalizer(fin)
    return


@pytest.fixture(scope='class')
def setup_astrodatafits(request):
    print('setup TestAstrodataFits')

    def fin():
        print('\nteardown TestAstrodataFits')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_astrodatafits')
class TestAstrodataFits:

    @pytest.mark.parametrize("filename, expected", [
        (GRACES, gemini_instruments.graces.adclass.AstroDataGraces),
        (GNIRS, gemini_instruments.gnirs.adclass.AstroDataGnirs),
        (GMOSN, gemini_instruments.gmos.adclass.AstroDataGmos),
        (GMOSS, gemini_instruments.gmos.adclass.AstroDataGmos),
        (NIFS,  gemini_instruments.nifs.adclass.AstroDataNifs),
        (NIRI, gemini_instruments.niri.adclass.AstroDataNiri),
        (F2, gemini_instruments.f2.adclass.AstroDataF2)
    ])
    def test_is_right_type(self, filename, expected, test_path):
        ad = astrodata.open(os.path.join(test_path, filename))
        assert type(ad) == expected

    @pytest.mark.parametrize("filename, expected", [
        (GRACES, gemini_instruments.graces.adclass.AstroDataGraces),
        (GNIRS, gemini_instruments.gnirs.adclass.AstroDataGnirs),
        (GMOSN, gemini_instruments.gmos.adclass.AstroDataGmos),
        (GMOSS, gemini_instruments.gmos.adclass.AstroDataGmos),
        (NIFS, gemini_instruments.nifs.adclass.AstroDataNifs),
        (NIRI, gemini_instruments.niri.adclass.AstroDataNiri),
        (F2, gemini_instruments.f2.adclass.AstroDataF2)
    ])
    def test_is_right_instance(self, filename, expected, test_path):
        
        ad = astrodata.open(os.path.join(test_path, filename))
        
        # YES, this *can* be different from test_is_right_type above.
        # Metaclasses!
        assert isinstance(ad, expected)

    @pytest.mark.parametrize("filename, expected", [
        (GRACES, (28, 190747)),
        (GNIRS,  (1022, 1024)),
        (GMOSN,  (2304, 1056)),
        (GMOSS,  (4224, 544)),
        (NIFS,   (2048, 2048)),
        (NIRI,   (1024, 1024)),
        (F2,     (1, 2048, 2048))
    ])
    def test_extension_data_shape(self, filename, expected, test_path):
        
        ad = astrodata.open(os.path.join(test_path, filename))
        data = ad[0].data
        
        assert data.shape == expected

    @pytest.mark.parametrize("filename, expected", [
        (GRACES,{'UNPREPARED', 'RAW', 'SPECT', 'GEMINI', 'GRACES'}),
        (GNIRS, {'RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'GNIRS',
                 'UNPREPARED', 'SPECT', 'XD'}),
        (GMOSN, {'RAW', 'GMOS', 'GEMINI', 'NORTH', 'SIDEREAL',
                 'UNPREPARED', 'SPECT', 'MOS'}),
        (GMOSS, {'SOUTH', 'RAW', 'GMOS', 'GEMINI', 'SIDEREAL',
                 'UNPREPARED', 'IMAGE', 'MASK', 'ACQUISITION'}),
        (NIFS,  {'DARK', 'RAW', 'AT_ZENITH', 'NORTH', 'AZEL_TARGET',
                 'CAL', 'UNPREPARED', 'NIFS', 'GEMINI', 'NON_SIDEREAL'}),
        (NIRI,  {'RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'UNPREPARED',
                 'IMAGE', 'NIRI'}),
        (F2,    {'IMAGE', 'F2', 'RAW', 'SOUTH', 'SIDEREAL', 'UNPREPARED',
                 'GEMINI', 'ACQUISITION'})
    ])
    def test_tags(self, filename, expected, test_path):
        
        ad = astrodata.open(os.path.join(test_path, filename))
        tags = ad.tags
        
        assert expected.issubset(tags)

    @pytest.mark.parametrize("filename, expected", [
        (GRACES,'GRACES'),
        (GNIRS, 'GNIRS'),
        (GMOSN, 'GMOS-N'),
        (GMOSS, 'GMOS-S'),
        (NIFS,  'NIFS'),
        (NIRI,  'NIRI'),
        (F2,    'F2')
    ])
    def test_can_return_instrument(self, filename, expected, test_path):
        
        ad = astrodata.open(os.path.join(test_path, filename))

        assert ad.phu['INSTRUME'] == expected
        assert ad.instrument() == ad.phu['INSTRUME']


    @pytest.mark.parametrize("filename, expected", [
        (GMOSN, 3),
        (GMOSS, 12),
        (NIFS,  1)
    ])
    def test_can_return_ad_length(self, filename, expected, test_path):
        
        ad = astrodata.open(os.path.join(test_path, filename))
        
        assert len(ad) == expected

    @pytest.mark.parametrize("filename, expected", [
        (GMOSN, 2),
        (GMOSS, 11),
        (NIFS,  0)
    ])
    def test_slice_range(self, filename, expected, test_path):
        
        ad = astrodata.open(os.path.join(test_path, filename))

        metadata = ('SCI', 2), ('SCI', 3)
        slc = ad[1:]

        assert len(slc) == expected

        for ext, md in zip(slc, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    @pytest.mark.parametrize("filename, expected", [
        (GMOSN, 'GMOS + Red1'),
        (GMOSS, 'GMOS + Hamamatsu_new'),
        (NIFS,  'NIFS')
    ])
    def test_read_a_keyword_from_phu(self, filename, expected, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))

        assert ad.phu['DETECTOR'] == expected

    @pytest.mark.parametrize("filename, expected", [
        (GMOSN, ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']),
        (GMOSS, ['BI5-36-4k-2', 'BI5-36-4k-2', 'BI5-36-4k-2',
                 'BI5-36-4k-2',  'BI11-33-4k-1', 'BI11-33-4k-1',
                 'BI11-33-4k-1', 'BI11-33-4k-1', 'BI12-34-4k-1',
                 'BI12-34-4k-1', 'BI12-34-4k-1', 'BI12-34-4k-1']),
        (NIFS, 'Error should raise!'),
    ])
    def Test_read_a_keyword_from_hdr(self, filename, expected, test_path):

        ad = astrodata.open(os.path.join(test_path, filename))

        try:
            assert ad.hdr['CCDNAME'] == expected

        except KeyError:
            # KeyError only accepted if it's because headers out of range
            assert len(ad) == 1

    # Access to headers: DEPRECATED METHODS
    # These should fail at some point
    @pytest.mark.skip(reason="Deprecated methods")
    def test_read_a_keyword_from_phu_deprecated(self):

        ad = astrodata.open('N20110826S0336.fits')

        with pytest.raises(AttributeError):
            assert ad.phu.DETECTOR == 'GMOS + Red1'

    @pytest.mark.skip(reason="Deprecated methods")
    def test_read_a_keyword_from_hdr_deprecated(self):

        ad = astrodata.open('N20110826S0336.fits')

        with pytest.raises(AttributeError):
            assert ad.hdr.CCDNAME == [
                'EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03'
            ]

    @pytest.mark.skip(reason="Deprecated methods")
    def test_set_a_keyword_on_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AssertionError):
            ad.phu.DETECTOR = 'FooBar'
            ad.phu.ARBTRARY = 'BarBaz'
            assert ad.phu.DETECTOR == 'FooBar'
            assert ad.phu.ARBTRARY == 'BarBaz'
            assert ad.phu['DETECTOR'] == 'FooBar'

    @pytest.mark.skip(reason="Deprecated methods")
    def test_remove_a_keyword_from_phu_deprecated(self):
        ad = astrodata.open('N20110826S0336.fits')
        with pytest.raises(AttributeError):
            del ad.phu.DETECTOR
            assert 'DETECTOR' not in ad.phu

    # Regression:
    # Make sure that references to associated
    # extension objects are copied across
    # Trying to access a missing attribute in
    # the data provider should raise an
    # AttributeError
        test_data_name = "N20110826S0336.fits"
    @pytest.mark.skip(reason="uses chara")
    def test_raise_attribute_error_when_accessing_missing_extenions(self):
        ad = from_chara('N20131215S0202_refcatAdded.fits')
        with pytest.raises(AttributeError) as excinfo:
            ad.ABC

    # Some times, internal changes break the writing capability. Make sure that
    # this is the case, always
    @pytest.mark.skip(reason="uses chara")
    def test_write_without_exceptions(self):
        # Use an image that we know contains complex structure
        ad = from_chara('N20131215S0202_refcatAdded.fits')
        with tempfile.TemporaryFile() as tf:
            ad.write(tf)
