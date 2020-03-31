import pytest
import astrodata
import gemini_instruments


def test_is_right_instance(gmos_files):

    for _file in gmos_files:

        ad = astrodata.open(_file)
        assert isinstance(ad, gemini_instruments.gmos.adclass.AstroDataGmos)


def test_can_return_instrument(gmos_files):

    for _file in gmos_files:

        ad = astrodata.open(_file)

        assert ad.phu['INSTRUME'] in ['GMOS-N', 'GMOS-S']
        assert ad.instrument() == ad.phu['INSTRUME']


def test_can_return_ad_length(gmos_files):

    for _file in gmos_files:

        ad = astrodata.open(_file)

        assert len(ad)


if __name__ == '__main__':

    pytest.main()
