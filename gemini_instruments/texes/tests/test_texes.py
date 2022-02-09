import pytest


@pytest.mark.skip  # skip fornow
def test_ra_dec_from_text(astrofaker):
    ad = astrofaker.create('TEXES', ['IMAGE'],
                           extra_keywords={'RA': '03:48:30.113',
                                           'DEC': '+24:20:43.00',
                                           'DATE-OBS': '2021-01-01T12:00:00.000'}
                           )
    assert ad.ra() == 57.12547083333333
    assert ad.dec() == 24.345277777777778


if __name__ == "__main__":
    pytest.main()
