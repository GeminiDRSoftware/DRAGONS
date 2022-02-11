import pytest


# Until we add Hokupaa to astrofaker
from gemini_instruments.hokupaa_quirc import AstroDataHokupaaQUIRC


def test_ra_dec_from_text():
    ad = AstroDataHokupaaQUIRC()
    ad.phu['RA'] = '03:48:30.113'
    ad.phu['DEC'] = '+24:20:43.00'
    assert pytest.approx(ad.ra(), 57.12547083333333)
    assert pytest.approx(ad.dec(), 24.345277777777778)


if __name__ == "__main__":
    pytest.main()
