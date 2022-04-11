import pytest

from gemini_instruments.gsaoi import AstroDataGsaoi


def test_ra_dec_from_text():
    ad = AstroDataGsaoi()
    ad.phu['RA'] = '03:48:30.113'
    ad.phu['DEC'] = '+24:20:43.00'
    assert ad.target_ra() == pytest.approx(57.12547083333333)
    assert ad.target_dec() == pytest.approx(24.345277777777778)


if __name__ == "__main__":
    pytest.main()
