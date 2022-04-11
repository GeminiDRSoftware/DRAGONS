import pytest

from gemini_instruments.texes import AstroDataTexes


@pytest.mark.skip  # skip fornow
def test_ra_dec_from_text():
    ad = AstroDataTexes()
    ad.phu['RA'] = '03:48:30.113'
    ad.phu['DEC'] = '+24:20:43.00'
    assert ad.ra() == pytest.approx(57.12547083333333)
    assert ad.dec() == pytest.approx(24.345277777777778)
    assert ad.target_ra() == pytest.approx(57.12547083333333)
    assert ad.target_dec() == pytest.approx(24.345277777777778)


if __name__ == "__main__":
    pytest.main()
