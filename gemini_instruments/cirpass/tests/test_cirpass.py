import pytest
import astrodata

# until we add CIRPASS to astrofaker
from gemini_instruments.cirpass import AstroDataCirpass


def test_ra_dec_from_text():
    ad = AstroDataCirpass()
    ad.phu['TEL_RA'] = '03:48:30.113'
    ad.phu['TEL_DEC'] = '+24:20:43.00'
    assert ad.ra() == 57.12547083333333
    assert ad.dec() == 24.345277777777778


if __name__ == "__main__":
    pytest.main()
