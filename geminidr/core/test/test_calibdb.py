import pytest
from ..primitives_calibdb import _update_datalab

@pytest.fixture()
def ad():
    try:
        from AstroFaker import astrofaker
    except ImportError:
        pytest.skip("astrofaker is not installed")

    simple_ad = astrofaker.create('NIRI', 'IMAGE')
    return simple_ad


def test_update_datalab(ad):
    kw_lut = {'DATALAB': 'comment'}
    kw_datalab = ad._keyword_for('data_label')
    orig_datalab = 'GN-2001A-Q-9-52-001'
    ad.phu[kw_datalab] = orig_datalab
    _update_datalab(ad, '_flat', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '_flat'
    _update_datalab(ad, '_flat', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '_flat'
    _update_datalab(ad, '_bias', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '_bias'
