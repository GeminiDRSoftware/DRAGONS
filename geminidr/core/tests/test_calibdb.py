import pytest
from geminidr.core.primitives_calibdb import _update_datalab


astrofaker = pytest.importorskip("astrofaker")


@pytest.fixture()
def ad():
    return astrofaker.create('NIRI', 'IMAGE')


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
