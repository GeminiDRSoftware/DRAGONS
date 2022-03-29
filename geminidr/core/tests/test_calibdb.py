import pytest
from geminidr.core.primitives_calibdb import _update_datalab


@pytest.fixture()
def ad(astrofaker):
    return astrofaker.create('NIRI', 'IMAGE')


def test_update_datalab(ad):
    kw_lut = {'DATALAB': 'comment'}
    kw_datalab = ad._keyword_for('data_label')
    orig_datalab = 'GN-2001A-Q-9-52-001'
    ad.phu[kw_datalab] = orig_datalab
    _update_datalab(ad, '_flat', 'sq', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '-SQ-FLAT'
    _update_datalab(ad, '_flat', 'sq', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '-SQ-FLAT'
    _update_datalab(ad, '_bias', 'sq', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '-SQ-BIAS'
