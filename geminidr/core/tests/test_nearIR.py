"""
Tests applied to primitives_nearIR.py
"""

from datetime import datetime

from geminidr.core.tests.test_spect import create_zero_filled_fake_astrodata
from geminidr.core import primitives_nearIR


# -- Tests --------------------------------------------------------------------


def test_remove_first_frame():
    # ad input list maker functions
    def make_ad(timestamp):
        ad = create_zero_filled_fake_astrodata(100, 200)
        ad.ut_datetime = timestamp
        return ad

    def make_ads(timestamps):
        return [make_ad(ts) for idx, ts in enumerate(timestamps)]

    jan = datetime(year=2021, month=1, day=1)
    feb = datetime(year=2021, month=2, day=1)
    mar = datetime(year=2021, month=3, day=1)

    # Simple case, in order
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([jan, feb, mar])
    ad_out = _p.removeFirstFrame(ad_in)
    assert len(ad_out) == 2
    assert ad_out[0] == ad_in[1]
    assert ad_out[1] == ad_in[2]

    # reverse order
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([mar, feb, jan])
    ad_out = _p.removeFirstFrame(ad_in)
    assert len(ad_out) == 2
    assert ad_out[0] == ad_in[0]
    assert ad_out[1] == ad_in[1]

    # jan in middle
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([feb, jan, mar])
    ad_out = _p.removeFirstFrame(ad_in)
    assert len(ad_out) == 2
    assert ad_out[0] == ad_in[0]
    assert ad_out[1] == ad_in[2]

    # duplicate ut_datetime
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([feb, jan, jan, mar])
    ad_out = _p.removeFirstFrame(ad_in)
    assert len(ad_out) == 3
    assert ad_out[0] == ad_in[0]
    assert ad_out[1] == ad_in[2]
    assert ad_out[2] == ad_in[3]
