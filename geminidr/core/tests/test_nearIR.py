"""
Tests applied to primitives_nearIR.py
"""

from datetime import datetime
import os

import astrodata
from astrodata.testing import ad_compare
from geminidr.core.tests.test_spect import create_zero_filled_fake_astrodata
from geminidr.core import primitives_nearIR
import pytest


# ad input list maker functions
def make_ads(timestamps):
    def make_ad(filename, timestamp):
        ad = create_zero_filled_fake_astrodata(100, 200)
        ad.filename = filename
        ad.ut_datetime = timestamp
        return ad

    return [make_ad(f"test{idx}", ts) for idx, ts in enumerate(timestamps, start=1)]


# -- Tests --------------------------------------------------------------------


def test_remove_first_frame_by_time():
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


def test_remove_first_frame_by_filename():
    jan = datetime(year=2021, month=1, day=1)
    feb = datetime(year=2021, month=2, day=1)
    mar = datetime(year=2021, month=3, day=1)

    # Remove first frame and second frame by filename
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([jan, feb, mar])
    ad_out = _p.removeFirstFrame(ad_in, remove_files='test2')
    assert len(ad_out) == 1
    assert ad_out[0] == ad_in[2]

    # Remove first frame by filename
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([jan, feb, mar])
    ad_out = _p.removeFirstFrame(ad_in, remove_first=False, remove_files='test1')
    assert len(ad_out) == 2
    assert ad_out[0] == ad_in[1]
    assert ad_out[1] == ad_in[2]

    # Remove first 2 frames by filename
    _p = primitives_nearIR.NearIR([])
    ad_in = make_ads([jan, feb, mar])
    ad_out = _p.removeFirstFrame(ad_in, remove_first=False, remove_files='test1,test2')
    assert len(ad_out) == 1
    assert ad_out[0] == ad_in[2]


# These tests check the observing modes for GNIRS and NIRI for which we've
# confirmed cleanReadout() is able to remove fixed pattern noise. We don't (as
# of 2022-10-25) have an example of it in GNIRS imaging. DB
@pytest.mark.slow
@pytest.mark.regression
def test_clean_readout_gnirs_spec(path_to_inputs, path_to_refs):
    ad = astrodata.open(os.path.join(path_to_inputs,
                                     "S20060826S0305_skyAssociated.fits"))
    p = primitives_nearIR.NearIR([ad])
    ad_out = p.cleanReadout(clean="default")[0]

    ref = astrodata.open(os.path.join(path_to_refs,
                                      "S20060826S0305_readoutCleaned.fits"))
    assert ad_compare(ad_out, ref)


@pytest.mark.slow
@pytest.mark.regression
def test_clean_readout_niri_spec(path_to_inputs, path_to_refs):
    ad = astrodata.open(os.path.join(path_to_inputs,
                                     "N20050614S0190_skyAssociated.fits"))
    p = primitives_nearIR.NearIR([ad])
    ad_out = p.cleanReadout(clean="default")[0]

    ref = astrodata.open(os.path.join(path_to_refs,
                                      "N20050614S0190_readoutCleaned.fits"))
    assert ad_compare(ad_out, ref)

@pytest.mark.slow
@pytest.mark.regression
def test_clean_readout_niri_image(path_to_inputs, path_to_refs):
    ad = astrodata.open(os.path.join(path_to_inputs,
                                     "N20170505S0146_skyAssociated.fits"))
    p = primitives_nearIR.NearIR([ad])
    ad_out = p.cleanReadout(clean="default")[0]

    ref = astrodata.open(os.path.join(path_to_refs,
                                      "N20170505S0146_readoutCleaned.fits"))
    assert ad_compare(ad_out, ref)
