"""
Tests applied to primitives_nearIR.py
"""
import pytest

from datetime import datetime
import os

import astrodata
from astrodata.testing import ad_compare
from geminidr.core.tests.test_spect import create_zero_filled_fake_astrodata
from geminidr.core import primitives_nearIR
from recipe_system.mappers.primitiveMapper import PrimitiveMapper


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
# confirmed cleanReadout() is (more-or-less) able to remove fixed pattern noise.
# We don't (as of 2022-10-25) have an example of it in GNIRS imaging. DB
@pytest.mark.regression
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("in_file",
                         ["S20060826S0305",  # GNIRS LS (parameters need tweaks)
                          "N20231112S0137",  # GNIRS LS (spectrum across boundary)
                          "N20050614S0190",  # NIRI LS (parameters need tweaks)
                          "N20170505S0146",  # NIRI image, single star
                          "N20220902S0145",  # NIRI image, extended source
                          "N20051120S0378",  # NIRI image, single star
                          "N20060103S0010",  # NIRI image, star field
                          "N20060218S0138",  # NIRI image, single star
                          "S20060501S0081",  # GNIRS XD spectrum
                          "S20060806S0080",  # GNIRS XD spectrum
                          "S20070131S0105",  # GNIRS XD spectrum
                          ])
def test_clean_readout(in_file, path_to_inputs, path_to_refs):
    ad = astrodata.from_file(os.path.join(path_to_inputs,
                                     in_file + '_skyCorrected.fits'))

    # Must use the correct default parameters, since this is a test that the
    # defaults haven't changed
    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode="sq", drpkg="geminidr")
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    ad_out = p.cleanReadout(clean="default")[0]

    ref = astrodata.from_file(os.path.join(path_to_refs, ad.filename))
    assert ad_compare(ad_out, ref, atol=0.01)


@pytest.mark.regression
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("in_file",
                         ["S20060826S0305",  # GNIRS LS
                          "N20231112S0137",  # GNIRS LS
                          "N20050614S0190",  # NIRI LS
                          "N20170505S0146",  # NIRI image
                          "N20220902S0145",  # NIRI image, extended source
                          "N20051120S0378",  # NIRI image
                          "N20060103S0010",  # NIRI image
                          "N20060218S0138",  # NIRI image
                          "S20060501S0081",  # GNIRS XD spectrum
                          "S20060806S0080",  # GNIRS XD spectrum
                          "S20070131S0105",  # GNIRS XD spectrum  
                          "N20101227S0040",  # GNIRS LS (par needs tweaking pat_thres=0.1). Only FFT can handle this frame.
                          "N20231112S0136",  # GNIRS LS
                          ])
def test_clean_fftreadout(in_file, path_to_inputs, path_to_refs):
    ad = astrodata.from_file(os.path.join(path_to_inputs, in_file + '_skyCorrected.fits'))
    # Must use the correct default parameters, since this is a test that the
    # defaults haven't changed
    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode="sq", drpkg="geminidr")
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])    
    ad_out = p.cleanFFTReadout(clean="default")[0]
    ref = astrodata.from_file(os.path.join(path_to_refs, in_file + '_readoutFFTCleaned.fits'))
    assert ad_compare(ad_out, ref, atol=0.01)
