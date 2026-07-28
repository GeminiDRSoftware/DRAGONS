import pytest

from copy import deepcopy
import itertools

import numpy as np

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect

from . import ad_min  # minimum AD fixture


@pytest.mark.ghostspect
@pytest.mark.parametrize('xbin, ybin',
                         list(itertools.product(*[
                             [1, 2, ],  # x binning
                             [1, 2, 4, 8, ],  # y binning
                         ]))
                         )
def test_darkCorrect_rebin(ad_min, xbin, ybin):
    """
    Checks to make:

    - Ensure re-binning of darks is correctly triggered (i.e. deliberately
      pass science and dark data w/ different binning, ensure no failure)
    - Error mode check: see for error if length of dark list != length of
      science list
    - Check for DARKIM in output header
    - Check before & after data shape
    """
    dark = deepcopy(ad_min)
    dark.filename = 'dark.fits'

    # 'Re-bin' the data file
    ad_min[0].data = np.ones((int(1024 / ybin), int(1024 / xbin),), dtype=np.float32)
    ad_min[0].hdr.set('CCDSUM', '{} {}'.format(xbin, ybin, ))

    gs = GHOSTSpect([])
    input_shape = ad_min[0].data.shape
    ad_out = gs.darkCorrect([ad_min], dark=dark, do_cal="force")[0]

    assert ad_out[0].data.shape == input_shape, "darkCorrect has mangled " \
                                                "the shape of the input " \
                                                "data"


@pytest.mark.ghostspect
def test_darkCorrect_errors(ad_min):
    dark = deepcopy(ad_min)
    dark.filename = 'dark.fits'

    gs = GHOSTSpect([])

    # Passing in data inputs with different binnings
    with pytest.raises(ValueError):
        ad2 = deepcopy(ad_min)
        ad2[0].hdr.set('CCDSUM', '2 2')
        gs.darkCorrect([ad_min, ad2, ], dark=[dark, dark, ], do_cal="force")

    # Mismatched list lengths
    with pytest.raises(Exception):
        gs.darkCorrect([ad_min, ad2, ad_min], dark=[dark, dark, ], do_cal="force")


@pytest.mark.ghostspect
def test_darkCorrect(ad_min):
    dark = deepcopy(ad_min)
    dark.filename = 'dark.fits'

    gs = GHOSTSpect([])
    ad_out = gs.darkCorrect([ad_min], dark=[dark], do_cal="force")

    # import pdb; pdb.set_trace()

    assert ad_out[0].phu.get('DARKIM') == dark.filename, \
        "darkCorrect failed to record the name of the dark " \
        "file used in the output header (expected {}, got {})".format(
            dark.filename, ad_out[0].phu.get('DARKIM'),
        )

    assert ad_out[0].phu.get(
        gs.timestamp_keys['darkCorrect']), "darkCorrect did not " \
                                           "timestamp-mark the " \
                                           "output file"

