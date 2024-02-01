import pytest

import datetime
import itertools

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect

from . import ad_min  # minimum AD fixture


@pytest.mark.ghostspect
@pytest.mark.parametrize('arm,res,caltype', list(itertools.product(*[
    ['BLUE', 'RED'],  # Arm
    ['LO_ONLY', 'HI_ONLY'],  # Res. mode
    ['xmod', 'wavemod', 'spatmod', 'specmod', 'rotmod'],  # Cal. type
])))
def test__get_polyfit_filename(ad_min, arm, res, caltype):
    """
    Checks to make:

    - Provide a set of input (arm, res, ) arguments, see if a/ name is
      returned
    """
    ad_min.phu.set('SMPNAME', res)
    ad_min.phu.set('CAMERA', arm)
    ad_min.phu.set('UTSTART', datetime.datetime.now().time().strftime(
        '%H:%M:%S'))
    ad_min.phu.set('DATE-OBS', datetime.datetime.now().date().strftime(
        '%Y-%m-%d'))

    gs = GHOSTSpect([])
    polyfit_file = gs._get_polyfit_filename(ad_min, caltype)

    assert polyfit_file is not None, "Could not find polyfit file"


@pytest.mark.ghostspect
def test__get_polyfit_filename_errors(ad_min):
    """
    Check passing an invalid calib. type throws an error
    """
    ad_min.phu.set('SMPNAME', 'HI_ONLY')
    ad_min.phu.set('CAMERA', 'RED')
    ad_min.phu.set('UTSTART', datetime.datetime.now().time().strftime(
        '%H:%M:%S'))
    ad_min.phu.set('DATE-OBS', datetime.datetime.now().date().strftime(
        '%Y-%m-%d'))

    gs = GHOSTSpect([])
    polyfit_file = gs._get_polyfit_filename(ad_min, 'not-a-cal-type')
    assert polyfit_file is None, "_get_polyfit_filename didn't return " \
                                 "None when asked for a bogus " \
                                 "model type"

