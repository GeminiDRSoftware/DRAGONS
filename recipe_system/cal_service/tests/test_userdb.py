from tempfile import mkdtemp

import pytest
import astrodata
from recipe_system.cal_service import UserDB, caldb


@pytest.fixture()
def ad():
    astrofaker = pytest.importorskip('astrofaker')
    extra_keywords = {'DATALAB': 'datalabel'}
    return astrofaker.create('NIRI', 'IMAGE', extra_keywords=extra_keywords)


@pytest.fixture()
def userdb(monkeypatch):
    monkeypatch.setattr(caldb, 'CALDIR', mkdtemp())
    user_cals = {}
    valid_caltypes = ['processed_arc']
    return UserDB(user_cals=user_cals, valid_caltypes=valid_caltypes)


def _gen_adinputs():
    astrofaker = pytest.importorskip('astrofaker')
    extra_keywords = {'DATALAB': 'datalabel'}
    return [astrofaker.create('NIRI', 'IMAGE', extra_keywords=extra_keywords)]  # we are just going to check these passed through, so it doesn't have to be real data


def test_set_calibrations(userdb):
    adinputs = _gen_adinputs()
    assert(adinputs[0].calibration_key() == 'datalabel')
    userdb.set_calibrations(adinputs, caltype='processed_arc', calfile="somecal.fits")
    c = userdb.get_calibrations(adinputs, caltype='processed_arc')
    assert(c.files == ['somecal.fits'])


def test_unset_calibrations(userdb):
    adinputs = _gen_adinputs()
    userdb.set_calibrations(adinputs, caltype='processed_arc', calfile="somecal.fits")
    userdb.unset_calibrations(adinputs, caltype='processed_arc')
    c = userdb.get_calibrations(adinputs, caltype='processed_arc')
    assert(c.files == [None])
    assert(c.origins == [None])


def test_clear_calibrations(userdb):
    adinputs = _gen_adinputs()
    userdb.set_calibrations(adinputs, caltype='processed_arc', calfile="somecal.fits")
    userdb.clear_calibrations()
    c = userdb.get_calibrations(adinputs, caltype='processed_arc')
    assert(c.files == [None])
    assert(c.origins == [None])
