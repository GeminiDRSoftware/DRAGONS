import pytest

import astrodata
import igrins_instruments

# from igrins_instruments.igrins.tests import test_data
from . import test_data
from importlib.resources import files

dataroot = files(test_data)
sample_mef = list((dataroot / "sample_mef").glob("N*.fits"))

samples = dict((kind, list((dataroot / f"sample_{kind.lower()}").glob("N*.fits")))
               for kind in ["TGT", "STD", "SKY", "FLATOFF", "FLATON"])

allfiles = [fn
            for fnlist in samples.values()
            for fn in fnlist]

def get_ad(request):
    return astrodata.open(request.param)

ad_mef = pytest.fixture(scope="module", params=sample_mef)(get_ad)
ad_any = pytest.fixture(scope="module", params=allfiles)(get_ad)
ad_tgt = pytest.fixture(scope="module", params=samples["TGT"])(get_ad)
ad_std = pytest.fixture(scope="module", params=samples["STD"])(get_ad)
ad_sky = pytest.fixture(scope="module", params=samples["SKY"])(get_ad)
ad_flaton = pytest.fixture(scope="module", params=samples["FLATON"])(get_ad)
ad_flatoff = pytest.fixture(scope="module", params=samples["FLATOFF"])(get_ad)

def _check_tags(ad, expected_tags, unexpected_tags):

    assert not set(expected_tags).difference(ad.tags), f"expected tag NOT found in {ad.filename}"
    assert not set(unexpected_tags).intersection(ad.tags), f"UNexpected tag found in {ad.filename}"


def test_tags_tgt(ad_tgt):
    expected_tags = ['SIDEREAL']
    unexpected_tags = ['STANDARD', 'SKY', 'FLAT']

    _check_tags(ad_tgt, expected_tags, unexpected_tags)


def test_tags_std(ad_std):
    expected_tags = ['SIDEREAL', 'STANDARD']
    unexpected_tags = ['SKY', 'FLAT']

    _check_tags(ad_std, expected_tags, unexpected_tags)


def test_tags_sky(ad_sky):
    expected_tags = ['SIDEREAL', 'SKY']
    unexpected_tags = ['STANDARD', 'FLAT']

    _check_tags(ad_sky, expected_tags, unexpected_tags)


def test_tags_flaton(ad_flaton):
    expected_tags = ['CAL', 'FLAT', 'GCALFLAT', 'LAMPON']
    unexpected_tags = ['SIDEREAL', 'SKY', 'STANDARD', 'LAMPOFF']

    _check_tags(ad_flaton, expected_tags, unexpected_tags)


def test_tags_flatoff(ad_flatoff):
    expected_tags = ['CAL', 'FLAT', 'GCALFLAT', 'LAMPOFF']
    unexpected_tags = ['SIDEREAL', 'SKY', 'STANDARD', 'LAMPON']

    _check_tags(ad_flatoff, expected_tags, unexpected_tags)


def test_tags_common(ad_any):
    expected_tags = ['GEMINI', 'IGRINS', 'IGRINS-2', 'NORTH', 'UNPREPARED']
    unexpected_tags = ['BUNDLE']

    _check_tags(ad_any, expected_tags, unexpected_tags)


def test_tags_mef(ad_mef):
    expected_tags = {'GEMINI', 'IGRINS', 'IGRINS-2', 'NORTH', 'RAW', 'BUNDLE'}
    unexpected_tags = []

    _check_tags(ad_mef, expected_tags, unexpected_tags)
